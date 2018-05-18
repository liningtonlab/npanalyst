import pandas as pd
import numpy as np
from IPython import embed

from rtree import index


import os
from pathlib import Path
import configparser

from concurrent.futures import ProcessPoolExecutor,as_completed

from collections import defaultdict
from itertools import chain
from tqdm import tqdm
from numba import jitclass, jit, float32,float64,int32
# data = np.random.rand(1000000,4)*10

KEYCOLS = ["PrecMz","PrecZ","RetTime","CCS"]
DATACOLS = ["PrecMz","PrecZ","CCS","RetTime","ProdMz","Ar1"]#,"PrecIntensity"]
MZ1DATACOLS = ["PrecMz","PrecZ","CCS","RetTime"]
MZ2DATACOLS = DATACOLS

ERRORCOLS = list(chain(*[[f"{dcol}_low",f"{dcol}_high"] for dcol in DATACOLS])) #insane line to get a list of <val>_low <val>_high names
ERRORCOLS.sort(key=lambda x: list(x.split("_"))[::-1],reverse=True) #a more insane line to get them in the order <x>_low, <y>_low, <x>_high, <y>_high

ERRORKEYCOLS = [cn for cn in ERRORCOLS if cn.split('_')[0] in KEYCOLS]
ERRORINFO = {
    "PrecMz":('ppm',10),
    "PrecZ":(None,None),
    "CCS":('ppm',5000),
    "RetTime":('window',0.03),
    "ProdMz":('ppm',50),
    "PrecIntensity":('window',10),
    "Ar1":('window',0.33)
    }
FLOATPREC = 'float64'

spec = [
    ('ev',float64[:,:]),
    ('v',float64[:]),
    ('idxs',int32[:]),
    ]
@jitclass(spec)
class MZ1Feature(object):
    def __init__(self,idxs,ev,v):
        self.ev = ev
        self.v = v
        self.idxs =idxs
    def eq(self,other):
        eqboth = np.all((other.ev[:,0] <= self.ev[:,1]) & (self.ev[:,1] <= other.ev[:,1]))
        if eqboth:
            return True
        else:
            return False

    def add(self,other):
        nev = np.mean([self.ev,other.ev],axis=0)
        nv = np.mean([self.v,other.v],axis=0)
        idxs = np.concatenate(self.idxs, other.idxs)

        return MZ1Feature(idxs,nev,nv)


def gen_error_cols(df,datacols=DATACOLS):
    """
    Uses the global ERRORINFO and DATACOLS lists to generate
    error windows for each of the DATACOLS. Mutates dataframe inplace for
    some memory conservation.
    
    Args:
        df (pandas.DataFram): input dataframe to calc error windows (modified in place)
    """

    for dcol in datacols:
        einfo = ERRORINFO[dcol]
        col = df[dcol]
        etype, evalue = einfo
        if etype == 'ppm':
            efunc = lambda x:x*(evalue*1e-6)
        elif etype == 'window':
            efunc = lambda x:evalue
        elif etype is None:
            efunc = lambda x:0
        errors = col.apply(efunc)
        df[f"{dcol}_low"] = df[dcol] - errors
        df[f"{dcol}_high"] = df[dcol] + errors

def get_replicate_files(folder):
    data_path = Path(folder)
    sd = os.scandir(data_path)
    replicates = defaultdict(list)
    for f in sd:
        sample = str(f).split("_")[1]
        if sample.lower() != "blank":
            pf = data_path.joinpath(f)
            replicates[sample].append(pf)
    replicates = dict(replicates)
    todel = [key for key,val in replicates.items() if len(val)<2]
    for key in todel: del replicates[key]
    return replicates

def load_replicates(sample,files):
    frames = []
    for i,f in enumerate(files):
        frame = pd.read_csv(f)
        frame['fileno']=i
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    return sample,df

def build_spatial_idx(rectangles):
    p = index.Property()
    p.dimension = rectangles.shape[1] // 2
    # print('building index')
    stream = ((i,tuple(coords),None) for i,coords in enumerate(rectangles))
    idx = index.Index(stream, properties=p)
    return idx


def get_intersections(rectangles,min_intersection=0,show_pbar=False):
    idx = build_spatial_idx(rectangles)
    if show_pbar:
        pbar = tqdm(total=rectangles.shape[0],desc="finding intersections")
    
    intersections = set()
    for coords in rectangles:
        inter = tuple(idx.intersection(coords))
        if len(inter) >= min_intersection:
            intersections.add(inter)
        if show_pbar:
            pbar.update()
    if show_pbar:
        pbar.close()

    intersections = sorted(list(intersections))
    return intersections


@jit(cache=True)
def numba_take2d(ary,idxs):
    shape = (len(idxs),ary.shape[1])
    res = np.zeros(shape)
    for i,idx in enumerate(idxs):
        res[i] = ary[idx]
    return res

@jit(cache=True)
def numba_take1d(ary,idxs):
    shape = len(idxs)
    res = np.zeros(shape)
    for i,idx in enumerate(idxs):
        res[i] = ary[idx]
    return res

@jit(cache=True)
def numba_get_replicates(filecol,intersections,minfiles=2):
    mask = np.zeros(len(intersections))
    for i,idxs in enumerate(intersections):
        filenos = numba_take1d(filecol,idxs)
        if np.unique(filenos).shape[0] >= minfiles:
            mask[i] = 1
    return mask.astype('bool')


def joiner(x):
    if str(x.dtype) in {'float64','int64'}:
        return x.mean()
    else:
        return pd.Series(x.unique()).str.cat(sep=";")

def reduce_df(df,replica_idxs):
    repdf = df.loc[[x for tup in replica_idxs for x in tup]].copy()
    repdf['group'] = [i for i,tup in enumerate(replica_idxs) for _ in tup]
    fdf = repdf.groupby('group').agg(joiner)
    return fdf

def basket_df(df,basket_idxs):
    non_keys = [c for c in df.columns if c not in KEYCOLS]
    df[non_keys] =df[non_keys].astype(str)
    basket_membership = {idx:i for i,tup in enumerate(basket_idxs) for idx in tup}
    group_col = []
    for i in range(df.shape[0]):
        try:
            group_col.append(f"b{basket_membership[i]}")
        except KeyError:
            group_col.append(f"s{i}")
    df['group'] = group_col
    fdf = df.groupby('group').agg(joiner)
    return fdf


def run_folder_rep(folder):
    replicates = get_replicate_files(folder)
    for sample,files in tqdm(replicates.items()):
        process_rep(sample,files)
        
def prun_folder_rep(folder):
    replicates = get_replicate_files(folder)
    with tqdm(total=len(replicates),desc="Replica Comparison") as pbar:
        with ProcessPoolExecutor(max_workers=15) as executor:
            futs = [executor.submit(process_rep,sample,files) for sample,files in replicates.items()]
            for fut in as_completed(futs):
                fut.result()
                pbar.update()

def process_rep(sample,files):
    sample,df = load_replicates(sample,files)
    NUMERICCOLS = list(df._get_numeric_data().columns)
    df[NUMERICCOLS] = df[NUMERICCOLS].astype(FLOATPREC)
       
    gen_error_cols(df)
    evals = df[ERRORCOLS].values
    intersections = get_intersections(evals,min_intersection=1)
    mask = numba_get_replicates(df.fileno.values,intersections)
    replicas = np.asarray(intersections)[mask]
    fdf = reduce_df(df,replicas)
    fdf.to_csv(f'./results/{sample}_replicated.csv')
    
    return f"{sample} DONE"

COLSTOKEEP = KEYCOLS + ['ProdMz','ProdIntensity','Sample']

def process_basket(folder):
    data_path = Path(folder)
    sd = [f for f in os.scandir(data_path) if f.is_file() and f.name.lower().endswith(".csv")]
    frames = []
    for file in tqdm(sd,desc='loading data'):
        frames.append(pd.read_csv(file))
    df = pd.concat(frames,ignore_index=True)
    df = df[COLSTOKEEP]
    gen_error_cols(df,KEYCOLS)
    error_cols= list(chain(*[[f"{dcol}_low",f"{dcol}_high"] for dcol in KEYCOLS]))
    error_cols.sort(key=lambda x: list(x.split("_"))[::-1],reverse=True)
    evals = df[error_cols].values
    intersections = get_intersections(evals,min_intersection=0,show_pbar=True)
    # embed()
    bdf = basket_df(df,intersections)
    bdf.to_csv("Basketed.csv")




    
        
  



if __name__ == "__main__":
    import sys
    import warnings
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--replicate',help='folder to perform replicate analysis')
    parser.add_argument('-b','--basket',help='folder to perform basketing')
    args = parser.parse_args()

    if args.replicate:  
        try: 
            os.mkdir('results')
        except FileExistsError:
            "Warning, contents of results will be overwritten..."

        with warnings.catch_warnings():
            prun_folder_rep(args.replicate)
    
    if args.basket:

            process_basket(args.basket)


# embed()



# gbl = list(df.groupby('sequence'))

# new_rows = []
# for idx,chunk in tqdm(gbl):
#     for ridx,row in chunk.iterrows():
#         rmatch =  match(row,chunk)
#         if rmatch:
#             new_rows.append(rmatch) 