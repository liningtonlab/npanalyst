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
DATACOLS = ["PrecMz","PrecZ","CCS","RetTime","ProdMz"]
MZ1DATACOLS = ["PrecMz","PrecZ","CCS","RetTime"]
MZ2DATACOLS = DATACOLS
ERRORCOLS = list(chain(*[[f"{dcol}_low",f"{dcol}_high"] for dcol in DATACOLS]))
ERRORKEYCOLS = [cn for cn in ERRORCOLS if cn.split('_')[0] in KEYCOLS]
ERRORINFO = [('ppm',10),(None,None),('ppm',10000),('window',0.03),('ppm',50),]
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


def build_map(d,ks):
    d = d.tolist()
    dd = defaultdict(list)
    for v,k in zip(d,ks):
        dd[k].append(v)
    return dict(dd)


def gen_error_cols(df):
    """
    Uses the global ERRORINFO and DATACOLS lists to generate
    error windows for each of the DATACOLS. Mutates dataframe inplace for
    some memory conservation.
    
    Args:
        df (pandas.DataFram): input dataframe to calc error windows (modified in place)
    """

    for einfo,dcol in zip(ERRORINFO,DATACOLS):
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


# def get_neigbor_bins(ev,v):
#     ev = ev.reshape(len(KEYCOLS),2)
#     floored = np.floor(v.astype(FLOATPREC))
#     downs = ev[:,0] <= floored
#     ups = ev[:,0] <= floored
#     nbins = []
#     for i,bools in enumerate(zip(downs,ups)):
#         upbool,downbool = bools
#         if upbool:
#             new = floored.copy()
#             new[i] +=1
#             new = '_'.join(new.astype(str))
#             if new in unibins:
#                 nbins.append(new)
#         if downbool:
#             new = floored.copy()
#             new[i] -=1
#             new = '_'.join(new.astype(str))
#             if new in unibins:
#                 nbins.append(new)
#     return nbins


# def is_match(row1,row2):
#     ev1 = row1[ERRORCOLS].values.reshape(len(DATACOLS),2)
#     ev2 = row2[ERRORCOLS].values.reshape(len(DATACOLS),2)
#     if np.all((ev2[:,0] <= ev1[:,1]) & (ev1[:,1] <= ev2[:,1])):
#         return True
#     else:
#         # print((ev2[:,0] <= ev1[:,1]) & (ev1[:,1] <= ev2[:,1]))
#         return False

# def combine(row1,row2):
#     nrow = row1.copy()
#     nrow[NUMERICCOLS] = (row1[NUMERICCOLS].values + row2[NUMERICCOLS].values) /2
#     for col in STRCOLS:
#         nrow[col] = ";".join((row1[col],row2[col]))
#     return nrow


# def match(row,chunk):
#     for idx,crow in chunk.iterrows():
#         if is_match(row,crow):
#             row = combine(row,crow)
#     if row['count'] > 1:
#         return row




# def gen_windows(data):
#     ncols = data.shape[1]
#     new_dims = list(data.shape) + [2]
#     output = np.zeros(new_dims)
#     for i in range(ncols):
#         output[:,i,0] = data[:,i] - 1
#         output[:,i,1] = data[:,i] + 1
#     return output


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
    for key,val in replicates.items():
        if len(val) < 2:
            del replicates[key]
    return replicates

def load_replicates(sample,files):
    frames = []
    for i,f in enumerate(files):
        frame = pd.read_csv(f)
        frame['fileno']=i
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    return sample,df

def gen_replicate_df(replicates):
    for sample,files in replicates.items():
        sample,df = load_replicates(sample,files)
        yield (sample,df)


        
# data_path = Path('/home/cpye/pydev/hifan/test_data')
# sd = os.scandir(data_path)
# frames = []
# for i,f in enumerate(sd):
#     frame = pd.read_csv(data_path.joinpath(f))
#     frame['fileno']=i
#     frames.append(frame)

# df = pd.concat(frames, ignore_index=True)


# df[DATACOLS] = df[DATACOLS].astype(FLOATPREC)


# NUMERICCOLS = list(df._get_numeric_data().columns)
# STRCOLS = [c for c in df.columns if c not in NUMERICCOLS]

# data = df[DATACOLS].values

# tofloor = df[KEYCOLS]
# tofloor['PrecMz'] *10
# floored = np.floor(tofloor[KEYCOLS].values).astype(int).astype(str)
# keys = ['_'.join(f) for f in floored]
# unibins = set(keys)
# df['sequence'] = keys
# df['count'] = 1
# gen_error_cols(df)


# df = df.loc[:1000]
# idxs = df.index.values
# evs = df[ERRORCOLS].values.reshape((idxs.shape[0],len(DATACOLS),2))
# vs = df[DATACOLS].values
# numerics = df[NUMERICCOLS].values
# strs = df[STRCOLS].values

# feats = [MZ1Feature(np.array([idx],dtype='int32'),ev.reshape((len(DATACOLS),2)),v) for idx,ev,v in zip(idxs,evs,vs)]

# evals = df[ERRORCOLS].values



def get_intersections(rectangles,show_pbar=False):
    p = index.Property()
    p.dimension = rectangles.shape[1] // 2
    # print('building index')
    stream = ((i,tuple(coords),None) for i,coords in enumerate(rectangles))
    idx = index.Index(stream, properties=p, interleaved=False)
    intersections = set()
    if show_pbar:
        pbar = tqdm(total=rectangles.shape[0])
    
    # for coords in tqdm(rectangles.tolist(),desc="Finding Overlap"):
    for coords in rectangles:
        inter = tuple(idx.intersection(coords))
        if len(inter) > 1:
            intersections.add(inter)
        if show_pbar:
            pbar.update()
    if show_pbar:
        pbar.close()
    intersections = sorted(list(intersections))
    return intersections




# intersections = get_intersections(evals)


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

def avg_numrows(ary,idxs):
    return numba_take2d(ary,idxs).mean(axis=0)

def append_strings(array):
    strings = []
    for col in array.T:
        col = list(set(col))
        strings.append(';'.join(col))
    return strings

def joiner(x):
    if str(x.dtype) in {'float64','int64'}:
        return x.mean()
    else:
        return pd.Series(x.unique()).str.cat(sep=";")
        # return x.str.cat(sep=';')

def reduce_df(df,replica_idxs,STRCOLS,NUMERICCOLS):
    repdf = df.loc[[x for tup in replica_idxs for x in tup]].copy()
    repdf['group'] = [i for i,tup in enumerate(replica_idxs) for _ in tup]
    # repdf[NUMERICCOLS] = repdf[NUMERICCOLS].astype(FLOATPREC)
    # numcols = []
    # strcols = []
    # for idxs in replica_idxs:
    #     numcols.append(
    #         repdf[NUMERICCOLS].loc[list(idxs)].values.mean(axis=0)
    #     )
    #     strcols.append(
    #         append_strings(repdf[STRCOLS].loc[list(idxs)].values)
    #     )
    # ndf = pd.DataFrame(numcols,columns=NUMERICCOLS)
    # sdf = pd.DataFrame(strcols,columns=STRCOLS)
    # fdf = ndf.merge(sdf,left_index=True,right_index=True)
    fdf = repdf.groupby('group').agg(joiner)
    return fdf



def run_folder_rep(folder):
    replicates = get_replicate_files(folder)
    for sample,df in tqdm(gen_replicate_df(replicates),total=len(replicates)):
        process(sample,df)
        
def prun_folder_rep(folder):
    replicates = get_replicate_files(folder)
    with tqdm(total=len(replicates)) as pbar:
        with ProcessPoolExecutor(max_workers=15) as executor:
            futs = [executor.submit(process,sample,files) for sample,files in replicates.items()]
            for fut in as_completed(futs):
                fut.result()
                pbar.update()


def process_rep(sample,files):
    sample,df = load_replicates(sample,files)
    NUMERICCOLS = list(df._get_numeric_data().columns)
    STRCOLS = [c for c in df.columns if c not in NUMERICCOLS]
    
    # STRCOLS = ["Sequence","Mode","MgfFileName"]
    # NUMERICCOLS = [c for c in df.columns if c not in STRCOLS]
    
    # df[NUMERICCOLS] = df[NUMERICCOLS].fillna(0)
    df[NUMERICCOLS] = df[NUMERICCOLS].astype(FLOATPREC)

    gen_error_cols(df)
    evals = df[ERRORCOLS].values
    intersections = get_intersections(evals)

    mask = numba_get_replicates(df.fileno.values,intersections)
    replicas = np.asarray(intersections)[mask]
    fdf = reduce_df(df,replicas,STRCOLS,NUMERICCOLS)
    fdf.to_csv(f'./results/{sample}_replicated.csv')
    
    return f"{sample} DONE"


def process_basket(folder):
    data_path = Path(folder)
    sd = os.scandir(data_path)
    frames = []
    for file in sd:
        
    



if __name__ == "__main__":
    import sys
    import warnings
    script, folder = sys.argv
    
    try: 
        os.mkdir('results')
    except FileExistsError:
        "Warning, contents of results will be overwritten..."



    with warnings.catch_warnings():
        prun_folder(folder)


# embed()



# gbl = list(df.groupby('sequence'))

# new_rows = []
# for idx,chunk in tqdm(gbl):
#     for ridx,row in chunk.iterrows():
#         rmatch =  match(row,chunk)
#         if rmatch:
#             new_rows.append(rmatch) 