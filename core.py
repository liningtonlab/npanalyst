import pandas as pd
import numpy as np
from rtree import index

from functools import partial

import os
import json


from concurrent.futures import ProcessPoolExecutor,as_completed

from pathlib import Path
import configparser
from tqdm import tqdm
from collections import defaultdict
from itertools import chain
# data = np.random.rand(1000000,4)*10

# KEYCOLS = ["PrecMz","PrecZ","RetTime"]
# ERRORKEYCOLS = [cn for cn in ERRORCOLS if cn.split('_')[0] in KEYCOLS]
DATACOLS = ["PrecMz","PrecZ","CCS","RetTime","ProdMz"]
ERRORCOLS = [f"{dcol}_low" for dcol in DATACOLS]
ERRORCOLS = ERRORCOLS + [f"{dcol}_high" for dcol in DATACOLS]
ERRORINFO = [('ppm',10),(None,None),('ppm',10000),('window',0.1),('ppm',50),]
DATACOLS = DATACOLS + ['PrecIntensity','ProdIntensity','Ar1','Ar3']
FLOATPREC = 'float64'
FILENAMECOL = "MgfFileName"



def gen_rep_df_paths(datadir):
    """
    generator func which yields concatenated replica file dataframes
    
    Args:
        datadir (str): data directory path
    """

    sd = os.scandir(datadir)
    csvs = [f.name for f in sd if f.name.lower().endswith('.csv')]
    repd = defaultdict(list)
    for fname in csvs:
        repd[fname.split("_")[1]].append(fname) #parse rule.. probably needs to be more flexible

    for sample,files in repd.items():
        if len(files) > 1:
            paths = [os.path.join(datadir,f) for f in files]
            yield (sample, paths)

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

def get_rects(df):
    """
    get the hyperrectangles defined by the error funcs. assumes error cols are present in df.
    
    Args:
        df (pd.DataFrame): datafame with error cols in format <datacol>_low and <datacol>_high
    
    Returns:
        np.array: array of hyperrectangles in format [[x_low,y_low...x_high,y_high]]
    """

    # order = [f"{dcol}_low" for dcol in DATACOLS] + [f"{dcol}_high" for dcol in DATACOLS]
    order = ERRORCOLS
    return df[order].values

def build_rtree(df):
    """
    build an rtree index from a dataframe for fast range queries.
    
    Args:
        df (pd.DataFrame): dataframe with error cols
    
    Returns:
        rtree.Index: a rtree index built from df data
    """

    dims = len(ERRORCOLS) // 2 
    p = index.Property()
    p.dimension = dims
    rgen = ((i,r,None) for i,r in enumerate(get_rects(df)))
    idx = index.Index(rgen,properties=p)
    return idx

def gen_con_comps(rtree,rects,pbar=False):
    """
    Generate connected components subgraphs for a graph where nodes are hyperrectangles
    and edges are overlapping hyperrectangles. This is done using the rtree index and
    a depth first search. 
    
    Args:
        rtree (rtree.Index): rtree index to use
        rects (iterable): array like object of rectangles used to build the rtree
        pbar (bool, optional): Defaults to False. Whether or not to display a progress bar
    """  
    seen = set()
    if pbar:
        to_it = tqdm(range(len(rects)))
    else:
        to_it = range(len(rects))

    for i in to_it:
        if i in seen:
            continue
        else:
            search_idxs = [i]
            c = {i}
            while search_idxs:
                search = search_idxs.pop()
                neighbors = set(rtree.intersection(rects[search]))
                if neighbors.issubset(seen):
                    break
                else:
                    for n in neighbors - seen: #set math
                        c.add(n)
                        search_idxs.append(n)
                        seen.add(n)
            yield c

def _combine_rows(df,cc,calc_basket_info=True):
    cc_df = df.iloc[list(cc)]
    uni_files = set(chain(*[fnames.split("|") for fnames in cc_df[FILENAMECOL]]))
    avgd = list(cc_df[DATACOLS].mean())
    avgd.append('|'.join(uni_files))
    if calc_basket_info:
        basket_info = {cn:[float(cc_df[cn].min()),float(cc_df[cn].max())] for cn in DATACOLS}
        basket_info['n'] = len(cc_df)
        avgd.append(json.dumps(basket_info))
    return avgd

def proc_con_comps(ccs,df,min_reps=2,calc_basket_info=True):
    """
    Takes the connected components from the overlapping hyperrectangles and averages (mean)
    the data values from which the error was generated. Unique filenames are concatenated with a 
    '|' delimiter. Only subgraphs with multiple nodes are used and further filtered for only those 
    which come from at least `min_reps` unique files.
    
    
    Args:
        ccs (set): connected component subgraph dataframe indices
        df (pd.DataFrame): the dataframe which the connected component subgraphs were calculated from
        min_reps (int, optional): Defaults to 2. Minimum number of files needed in subgraph to be used
        calc_basket_info(bool, optional): Defaults to True. Whether or not to include json basket info. 
    
    Returns:
        pd.DataFrame: newdata frame with data cols and file name col.
    """

    data = []
    for cc in ccs:
        if len(cc) > 1:
            cc_df = df.iloc[list(cc)]
            try:
                uni_files = set(cc_df[FILENAMECOL])
            except KeyError as e:
                uni_files= set(cc_df["Sample"])
            if len(uni_files) >= min_reps:
                avgd = _combine_rows(df,cc,calc_basket_info)
                data.append(avgd)
    if calc_basket_info:
        ndf = pd.DataFrame(data,columns=DATACOLS+[FILENAMECOL]+['BasketInfo'])
    else:
        ndf = pd.DataFrame(data,columns=DATACOLS+[FILENAMECOL])
    return ndf

def proc_con_comps_basket(ccs,df,calc_basket_info=True):
    """
    Takes the connected components from the overlapping hyperrectangles and averages (mean)
    the data values from which the error was generated. Unique filenames are concatenated with a 
    '|' delimiter. This is used for basketing accross multiple, pre-replicated files. Single node
    subgraphs are allowed.
    
    
    Args:
        ccs (set): connected component subgraph dataframe indices
        df (pd.DataFrame): the dataframe which the connected component subgraphs were calculated from
        calc_basket_info(bool, optional): Defaults to True. Whether or not to include json basket info. 
    
    Returns:
        pd.DataFrame: new dataframe with data cols and file name col.
    """

    data = [_combine_rows(df,cc,calc_basket_info) for cc in ccs]

    if calc_basket_info:
        ndf = pd.DataFrame(data,columns=DATACOLS+[FILENAMECOL]+['BasketInfo'])
    else:
        ndf = pd.DataFrame(data,columns=DATACOLS+[FILENAMECOL])
    return ndf

def _proc_one(sample,df_paths,calc_basket_info=False):
    """
    Process one replica sample. The replicated file is saved as ./Replicated/<sample>_Replicated.csv
    
    Args:
        sample (str): sample name
        df_paths (list): list of paths to replica files to be loaded
    
    Returns:
        str: "DONE" when completed
    """
    dfs = [pd.read_csv(p) for p in df_paths]
    df = pd.concat(dfs,sort=True)
    gen_error_cols(df)
    rtree = build_rtree(df)
    con_comps = gen_con_comps(rtree,get_rects(df))
    ndf = proc_con_comps(con_comps,df,calc_basket_info)
    ndf.to_csv(os.path.join("Replicated",f"{sample}_Replicated.csv"))
    return "DONE"

def proc_folder(datadir,calc_basket_info,max_workers):
    """process a folder of sample data replicates. output files will be saved in ./Replacted
    
    Args:
        datadir (str): data directory of sample replicates
    """

    try:
        os.mkdir('Replicated')
    except OSError:
        pass
    paths = list(gen_rep_df_paths(datadir))
    for sample,df in tqdm(paths,desc='proc_folder'):
        _proc_one(sample,df,calc_basket_info)

def _update(pbar,future):
    '''callback func for future object to update progress bar'''
    pbar.update()
 
def mp_proc_folder(datadir,calc_basket_info=False,max_workers=0):
    """
    multi proccesor version of proc_folder. by default will use cpu_count - 2 workers.  
    
    process a folder of sample data replicates. output files will be saved in ./Replacted
    
    Args:
        datadir (str): data direcory of sample replicates
        calc_basket_info (bool,optional): Defaults to False. Bool on whether or not to save bin info as json strings. 
        max_workers (int, optional): Defaults to None. If provided will use that many workers for processing. If there is limited system memory this might be good to set low. 
    """


    try:
        os.mkdir('Replicated')
    except OSError:
        pass

    if max_workers == 0:
        max_workers = os.cpu_count() - 2

    paths = list(gen_rep_df_paths(datadir))
    pbar = tqdm(desc='proc_folder',total=len(paths))
    samples_left = len(paths)
    paths_iter = iter(paths)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {}
        while samples_left:
                        
            for sample,paths in paths_iter:
                fut = ex.submit(_proc_one,sample,paths,calc_basket_info)
                fut.add_done_callback(partial(_update,pbar))
                futs[fut] = sample
                if len(futs) > max_workers:
                    break

            for fut in as_completed(futs):
                _ = fut.result()
                del futs[fut]
                samples_left -= 1
                break
    
def make_repdf(datadir):
    """
    Make a concatonated dataframe from all the replicated data files.
    Assumes filenames end in 'Replicated.csv'    
    Args:
        datadir (str): the directory with the replicated data files.
    
    Returns:
        pd.DataFrame: a dataframe with all replicate files concatenated
    """

    sd = os.scandir(datadir)
    csvs = [f.name for f in sd if f.name.lower().endswith('replicated.csv')]
    dfs = [pd.read_csv(os.path.join(datadir,f)) for f in csvs]
    return pd.concat(dfs)

def basket(datadir):
    """
    Basket all the replicates in a directory in to a single file called Baskted.csv in datadir
    Unique file names are kept and deliminated with a '|'   
    
    Args:
        datadir (str): the directory of replicated files. 
    """

    print('Loading Rep Files')
    df = make_repdf(datadir)
    gen_error_cols(df)
    print("Making Rtree Index")
    rtree = build_rtree(df)
    print('Generating Baskets')
    con_comps = gen_con_comps(rtree,get_rects(df),pbar=True)
    ndf = proc_con_comps_basket(con_comps,df)
    ndf.to_csv(os.path.join(datadir,"Basketed_LooseRT_RogerOnly.csv"))
    
def filename2sample(filename, fn_delim='_', sampleidx=1):
    sample = filename.split(fn_delim)[sampleidx]
    return sample

def filenames2samples(filenames, delim='|', fn_delim='_', sampleidx=1):
    samples = {filename.split(fn_delim)[sampleidx] for filename in filenames.split(delim)}
    return samples

def synth_fp(fpd,samples):
    to_cat = get_fps(fpd,samples)
    return np.vstack(to_cat).mean(axis=0)
    
def get_fps(fpd,samples):
    to_cat = []
    for samp in samples:
        try:
            to_cat.append(fpd[samp])
        except KeyError :
            pass
    if not to_cat:
        raise KeyError("No Fingerprints found...")
    return to_cat

from scipy.spatial.distance import pdist
def cluster_score(fpd,samples):
    fps = get_fps(fpd,samples)
    cubed = pdist(np.vstack(fps),metric='correlation')**3
    return cubed.mean()
    

# def mp_pairwise_score():
    # pass


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument('path',help='path to input for either task')
#     parser.add_argument('task',help='task to perform.',choices=['replicate','basket','both'])
#     parser.add_argument('-w','--workers',help="number of parallel workers to spin up",type=int,default=0)
#     parser.add_argument('--basket_info',help='Flag to save basket info as a json object in resulting files.',action='store_true')
#     args = parser.parse_args()

#     if args.task in ['replicate','both']:
#         data_path = Path(args.path)
#         mp_proc_folder(data_path,max_workers=args.workers,calc_basket_info=args.basket_info)

#     elif args.task in ['basket','both']:
#         if args.task == 'both':
#             data_path = 'Replicated'
#         else:
#             data_path = Path(args.path)
#         basket(data_path)
    


# '/home/cpye/pydev/hifan/test_data/RL_002_WPD_CSV'

# print("Processing Roger Replicates")


# print("Processing John Replicates")
# data_path = Path('/home/cpye/pydev/hifan/test_data/HIFAN_JM_001_CSV_WPD25_5')
# mp_proc_folder(data_path)

# print("Basketing")
# data_path = Path('/home/cpye/pydev/hifan/metabolate/Replicated')
# basket(data_path)






#############################
# OLD STUFF PROBABLY BROKEN #
#############################


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



# def match(row,chunk):
#     ev = row[ERRORCOLS].values
#     v = row[DATACOLS]
#     i  = 1
#     while  np.all(v < ev[:,1]):
#         cev = chunk.loc[i][ERRORCOLS]
#         if np.all((v>cev[:,0]) & (v<cev[:,1])):
        

    
    




# sd = os.scandir(data_path)
# frames = [pd.read_csv(data_path.joinpath(f)) for f in sd]
# df = pd.concat(frames, ignore_index=True)

# df[DATACOLS] = df[DATACOLS].astype(FLOATPREC)

# data = df[DATACOLS].values

# def build_map(d,ks):
#     d = d.tolist()
#     dd = defaultdict(list)
#     for v,k in zip(d,ks):
#         dd[k].append(v)
#     return dict(dd)

# floored = np.floor(df[KEYCOLS].values).astype(int).astype(str)
# keys = ['_'.join(f) for f in floored]
# unibins = set(keys)
# df['sequence'] = keys
# kd = build_map(data,keys)



# def gen_windows(data):
#     ncols = data.shape[1]
#     new_dims = list(data.shape) + [2]
#     output = np.zeros(new_dims)
#     for i in range(ncols):
#         output[:,i,0] = data[:,i] - 1
#         output[:,i,1] = data[:,i] + 1
#     return output



        
