import configparser
import os
import json
import gc
import math
import statistics

from collections import defaultdict,namedtuple
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import partial
from itertools import chain
from pathlib import Path


import numpy as np
import pandas as pd
from rtree import index
from scipy.spatial.distance import pdist
from tqdm import tqdm

# from numba import jit
from IPython import embed



#partially finished classes, not used currently
class Interval(object):
    __slots__ = ('low','high','center')
    def __init__(self,low,high,center=None):
        self.low = low
        self.high = high
        if center:
            self.center = center
        else:
            self.center = statistics.mean((low,high))

    def __eq__(self,other):
        if other.low <= self.high or other.high >= self.low:
            return True
        return False

    def __lt__(self,other):
        if self.high < other.low:
            return True
        return False

    def __gt__(self,other):
        if other.high < self.low:
            return True
        return False
    
    def __hash__(self):
        return hash((self.low,self.high))

    def  __repr__(self):
        return f"<{self.__class__.__name__}(low={self.low},high={self.high}) at {id(self)}>"
    
    def __str__(self):
        return f"{self.low}<>{self.high}"



class PrecIon(object):
    __slots__ = ('rt','z','mz','ccs','ms2','info')
    def __init__(self,rt,z,mz,ccs,ms2):
        self.rt = rt
        self.z = z
        self.mz = mz
        self.ccs = ccs
        self.ms2 = ms2
    

    def merge(self,other):
        pass




def _make_error_col_names(qcols):
    '''helper func to make error column names of the form
    <col_name>_low ... <col_name>_high
    
    Args:
        qcols (iterable): an iterable of column names used for matching
    
    Returns:
        list: list of error col names in non-interleaved order
    '''
    error_cols = [f"{dcol}_low" for dcol in qcols]
    error_cols = error_cols + [f"{dcol}_high" for dcol in qcols]
    return error_cols



def _load_config(config_path=None):
    '''loads the config_path config file and stores a bunch of values as globals
        config_path (str, optional): Defaults to 'defualt.cfg'. 
            path to the config file, default can be overridden. 
    '''    
    config = configparser.ConfigParser()
    config.optionxform = str #make sure things don't get lowercased
    if config_path is None:
        p = Path(__file__).resolve().parent.parent.joinpath('default.cfg')
        config.read(str(p))
    else:
        config.read(config_path)

    global MS1COLS
    MS1COLS = config['MSFileInfo']['MS1Cols'].split(',')
    global MS1COLSTOMATCH
    MS1COLSTOMATCH = config['MSFileInfo']['MS1ColsToMatch'].split(',')
    global MS2COLS
    MS2COLS = config['MSFileInfo']['MS2Cols'].split(',')
    global MS2COLSTOMATCH
    MS2COLSTOMATCH = config['MSFileInfo']['MS2ColsToMatch'].split(',')
    global ERRORINFO
    ERRORINFO = {}
    for name,tup in config['Tolerances'].items():
        etype,ev = tup.split(',')
        if etype == 'None':
            etype = None
        if ev == 'None':
            ev = None
        else:
            ev = float(ev)
        ERRORINFO[name] = (etype,ev)
    global FILENAMECOL
    FILENAMECOL = config['MSFileInfo']['FileNameCol']
    global MS1ERRORCOLS
    MS1ERRORCOLS = _make_error_col_names(MS1COLSTOMATCH)
    global MS2ERRORCOLS
    MS2ERRORCOLS = _make_error_col_names(MS2COLSTOMATCH)

_load_config() #pull config info into global namespace


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

def gen_error_cols(df,qcols=MS1COLSTOMATCH):
    """
    Uses the global ERRORINFO dict to generate
    error windows for each of the qcols. 
    Mutates dataframe inplace for some memory conservation.
    possible error types are:
    * ppm - parts per million
    * perc - percentage
    * factor - a multiplier (ie 10 = 10x)
    * window - a standard fixed error window
    
    Args:
        df (pandas.DataFrame): input dataframe to calc error windows (modified in place)
    """

    for dcol in qcols:
        einfo = ERRORINFO[dcol]
        col = df[dcol]
        etype, evalue = einfo
        if etype == 'ppm':
            efunc = lambda x:x*(evalue*1e-6)
        if etype == 'perc':
            efunc = lambda x:x*(evalue/100)
        if etype == 'factor':
            efunc = lambda x:x*evalue
        if etype == 'window':
            efunc = lambda x:evalue
        if etype is None:
            efunc = lambda x:0
        errors = col.apply(efunc)
        df[f"{dcol}_low"] = df[dcol] - errors
        df[f"{dcol}_high"] = df[dcol] + errors

def get_rects(df,errorcols=MS1ERRORCOLS):
    """
    get the hyperrectangles defined by the error funcs. assumes error cols are present in df.
    
    Args:
        errorcols (iterable): the error cols to make rectangles from
        df (pd.DataFrame): datafame with error cols in format <datacol>_low and <datacol>_high
    
    Returns:
        np.array: array of hyperrectangles in format [[x_low,y_low...x_high,y_high]]
    """
    order = errorcols
    return df[order].values

def build_rtree(df,errorcols=MS1ERRORCOLS):
    """
    build an rtree index from a dataframe for fast range queries.
    dataframe needs to have error cols pre-calced
    
    Args:
        df (pd.DataFrame): dataframe with error cols
    
    Returns:
        rtree.Index: a rtree index built from df rectangles
    """

    dims = len(errorcols) // 2 
    p = index.Property()
    p.dimension = dims
    p.interleaved = False
    rgen = ((i,r,None) for i,r in enumerate(get_rects(df,errorcols)))
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
                for n in neighbors - seen: #set math
                    c.add(n)
                    search_idxs.append(n)
                    seen.add(n)
            yield c

def reduce_to_ms1(df,FILENAMECOL):
    '''takes a dataframe w/ ms2 data in "tidy dataformat" and
    reduces it down to a ms1 df with a ms2df object stored in MS2Info
    
    Args:
        df (pd.DataFrame): ms2 dataframe in tidy (rectangluar) format
        FILENAMECOL (str): filename column (needed for de-replication)
    
    Returns:
        df: a ms1df which has the ms2 data in MS2Info column
    '''

    gb = df.groupby(MS1COLS+[FILENAMECOL])
    ms1_data = []
    for gbi,ms2df in gb:
        fname = gbi[-1]
        ms2df = ms2df.copy()
        ms2df[FILENAMECOL] = [fname]*len(ms2df)
        # ms1_data.append(list(gbi)+[ms2df.to_json(orient='split',index=False)])
        # ms1_data.append(list(gbi)+[ms2df.to_json()])
        ms1_data.append(list(gbi)+[ms2df])
    cols = MS1COLS+[FILENAMECOL,'MS2Info']
    ms1df = pd.DataFrame(ms1_data,columns=cols)

    return ms1df
    
def _average_data_rows(cc_df,datacols,calc_basket_info=False):
    '''average (mean) the datacols values. optionally calculates the bounds
    of each resulting basket. basket_info will be serialized as json.
    
    Args:
        cc_df (pd.DataFrame): connected component dataframe
        datacols (iterable): the columns whose data should be averaged
        calc_basket_info (bool, optional): Defaults to False. flag to compute bounds of basket parameters
    
    Returns:
        list: row of averaged values (+ optional basket_info json)
    '''

    avgd = list(cc_df[list(datacols)].mean())
    if calc_basket_info:
        basket_info = {cn:[float(cc_df[cn].min()),float(cc_df[cn].max())] for cn in datacols}
        basket_info['n'] = len(cc_df)
        avgd.append(json.dumps(basket_info))
    return avgd

def combine_ms2(cc_df,min_reps=2):
    '''combine the ms2 data for a given connected component graph of ms1 ions.
    this is done the same way as the ms1 matching (rtree index) but uses the columns
    defined in global MS2COLSTOMATCH
    
    Args:
        cc_df (pd.DataFrame): conncected component dataframe 
        min_reps (int, optional): Defaults to 2. minimum number of replicates required for ms2 ion to be included   
    
    Returns:
        pd.DataFrame: combined ms2 dataframe
    '''

    # ms2dfs = [pd.read_json(ms2df,orient='split') for ms2df in cc_df['MS2Info']]
    # ms2dfs = [pd.read_json(ms2df) for ms2df in cc_df['MS2Info']]
    ms2dfs = cc_df['MS2Info'].values.tolist()
    ms2df = pd.concat(ms2dfs,sort=True)
    # print(ms2df.columns)
    if ms2df.shape[0] > 1:
        gen_error_cols(ms2df,MS2COLSTOMATCH)
        rects = get_rects(ms2df,MS2ERRORCOLS)
        rtree = build_rtree(ms2df,MS2ERRORCOLS)
        ccs = gen_con_comps(rtree,rects)
        data = []
        file_col = []
        for cc in ccs:
            if len(cc) > 1:
                cc_df = ms2df.iloc[list(cc)]
                uni_files = set(cc_df[FILENAMECOL].values)
                if len(uni_files) >= min_reps:
                    data.append(_average_data_rows(cc_df,MS2COLS))
                    file_col.append('|'.join(uni_files))
            #     else:
            #         data.append([None]*len(MS2COLS))
            # else:
            #     data.append([None]*len(MS2COLS))
                    
        avg_ms2 = pd.DataFrame(data,columns=MS2COLS)
        avg_ms2[FILENAMECOL] = file_col
    else:
        avg_ms2 = ms2df
    # return avg_ms2.to_json(orient='split',index=False) #note that to read back to df orient='split' must be set in pd.read_json()
    return avg_ms2

def _combine_rows(cc_df,uni_files,min_reps,ms2,calc_basket_info=False):
    '''combine ms1 rows and optionally ms2 information in conncected component dataframe
    
    Args:
        cc_df (pd.DataFrame): conncected component dataframe
        uni_files (set): uniqe filenames from cc_df
        min_reps (int): minumum number of replicates (number of uniqe files) that must be in connected component
        ms2 (bool): whether or not to combine ms2 data
        calc_basket_info (bool, optional): Defaults to False. whether or not to calculate basket info (spans)
    
    Returns:
        list: averaged values from rows of cc_df and optionally ms2 info and/or basket info
    '''

    ms1vals = _average_data_rows(cc_df,MS1COLS,calc_basket_info=calc_basket_info)
    if ms2:
        ms2vals = combine_ms2(cc_df,min_reps)
        return ms1vals + [ms2vals]
    else:
        return ms1vals

def proc_con_comps(ccs,df,FILENAMECOL,datacols,min_reps=2,calc_basket_info=False,ms2=True):
    """
    Takes the connected components from the overlapping hyperrectangles and averages (mean)
    the data values from which the error was generated. Unique filenames are concatenated with a 
    '|' delimiter. Only subgraphs with multiple nodes are used and further filtered for only those 
    which come from at least `min_reps` unique files.
    
    
    Args:
        ccs (set): connected component subgraph dataframe indices
        df (pd.DataFrame): the dataframe which the connected component subgraphs were calculated from
        FILENAMECOL (str): filename column to be used for min_reps
        datacols (iterable): column names to be averaged in connected components
        min_reps (int, optional): Defaults to 2. Minimum number of files needed in subgraph to be used
        calc_basket_info(bool, optional): Defaults to False. Whether or not to include json basket info. 
        ms2(bool, optional): Defaults to True. Wether or not to average MS2 data 
    
    Returns:
        pd.DataFrame: newdata frame with data cols and file name col.
    """

    data = []
    file_col = []
    for cc in ccs:
        # if len(cc) > 1:
        cc_df = df.iloc[list(cc)]
        uni_files = set(cc_df[FILENAMECOL].values)
        if len(uni_files) >= min_reps:
            file_col.append('|'.join(uni_files))
            avgd = _combine_rows(cc_df,uni_files,calc_basket_info=calc_basket_info,min_reps=min_reps,ms2=ms2)
            data.append(avgd)
        else:
            continue
    cols = datacols[:]
    if calc_basket_info:
        cols += ['BasketInfo']
    if ms2:
        cols += ['MS2Info']
    
    ndf = pd.DataFrame(data,columns=cols)
    ndf[FILENAMECOL] = file_col

    return ndf

def _proc_one(sample,df_paths,FILENAMECOL,datadir,calc_basket_info=False):
    """
    Process one replica sample. The replicated file is saved as ./Replicated/<sample>_Replicated.csv
    
    Args:
        sample (str): sample name
        df_paths (list): list of paths to replica files to be loaded
    
    Returns:
        str: "DONE" when completed
    """


    dfs = [reduce_to_ms1(pd.read_csv(p),FILENAMECOL) for p in df_paths]
    df = pd.concat(dfs,sort=True)
    df.reset_index(inplace=True)
    gen_error_cols(df,MS1COLSTOMATCH)
    rtree = build_rtree(df,MS1ERRORCOLS)
    con_comps = gen_con_comps(rtree,get_rects(df,MS1ERRORCOLS))
    ndf = proc_con_comps(con_comps,df,FILENAMECOL,MS1COLS,calc_basket_info=calc_basket_info)
    ndf['MS2Info'] = [ms2df.to_json() for ms2df in ndf['MS2Info']]

    
    ndf.to_csv(datadir.joinpath("Replicated").joinpath(f"{sample}_Replicated.csv"))
    gc.collect() # attempt to fix rtree index memory leak...
    return "DONE"

def proc_folder(datadir,FILENAMECOL,calc_basket_info,max_workers):
    """process a folder of sample data replicates. output files will be saved in ./Replacted
    
    Args:
        datadir (str): data directory of sample replicates
    """

    try:
        os.mkdir(datadir.joinpath('Replicated'))
    except OSError:
        pass
    paths = list(gen_rep_df_paths(datadir))
    for sample,df in tqdm(paths,desc='proc_folder'):
        _proc_one(sample,df,FILENAMECOL,datadir,calc_basket_info)

def _update(pbar,future):
    '''callback func for future object to update progress bar'''
    pbar.update()
 
def mp_proc_folder(datadir,FILENAMECOL,calc_basket_info=False,max_workers=0):
    """
    multi proccesor version of proc_folder. by default will use cpu_count - 2 workers.  
    
    process a folder of sample data replicates. output files will be saved in ./Replacted
    
    Args:
        datadir (str): data direcory of sample replicates
        calc_basket_info (bool,optional): Defaults to False. Bool on whether or not to save bin info as json strings. 
        max_workers (int, optional): Defaults to None. If provided will use that many workers for processing. If there is limited system memory this might be good to set low. 
    """


    try:
        os.mkdir(datadir.joinpath('Replicated'))
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
                # fut = ex.submit(_proc_one,sample,paths,FILENAMECOL,datadir,calc_basket_info)
                fut = ex.submit(_proc_one,sample,paths,FILENAMECOL,datadir,calc_basket_info)
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
    return pd.concat(dfs,sort=False)

def _read_json(ms2json,i):
    '''helper func that can be serialized for multiproc json de-serialization'''
    return i,pd.read_json(ms2json)

def basket(datadir,FILENAMECOL,ms2=False,calc_basket_info=False):
    """
    Basket all the replicates in a directory in to a single file called Baskted.csv in datadir
    Unique file names are kept and deliminated with a '|'   
    
    Args:
        datadir (str): the directory of replicated files. 
    """

    print('Loading Rep Files')
    df = make_repdf(datadir)
    orig_len = df.shape[0]
    if ms2: #de-serialize the json df's w/ multiproc 
        with ProcessPoolExecutor() as ex:
            futs = [ex.submit(_read_json,ms2json,i)for i,ms2json in enumerate(df['MS2Info'])]
        ms2dfs = []
        for f in tqdm(as_completed(futs),total=orig_len):
            ms2dfs.append(f.result())
        ms2dfs.sort(key = lambda x:x[0])
        df['MS2Info'] = [x[1] for x in ms2dfs]

    # need to handle multiple file name cols from legacy/mixed input files
    df[FILENAMECOL] = np.where(df[FILENAMECOL].isnull(), df["Sample"], df[FILENAMECOL])
    df.dropna(subset=[FILENAMECOL],inplace=True)
    print(f"Dropped {orig_len-df.shape[0]} rows missing values in {FILENAMECOL}")
    gen_error_cols(df)
    print("Making Rtree Index")
    rtree = build_rtree(df)
    print('Generating Baskets')
    con_comps = gen_con_comps(rtree,get_rects(df),pbar=True)
    ndf = proc_con_comps(con_comps,df,FILENAMECOL,MS1COLS,min_reps=1,calc_basket_info=calc_basket_info,ms2=ms2)
    # ndf['MS2Info'] = [ms2df.to_json(orient='split',index=False) for ms2df in ndf['MS2Info']]
    ndf['freq'] = [len(s.split('|')) for s in ndf[FILENAMECOL]]
    ndf.to_csv(os.path.join(datadir,"Basketed.csv"))
 
def filename2sample(filename, fn_delim='_', sampleidx=1):
    sample = filename.split(fn_delim)[sampleidx]
    return sample

def filenames2samples(filenames, delim='|', fn_delim='_', sampleidx=1):
    samples = {filename.split(fn_delim)[sampleidx] for filename in filenames.split(delim)}
    return samples

def synth_fp(actd,samples):
    to_cat = get_fps(actd,samples)
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
    return np.asarray(to_cat)


def cluster_score(fpd,samples,metric='euclidean'):
    fps = get_fps(fpd,samples)
    if fps.shape[0] == 1:
        return 0
    cubed = pdist(np.vstack(fps),metric=metric)**3
    score = cubed.mean()
    # if np.isnan(cubed):
    #     print(fps)
    #     raise 
    return score
         
def load_basket_data(bpath:str, samplecol="Sample",mzcol="PrecMz",rtcol="RetTime", ccscol="CCS", intcol="PrecIntensity") -> list:
    df = pd.read_csv(bpath)
    ms1df = pd.DataFrame(list(set(df[[rtcol,mzcol,ccscol,intcol,samplecol]].itertuples(index=False))))
    baskets = []
    for rowt in ms1df.itertuples(index=False):
        rt,mz,ccs,inti,samplestr = rowt
        samples = filenames2samples(samplestr)
        bd = {'mz':mz,'rt':rt,'ccs':ccs,'inti':inti,'samples':samples}
        baskets.append(bd)
    return baskets
    

def load_activity_data(apath: str) -> dict:
    p = Path(apath)
    if p.is_dir():
        filenames = [sd.name for sd in os.scandir(p) if sd.name.lower().endswith('csv')]
        dfs = {}
        for fname in filenames:
            df = pd.read_csv(p.joinpath(fname),index_col=0)

            df.fillna(value=0, inplace=True)
            name = os.path.splitext(fname)[0]
            dfs[name]=df


    if p.is_file():
        name = os.path.splitext(p.name)[0]
        dfs[name] = pd.read_csv(apath)
    # df[name_col] = df[name_col].apply(lambda x:x.split('-')[0])
    actd = defaultdict(dict)
    for name,df in dfs.items():
        for row in df.itertuples():
            actd[name][row[0]] = np.asarray(row[1:])
    actd = dict(actd) #convert to normal dict
    return actd



def get_config(cpath:str):
    config = configparser.ConfigParser()
    with open(cpath) as fin:
        config.read_file(fin)
    return config
    
Scoret = namedtuple('Scoret', 'activity cluster')
def score_baskets(baskets,actd):
    scores = defaultdict(dict)
    for i,basket in tqdm(enumerate(baskets)):
        samples = basket['samples']
        for actname,fpd in actd.items():
            try:
                sfp = synth_fp(fpd,samples)
                act_score = np.sum(sfp**2)
                scores[actname][i] = Scoret(act_score,cluster_score(fpd,samples))
            except KeyError:
                pass
                # embed()
                # raise
    scores = dict(scores)
    return scores


def load_default_basket_and_activity():
    config = get_config('default.cfg')
    baskets = load_basket_data(config['BasketInfo']['path'],samplecol=config['BasketInfo']['samplecol'])
    actd = load_activity_data(config['ActivityFileInfo']['path'])
    return baskets,actd

def load_default_basket_and_activity(basket_path,act_path):
    baskets = load_basket_data(basket_path)
    actd = load_activity_data(act_path)
    return baskets,actd

def make_bokeh_input(baskets,actd):
    # baskets,actd = load_default_basket_and_activity()
    scores = score_baskets(baskets,actd)
    data = []
    for i,basket in enumerate(baskets):
        bid = f"Basket_{i}"
        freq = len(basket['samples'])
        samplelist = json.dumps(list(basket['samples']))
        try:
            cpact = scores['CPActivity'][i].activity
            cpclust = scores['CPActivity'][i].cluster
        except KeyError:
            cpact,cpclust = None,None
        try:
            fusact = scores['FusionActivity'][i].activity
            fusclust = scores['FusionActivity'][i].cluster
        except KeyError:
            fusact,fusclust = None,None
        
        row = (bid,freq,basket['mz'],basket['rt'],basket['ccs'],samplelist,cpact,cpclust,fusact,fusclust,cpclust)
        data.append(row)
    columns = ('BasketID','Frequency','PrecMz','RetTime','CCS','SampleList','CP_ACTIVITY','CP_CLUSTER_SCORE','FUSION_ACTIVITY','FUSION_CLUSTER_SCORE','SNF_CLUSTER_SCORE')
    df = pd.DataFrame(data,columns=columns)
    df.to_excel("BokehInput.xlsx")

def make_cytoscape_input(baskets,actd,act_thresh=5,clust_thresh=10):
    # baskets,actd = load_default_basket_and_activity()
    scores = score_baskets(baskets,actd)
    edges = []
    basket_info = []
    _basket_keys= ['mz','rt','ccs','inti']
    samples = set()
    for i,basket in enumerate(baskets):
        bid = f"Basket_{i}"
        if i in scores['CPActivity']:
            score = scores['CPActivity'][i]
            if score.activity > act_thresh and score.cluster < clust_thresh:
                samples.update(basket['samples'])
                for samp in basket['samples']:
                    edges.append((bid,samp))
        basket_info.append(
            [bid]+ 
            [basket[k] for k in _basket_keys]+ 
            [len(basket['samples'])]+ 
            [f"{basket['mz']:.4f}"])

    with open('EdgeList.txt','w') as fout:
        print('Source\tTarget\tScore',file=fout)
        for e in edges:
            print(f'{e[0]}\t{e[1]}\t1',file=fout)
    with open('Atributes.csv','w') as fout:
        print('bid,mz,rt,ccs,inti,freq,combo_name',file=fout)
        for b in basket_info:
            print(','.join(map(str,b)),file=fout)
        for samp in samples:
            print(f'{samp},,,,,,{samp}',file=fout)

def load_and_generate_act_outputs(basket_path,act_path):
    baskets = load_basket_data(basket_path)
    actd = load_activity_data(act_path)
    make_bokeh_input(baskets,actd)
    make_cytoscape_input(baskets,actd)