import matplotlib.pyplot as plt

import numpy as np

from rtree import index
from tqdm import tqdm


def plot_ranges(points,e=0.1):
    a = np.asarray(points)
    xs,ys = a.T
    plt.scatter(xs,ys)
    for x,y in a:
        plt.plot((x-e,x+e),(y,y),c='r')
        plt.plot((x,x),(y-e,y+e),c='r')


def make_rects(a,e=0.1):
    a = np.asarray(a)
    xlow = a[:,0] - e
    ylow = a[:,1] - e
    xhigh = a[:,0] + e
    yhigh = a[:,1] + e
    return [tup for tup in zip(xlow,ylow,xhigh,yhigh)]

def make_randi(points,e=0.1):
    rects = make_rects(points,e)
    idx = index.Index(((i,r,None) for i,r in enumerate(rects)))
    return rects,idx


from numba import jit, jitclass,float64,int64

spec = [
    ('idxs',int64[:]),
    ('vals',float64[:]),
    ('lows',float64[:]),
    ('highs',float64[:]),
]


@jit
def meanit(a1,a2):
    if a1.shape != a2.shape:
        raise ValueError("arrays must be same shape! (1d)")
    else:
        res = np.zeros(a1.shape[0])
        for i,tup in enumerate(zip(a1,a2)):
            res[i] = (tup[0] + tup[1]) / 2
        return res
            

@jitclass(spec)
class Rect(object):
    def __init__(self,idxs,vals,lows,highs):
        self.idxs = idxs
        self.vals = vals
        self.lows = lows
        self.highs = highs
    def eq(self,other):
        if np.all((self.lows <= other.highs) &  (self.highs<= other.highs)):
            return True
        return False
    def add(self,other):
        vals = meanit(self.vals,other.vals)
        lows = meanit(self.lows,other.lows)
        highs = meanit(self.highs,other.highs)
        idxs = np.concatenate((self.idxs,other.idxs))
        return Rect(idxs,vals,lows,highs)


@jit(nopython=True)
def gb_sum(d,grps):
    uniks = np.unique(grps)
    result = np.zeros_like(uniks)
    for i,k in enumerate(uniks):
        result[i] = d[grps == k].sum()
    return result



@jit(nopython=True)
def get_other_bins(rect):
    spatkey = np.floor(rect.vals)
    new_bins = np.zeros((spatkey.shape[0]*2,spatkey.shape[0]))
    lowes = rect.lows < spatkey
    highes = rect.highs > spatkey
    nb_idx = 0
    for i,b in enumerate(lowes):
        nb_idx +=1
        if b:
            newk = spatkey.copy()
            newk[i] = spatkey[i] - 1
            new_bins[i] = newk
    for i,b in enumerate(highes):
        nb_idx +=1
        if b:
            newk = spatkey.copy()
            newk[i] = spatkey[i] + 1
            new_bins[i] = newk
    return np.unique(new_bins)
        

def make_hashes():
    pass


# @jit(nopython=True)
def make_numba_rects(points):
    lows = points - 0.1
    highs = points + 0.1
    # rects = np.zeros(points.shape[0],dtype='object')
    rects = []
    for i,tup in enumerate(zip(points,lows,highs)):
        v,h,l = tup
        rects.append( Rect(np.array([i]).astype('int64'),v,l,h))
    return rects
        
idxs = np.array([1],dtype='int64')
v = np.array((1,1),dtype='float64')
l = np.array((0.9,0.9),dtype='float64')
h = np.array((1.1,1.1),dtype='float64')

r = Rect(idxs,v,l,h)


# import math
# from numba import vectorize, cuda
# import numpy as np

# @vectorize(['float32(float32, float32, float32)',
#             'float64(float64, float64, float64)'],
#            target='cuda')
# def cu_discriminant(a, b, c):
#     return math.sqrt(b ** 2 - 4 * a * c)

# N = int(1e+4)
# dtype = np.float32

# # prepare the input
# A = np.array(np.random.sample(N), dtype=dtype)
# B = np.array(np.random.sample(N) + 10, dtype=dtype)
# C = np.array(np.random.sample(N), dtype=dtype)

# D = cu_discriminant(A, B, C)



## Pure Python(ish) section ##


CONFIGD = {
    'filename':'MgfFileName',
    'mz':'PrecMz',
    'rt':'RetTime',
    'ccs':'CCS',
    'mz2':'ProdMz',
    'z': 'PrecZ'}

VALKEYS = ['z','mz','rt','ccs','mz2']

class MzFeature(object):
    __slots__ = ('values','lows','highs','infod','spatkey','fnames')
    def __init__(self,values,lows,highs,infod,fnames):
        self.values = values
        self.lows = lows
        self.highs = highs
        self.infod = infod
        self.spatkey = self._gen_spatkey()
        if isinstance(fnames,set):
            self.fnames == fnames
        elif isinstance(fnames,str):
            self.fnames = set([infod[CONFIGD['filename']]])

    def _combine_items(self,d1,d2):
        retd = {}
        for key,val in d1.items():
            if isinstance(val, (int,float)):
                retd[key] = (val+d2[key]) / 2
            elif isinstance(val,list):
                if isinstance(d2[key],list):
                    retd[key] = d2[key] + v
                else:
                    retd[key] = [d2[key]] + v
            else:
                retd[key] = [d2[key]] + [v]
        return retd

    def __str__(self):
        return f"<MzFeature z:{self.values[0]} mz:{self.values[1]:.4f} rt:{self.values[2]:.2f} ccs:{self.values[3]:.1f} mz2:{self.values[4]:.4f}>"

    def combine(self,other):
        nvals = np.mean((self.values,other.values),axis=0)
        nlows = np.mean((self.lows, other.lows),axis=0)
        nhighs = np.mean((self.highs,other.highs),axis=0)
        ninfod = self._combine_items(self.infod,other.infod)
        nfnames = self.fnames | other.fnames
        return MzFeature(nvals,nhighs,nlows,ninfod)

    def eq(self,other):
        if np.all((self.lows <= other.highs) &  (self.highs<= other.highs)):
            return True
        return False     

    def _gen_spatkey(self,returntype='str'):
        temp = self.values[:]
        temp[0] = temp[0] * 10
        temp[4] = temp[4] * 10
        floored = np.floor(temp)
        if returntype == 'str':
            return "_".join((str(v) for v in floored))
        elif returntype == 'array':
            return floored
        else:
            raise ValueError(f"Unkown returntype option {returntype}, allowable options 'str','array'")
    
from collections import defaultdict
from itertools import chain

class GroupedFeatures(object):
    def __init__(self,feats):
        self._dict = defaultdict(list)
        for feat in feats:
            self._dict[feat.spatkey].append(feat)
        self._dict = dict(self._dict)

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def values(self):
        return self._dict.values()
    
    def __getitem__(self,key):
        return self._dict[key]
    def __len__(self):
        return len(self._dict)


def load_file(f):
    df = pd.read_csv(f)
    vals = df[[CONFIGD[k] for k in VALKEYS]].values
    lows = vals - 0.1
    highs = vals + 0.1
    for i,t in enumerate(zip(vals,lows,highs)):
        infod = df.loc[i].to_dict()
        v,l,h = t
        yield MzFeature(v,l,h,infod)

def de_rep(mzfs):
    groupd = GroupedFeatures(mzfs)
    for key,group in tqdm(groupd.items()):
        yield (reduce_group(group))


def reduce_group(group):
    group = group[:]
    reduced = []
    while group:
        current = group.pop()
        to_del = []
        for i,other in enumerate(group):
            if other.eq(current):
                current = current.combine(other)
                to_del.append(i)
        reduced.append(current)
        for idx in to_del[::-1]:
            del group[idx]
    return reduced



        
        
        

        


