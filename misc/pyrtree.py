import numpy as np
import math
class Rectangle(object):
    def __init__(self,mins,maxes):
        self.maxes = maxes
        self.mins = mins
        if len(self.mins) != len(self.maxes):
            raise ValueError("Mins and Maxes must be equal length!")
        if not np.all(self.mins <= self.maxes):
            raise ValueError("All mins must be <= maxes!")
        self.k = len(self.mins)
        self.dims = self.k

    @property
    def center(self):
        return np.mean((self.mins,self.maxes),axis=0)
    def dimbounds(self,m):
        return (self.mins[m],self.maxes[m])

    def intersects(self,other):
        if np.all((self.mins <= other.maxes) & (other.mins <= self.maxes)):
            return True
        else:
            return False
    def contains(self,other):
        if np.all((self.mins <= other.mins) & (self.maxes >= other.maxes)):
            return True
        else:
            return False

class RectNode(Rectangle):
    def __init__(self,mins,maxes,children,dim):
        super().__init__(mins,maxes )
        self.children = children
        self.is_leaf = False if dim > 0 else True
    
    @classmethod
    def from_rect(cls,rect,dim):
        return cls(rect.mins,rect.maxes,children=[],dim=dim)
    @classmethod
    def from_children(cls,children,dim):
        child_mins = np.min([c.mins for c in children],axis=0)
        child_maxes = np.min([c.maxes for c in children],axis=0)
        return cls(child_mins,child_maxes,children=children,dim=dim)
    @property
    def MBR(self):
        child_mins = np.min([c.mins for c in self.children],axis=0)
        child_maxes = np.min([c.maxes for c in self.children],axis=0)
        return Rectangle(mins=child_maxes,maxes=child_maxes)
    
    # def split_dim(self,dim):
    #     '''split yourself in half, returning two nodes with the children in each'''
    #     x = x[x[:,dim].argsort()]

class RTree(object):
    def __init__(self,rects,M=100):
        self.rects = rects
        self.centers = np.asarray([r.center for r in self.rects])
        self.mins = np.asarray([r.mins for r in self.rects])
        self.maxes = np.asarray([r.maxes for r in self.rects])
        self.K = self.rects[0].dims-1
        self.N = len(self.mins)
        self.M = M
        self.min_bounds = self.mins.min(axis=0)
        self.max_bounds = self.mins.max(axis=0)
        self.idx = self._construct()
    
    
    def _construct(self):
        pass

    def _OTM(self):
        h = math.ceil(math.log(self.N,self.M))
        Nsub = int(self.M ** (h-1))
        S = math.floor(math.sqrt(math.ceil(self.N/Nsub)))
        pass
    
    def _STR(self):
        def _gen_slabs(x,slabn,k):
            x = x[np.argsort([y.center[k] for y in x])]
            j = 0
            for i in range(0,x.size,slabn):
                if j+i < x.size:
                    yield k,x[j:j+i]
                    j +=i
                else:
                    yield k,x[j:]
        k = self.K
        P = math.ceil(self.N/self.M)
        slabn = self.M * math.ceil(P**((k-1)/k))
        stack = []
        top_slabs = _gen_slabs(self.rects,slabn,k)
        stack = list(top_slabs)
        leaves = []
        nodes = []
        while stack:
            old_k,slab =  stack.pop()
            if k >1:
                k = old_k-1
                s = math.ceil(P**(1/k))
                slabn = self.M * math.ceil(P**((k-1)/k))
                slabs = list(_gen_slabs(slab,slabn,k))
                stack.extend(slabs)
                nodes 
            else:
                j = 0
                for i in range(0,len(slab),self.M):
                    try:
                        children = slab[j:j+i]
                        j+=i
                    except IndexError:
                        children = slab[j:]
                    if children.size > 0:
                        leaf = RectNode.from_children(children,dim=0)
                        leaves.append(leaf)
        self.leaves = leaves


def gen_synth_rects(n=1000,dim=4):
    data = np.random.rand(n,dim)
    dmins = data -0.1
    dmaxes = data + 0.1
    for mins,maxes in zip(dmins,dmaxes):
        yield Rectangle(mins,maxes)

def test_build():
    rects = np.asarray(list(gen_synth_rects()))
    rtree = RTree(rects)
    rtree._STR()
    return rtree