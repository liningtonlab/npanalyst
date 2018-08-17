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
    def __init__(self,mins,maxes,children,is_leaf=False):
        super().__init__(self,mins,maxes)
        self.children = children
    @classmethod
    def from_rect(cls,rect,is_leaf=False):
        return cls(rect.mins,rect.maxes,children=[],is_leaf=is_leaf)
    def split(self):
        '''split yourself in half, returning two nodes with the children in each'''
        pass

class RTree(object):
    def __init__(self,rects,M=100):
        self.rects = rects
        self.centers = np.asarray([r.center for r in self.rects])
        self.mins = np.asarray([r.mins for r in self.rects])
        self.maxes = np.asarray([r.maxes for r in self.rects])
        self.K = self.rects[0].dims
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
        k = self.K
        P = math.ceil(self.N/self.M)
        slabn = self.M * math.ceil(P**((k-1)/k))
        stack = []
        def _gen_slabs(x,slabn,k):
            x = x[x[:,k].argsort()]
            j = 0
            for i in range(x.size,slabn):
                if j+i < x.size:
                    yield x[j:j+i]
                    j +=i
                else:
                    yield x[j:]
        top_slabs = _gen_slabs(self.rects,slabn,k)
        satack = [RectNode]
        while k !=0 :
            s = math.ceil(P**(1/k))
            slabn = self.M * math.ceil(P**((k-1)/k))
            
            

            while stack:
                node = stack.pop()





