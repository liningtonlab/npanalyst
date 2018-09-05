import multiprocessing as mp

class Worker(mp.Process):
    def __init__(self,queue):
        super().__init__()
        self.queue = queue
    
    def run(self):
        pass

    