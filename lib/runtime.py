import timeit
from tracemalloc import stop

class Runtime:
    def start(self):
        start = timeit.default_timer()
        return start 
    def stop(self,start):
        stop = timeit.default_timer()
        print(f"Runtime: {stop-start}")
        return stop