import timeit

class Runtime:
    def start():
        start = timeit.default_timer()
        return start 
    def stop(start):
        stop = timeit.default_timer()
        print(f"Runtime: {stop-start}")
        return stop