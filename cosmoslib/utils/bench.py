"""Some benchmarking tools"""
import time
from contextlib import contextmanager


# Taken from astropaint
@contextmanager
def timeit(process_name="Process"):
    """Time the code in mins"""
    time_stamp = time.strftime("%H:%M:%S %p")
    print("{:=>50}\n{} started at {}\n".format("", process_name, time_stamp))
    t_i = time.time()
    yield
    t_f = time.time()
    t = t_f-t_i
    print("{} was done in {:.1f} min.\n{:=>50}\n".format(process_name, t/60,""))
