"""mpi wrapper imported from pixell to reduce dependency on pixell for
simple things. Notes from original script: Utilities for making mpi
use safer and easier.
"""


from __future__ import print_function
import sys, os, traceback

class FakeCommunicator:
    def __init__(self):
        self.size = 1
        self.rank = 0

FAKE_WORLD = FakeCommunicator()
COMM_WORLD = FAKE_WORLD
COMM_SELF  = FAKE_WORLD
disabled   = True

try:
    if not("DISABLE_MPI" in os.environ and os.environ["DISABLE_MPI"].lower() in ["true","1"]):
        from mpi4py.MPI import *
        disabled = False
except:
    pass
