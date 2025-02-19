import logging
import os
import time
import torch
from abc import abstractmethod

import torch.multiprocessing as mp

from ...core import default

logger = logging.getLogger(__name__)

# used as a global variable when processes are initially started.
process_initial_data = None


def _initializer(*args):
    """
    Process initializer function that is called when mp.Pool is started.
    :param args:    arguments that are to be copied to the target process. This can be a tuple for convenience.
    """
    global process_initial_data
    process_id, process_initial_data = args

    assert 'OMP_NUM_THREADS' in os.environ
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))

    # manually set process name
    with process_id.get_lock():
        mp.current_process().name = 'PoolWorker-' + str(process_id.value)
        logger.info('pid=' + str(os.getpid()) + ' : ' + mp.current_process().name)

        process_id.value += 1

class AbstractStatisticalModel:
    """
    AbstractStatisticalModel object class.
    A statistical model is a generative function, which tries to explain an observed stochastic process.
    """

    ####################################################################################################################
    ### Constructor:
    ####################################################################################################################

    def __init__(self, name='undefined'):
        self.name = name
        self.fixed_effects = {}
        self.priors = {}
        self.individual_random_effects = {}
        self.n_sources = 0

        self.pool = None

    def _cleanup_multiprocess_pool(self):
        if self.pool is not None:
            self.pool.terminate()

    ####################################################################################################################
    ### Common methods, not necessarily useful for every model.
    ####################################################################################################################

    def cleanup(self):
        self._cleanup_multiprocess_pool()

    def clear_memory(self):
        pass

