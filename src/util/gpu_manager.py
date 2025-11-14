import gc
from contextlib import contextmanager
import torch

class GPUManager:
    @classmethod
    def device(cls):
        if torch.cuda.is_available():
            return 'cuda:0'
        else:
            return 'cpu'

    @classmethod
    @contextmanager
    def gpu_routine(cls, enter_gpu=None, exit_gpu=None):
        if torch.cuda.is_available():
            enter_gpu and enter_gpu()
            with torch.no_grad():
                yield
            exit_gpu and exit_gpu()
            gc.collect()
            torch.cuda.empty_cache()
        else:
            yield
