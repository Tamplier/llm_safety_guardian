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
    def device_setup(cls, enter_gpu):
        torch.cuda.is_available() and enter_gpu()

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

            try:
                import cupy as cp
                mempool = cp.get_default_memory_pool()
                pinned_mempool = cp.get_default_pinned_memory_pool()
                mempool.free_all_blocks()
                pinned_mempool.free_all_blocks()
            except (ImportError, AttributeError):
                pass
        else:
            yield
