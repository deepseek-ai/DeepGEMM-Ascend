from contextlib import contextmanager
from typing import Callable
import os
import torch

@contextmanager
def dry_run():
    from deep_gemm._C import set_dry_run
    set_dry_run(True)
    try:
        yield
    finally:
        set_dry_run(False)


def par_compile(callables: list[Callable[[], None]], *, max_workers: int = None, progress=True):
    device = torch.npu.current_device()

    def run(fn):
        with torch.npu.device(device):
            torch.npu.current_stream()
            return fn()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    torch.npu.current_stream()
    if progress:
        from tqdm.auto import tqdm
        pbar = tqdm(total=len(callables), desc='Compiling')
    if max_workers is None:
        max_workers = int(os.environ.get('DG_TEST_MAX_WORKERS', os.cpu_count() or 1))
    with dry_run():
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='deepgemm-compile') as pool:
            futures = [pool.submit(run, fn) for fn in callables]
            for fut in as_completed(futures):
                fut.result()
                if progress:
                    pbar.update(1)
    if progress:
        pbar.close()
