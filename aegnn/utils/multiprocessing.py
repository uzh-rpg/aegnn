# from https://stackoverflow.com/questions/9601802/python-pool-apply-async-and-map-async-do-not-block-on-full-queue?rq=1
import logging
import multiprocessing
import torch
import torch.multiprocessing
import threading

from typing import Callable, Optional


class TaskManager:
    """Task Manager for multi-thread processing on cuda.

    Args:
        num_workers: number of parallel processes.
        queue_size: length of waiting queue.
        callback: function called after process has been finished, following the same
                  input structure as the actual processed function (default = None).
    """

    def __init__(self, num_workers: int, queue_size: int, callback: Optional[Callable] = None):
        self.callback = callback
        self.multiprocessing = num_workers > 1

        if self.multiprocessing:
            if torch.cuda.is_initialized():
                self._pool = torch.multiprocessing.Pool(processes=num_workers)
            else:
                self._pool = multiprocessing.Pool(processes=num_workers)
            self._workers = threading.Semaphore(num_workers + queue_size)
            self._outputs = []
            self._index = 0

    def queue(self, function, *args, **kwargs):
        # Multi-processing for performing task in several threads.
        if self.multiprocessing:
            def release_and_log(e):
                self._workers.release()
                logging.error(e)

            self._workers.acquire()
            res = self._pool.apply_async(function, args, kwargs, callback=self._done, error_callback=release_and_log)
            self._outputs += [(self._index, res)]
            self._index += 1

        # Single process processing, completely without the multiprocessing stuff.
        else:
            function(*args, **kwargs)
            self._done()

    def _done(self, *args, **kwargs):
        """Called once task is done, releases the queue is blocked."""
        if self.multiprocessing:
            self._workers.release()
        if self.callback:
            self.callback(*args, **kwargs)

    def __enter__(self):
        return self

    def join(self):
        if self.multiprocessing:
            self._pool.close()
            self._pool.join()

    def __exit__(self, error_type, value, traceback):
        if self.multiprocessing:
            self.join()
            self._outputs = [(i, r.get()) for i, r in self._outputs]
