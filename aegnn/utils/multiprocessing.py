# from https://stackoverflow.com/questions/9601802/python-pool-apply-async-and-map-async-do-not-block-on-full-queue?rq=1
import logging
import multiprocessing
import torch
import torch.multiprocessing
import threading
import tqdm

from typing import Callable


class TaskManager(object):
    """Task Manager for multi-thread processing on cuda.

    Args:
        num_workers: number of parallel processes.
        queue_size: length of waiting queue.
        callback: function called after process has been finished, following the same
                  input structure as the actual processed function (default = None).
        total: total number of elements to process (for progress visualization only).
    """

    def __init__(self, num_workers: int, queue_size: int, callback: Callable = None, total: int = None):
        self.__progress_bar = tqdm.tqdm(total=total)

        if torch.cuda.is_initialized():
            self.__pool = torch.multiprocessing.Pool(processes=num_workers)
        else:
            self.__pool = multiprocessing.Pool(processes=num_workers)

        self.__workers = threading.Semaphore(num_workers + queue_size)
        self.__callback = callback
        self.outputs = []
        self.index = 0

    def __enter__(self):
        return self

    def join(self):
        self.__pool.close()
        self.__pool.join()

    def __exit__(self, error_type, value, traceback):
        self.join()
        self.outputs = [(i, r.get()) for i, r in self.outputs]

    def queue(self, function, *args, **kwargs):
        """Start a new task, blocks if queue is full."""
        self.__workers.acquire()
        res = self.__pool.apply_async(function, args, kwargs, callback=self.__task_done,
                                      error_callback=self.__release_and_log)
        self.outputs += [(self.index, res)]
        self.index += 1

    def __task_done(self, *args, **kwargs):
        """Called once task is done, releases the queue is blocked."""
        self.__workers.release()
        if self.__callback is not None:
            self.__callback(*args, **kwargs)
        self.__progress_bar.update(1)

    def __release_and_log(self, e):
        self.__workers.release()
        logging.error(e)
