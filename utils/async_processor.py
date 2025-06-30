import asyncio
import queue
import threading
import time
import concurrent.futures
import logging
from typing import Callable, List, Any

logger = logging.getLogger(__name__)

class AsyncProcessor:
    """Asynchronous task processor with thread pool"""
    
    def __init__(self, max_workers: int = 4, queue_size: int = 100):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = queue.Queue(maxsize=queue_size)
        self._running = True
        self.worker_thread = threading.Thread(target=self._process_tasks, daemon=True)
        self.worker_thread.start()
    
    def _process_tasks(self):
        """Worker thread processing tasks from queue"""
        while self._running:
            try:
                future, fn, args, kwargs = self.task_queue.get(timeout=0.1)
                try:
                    result = fn(*args, **kwargs)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Task processing error: {e}")
    
    def submit(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit a task for asynchronous processing"""
        if not self._running:
            raise RuntimeError("Processor is shutting down")
            
        future = concurrent.futures.Future()
        try:
            self.task_queue.put((future, fn, args, kwargs), timeout=0.1)
        except queue.Full:
            future.set_exception(RuntimeError("Task queue full"))
        return future
    
    async def submit_async(self, fn: Callable, *args, **kwargs) -> Any:
        """Submit and await result asynchronously"""
        loop = asyncio.get_running_loop()
        future = self.submit(fn, *args, **kwargs)
        return await loop.run_in_executor(None, future.result)
    
    def shutdown(self, wait: bool = True):
        """Shutdown the processor"""
        self._running = False
        self.worker_thread.join(timeout=5.0)
        self.executor.shutdown(wait=wait)


class AsyncBatchProcessor:
    """Efficient batch processor with parallel processing and strict queue limits"""

    def __init__(
        self,
        batch_size: int = 8,
        max_workers: int = 4,
        max_queue_size: int = 100,
    ):
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size

        self.batch_queue = queue.Queue()
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        )

        self._lock = threading.Lock()
        self._outstanding = 0  

        self._running = True
        self.worker_thread = threading.Thread(
            target=self._process_batches, daemon=True
        )
        self.worker_thread.start()

    def submit_batch(self, fn: Callable, args_list: List[Any]) -> List[concurrent.futures.Future]:
        """Submit each args in args_list as its own batch‐of‐1 under fn."""
        futures: List[concurrent.futures.Future] = []

        for args in args_list:
            fut = concurrent.futures.Future()
            with self._lock:
                if self._outstanding >= self.max_queue_size:
                    fut.set_exception(RuntimeError("Batch queue full"))
                else:
                    self._outstanding += 1
                    self.batch_queue.put((fut, fn, args))
            futures.append(fut)

        return futures

    def _process_batches(self):
        """Continuously pull up to batch_size items, dispatch them,
           then decrement the outstanding counter."""
        while self._running:
            try:
                batch = []
                while len(batch) < self.batch_size:
                    try:
                        task = self.batch_queue.get(timeout=0.1)
                        batch.append(task)
                    except queue.Empty:
                        break

                if not batch:
                    continue

                proc_futs = []
                for orig_fut, fn, args in batch:
                    proc = self.executor.submit(fn, [args])
                    proc_futs.append((orig_fut, proc))

                for orig_fut, proc in proc_futs:
                    try:
                        res_list = proc.result(timeout=1.0)
                        orig_fut.set_result(res_list[0])
                    except Exception as e:
                        orig_fut.set_exception(e)

                for _ in batch:
                    self.batch_queue.task_done()

                with self._lock:
                    self._outstanding -= len(batch)

            except Exception as e:
                logger.error(f"Batch processing error: {e}")

    def shutdown(self):
        """Shutdown processing thread + executor."""
        self._running = False
        self.worker_thread.join(timeout=1.0)
        self.executor.shutdown(wait=False)