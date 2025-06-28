import asyncio
import concurrent.futures
from typing import Callable, Any, Coroutine
import logging
from queue import Queue
from threading import Thread

logger = logging.getLogger(__name__)

class AsyncProcessor:
    """Advanced async task processor with thread pool and queue management"""
    
    def __init__(self, max_workers: int = 4, queue_size: int = 100):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = Queue(maxsize=queue_size)
        self._running = True
        
        # Start worker thread
        self.worker_thread = Thread(target=self._process_tasks, daemon=True)
        self.worker_thread.start()
    
    def _process_tasks(self):
        """Worker thread processing tasks from queue"""
        while self._running or not self.task_queue.empty():
            try:
                future, fn, args, kwargs = self.task_queue.get(timeout=0.1)
                try:
                    result = fn(*args, **kwargs)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                self.task_queue.task_done()
            except:
                continue
    
    def submit(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit a task for asynchronous processing"""
        if not self._running:
            raise RuntimeError("Processor is shutting down")
            
        future = concurrent.futures.Future()
        self.task_queue.put((future, fn, args, kwargs))
        return future
    
    async def submit_async(self, fn: Callable, *args, **kwargs) -> Any:
        """Submit and await result asynchronously"""
        loop = asyncio.get_running_loop()
        future = self.submit(fn, *args, **kwargs)
        return await loop.run_in_executor(None, future.result)
    
    def shutdown(self, wait: bool = True):
        """Shutdown the processor"""
        self._running = False
        self.executor.shutdown(wait=wait)
        self.worker_thread.join(timeout=5.0)


class AsyncBatchProcessor(AsyncProcessor):
    """Batch processing with configurable batching strategies"""
    
    def __init__(self, batch_size: int = 8, timeout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.timeout = timeout
        self.batch_queue = Queue()
        self.batch_worker = Thread(target=self._process_batches, daemon=True)
        self.batch_worker.start()
    
    def _process_batches(self):
        """Process tasks in batches"""
        while self._running or not self.batch_queue.empty():
            try:
                batch = []
                # Wait for batch to fill or timeout
                while len(batch) < self.batch_size:
                    try:
                        task = self.batch_queue.get(timeout=self.timeout)
                        batch.append(task)
                    except:
                        if batch:
                            break
                
                if not batch:
                    continue
                
                # Process batch
                futures, fn, batch_args = zip(*[(t[0], t[1], t[2]) for t in batch])
                try:
                    results = fn(batch_args)
                    for future, result in zip(futures, results):
                        future.set_result(result)
                except Exception as e:
                    for future in futures:
                        future.set_exception(e)
                
                for _ in batch:
                    self.batch_queue.task_done()
                    
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
    
    def submit_batch(self, fn: Callable, args_list: list) -> list:
        """Submit a batch of tasks"""
        futures = []
        for args in args_list:
            future = concurrent.futures.Future()
            self.batch_queue.put((future, fn, args))
            futures.append(future)
        return futures