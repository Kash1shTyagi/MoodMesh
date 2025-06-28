import asyncio
import concurrent.futures
from typing import Callable, Any, Coroutine
import logging
import queue
from queue import Queue
from threading import Thread
import threading

logger = logging.getLogger(__name__)

class AsyncProcessor:
    """Advanced async task processor with thread pool and queue management"""
    
    def __init__(self, max_workers: int = 4, queue_size: int = 100):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = Queue(maxsize=queue_size)
        self._running = True
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



class AsyncBatchProcessor:
    def __init__(self, batch_size: int = 8, timeout: float = 0.1, 
                 max_queue_size: int = 100, **kwargs):
        self.batch_size = batch_size
        self.timeout = timeout
        self.batch_queue = queue.Queue(maxsize=max_queue_size)
        self._running = True
        self.processing_lock = threading.Lock()
        self.processing_lock.acquire()  
        
        self.worker_thread = threading.Thread(target=self._process_batches, daemon=True)
        self.worker_thread.start()
    
    def _process_batches(self):
        """Process tasks in batches"""
        while self._running:
            try:
                self.processing_lock.acquire()
                
                batch = []
                try:
                    task = self.batch_queue.get(timeout=self.timeout)
                    batch.append(task)
                except queue.Empty:
                    continue
                
                future, fn, arg = batch[0]
                
                try:
                    result = fn([arg])[0]  
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                
                self.batch_queue.task_done()
                    
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
    
    def allow_processing(self):
        """Release the processing lock to allow items to be processed"""
        try:
            self.processing_lock.release()
        except RuntimeError:
            pass
    
    def submit_batch(self, fn: Callable, args_list: list) -> list:
        """Submit a batch of tasks"""
        futures = []
        for args in args_list:
            future = concurrent.futures.Future()
            try:
                self.batch_queue.put((future, fn, args), timeout=0.01)
                futures.append(future)
            except queue.Full:
                future.set_exception(RuntimeError("Batch queue full"))
                futures.append(future)
        return futures
    
    def shutdown(self, wait: bool = True):
        """Shutdown the processor"""
        self._running = False
        self.allow_processing()
        self.worker_thread.join(timeout=5.0)