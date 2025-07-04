import time
import pytest
import threading
from utils.async_processor import AsyncProcessor, AsyncBatchProcessor

def square(x):
    return x * x

def slow_function(x):
    time.sleep(0.1)
    return x * 2

def batch_process(items):
    return [item[0] * 2 for item in items] 

def slow_batch_function(items):
    time.sleep(0.1)
    return [item[0] * 2 for item in items] 

@pytest.fixture
def processor():
    return AsyncProcessor(max_workers=2)

@pytest.fixture
def batch_processor():
    return AsyncBatchProcessor(
        batch_size=4,
        max_workers=4,
        max_queue_size=100
    )

@pytest.fixture
def controllable_processor():
    return AsyncBatchProcessor(
        batch_size=1,
        max_workers=1,
        max_queue_size=1
    )

def test_async_submit(processor):
    future = processor.submit(square, 5)
    assert future.result(timeout=1.0) == 25

@pytest.mark.asyncio
async def test_async_submit_async(processor):
    result = await processor.submit_async(square, 6)
    assert result == 36

def test_async_multiple(processor):
    futures = [processor.submit(square, i) for i in range(5)]
    results = [f.result(timeout=1.0) for f in futures]
    assert results == [0, 1, 4, 9, 16]

def test_async_shutdown(processor):
    processor.shutdown()
    with pytest.raises(RuntimeError):
        processor.submit(square, 5)

def test_batch_processing(batch_processor):
    args_list = [[1], [2], [3], [4]]
    futures = batch_processor.submit_batch(batch_process, args_list)
    results = [f.result(timeout=1.0) for f in futures]
    assert results == [2, 4, 6, 8]

def test_batch_timeout(batch_processor):
    args_list = [[1], [2], [3]]
    futures = batch_processor.submit_batch(slow_batch_function, args_list)
    results = [f.result(timeout=2.0) for f in futures]  # Increased timeout
    assert results == [2, 4, 6]

def test_batch_large_queue(controllable_processor):
    future1 = controllable_processor.submit_batch(batch_process, [[1]])[0]
    
    future2 = controllable_processor.submit_batch(batch_process, [[2]])[0]
    
    assert future1.result(timeout=1.0) == 2
    
    with pytest.raises(RuntimeError) as exc_info:
        future2.result(timeout=0.5)
    
    assert "Batch queue full" in str(exc_info.value)
    
    controllable_processor.shutdown()   