import time
import asyncio
import pytest
from utils.async_processor import AsyncProcessor

def process_item(x):
    time.sleep(0.01)
    return x * 2

@pytest.mark.asyncio
async def test_async_processor_integration():
    processor = AsyncProcessor(max_workers=4)
    
    futures = [processor.submit(process_item, i) for i in range(10)]
    
    async_result = await processor.submit_async(process_item, 10)
    
    results = [f.result(timeout=1.0) for f in futures]
    assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    assert async_result == 20
    
    processor.shutdown()