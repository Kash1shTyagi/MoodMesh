import time
import pytest
from utils.async_processor import AsyncProcessor

def process_item(x):
    time.sleep(0.05)
    return x * 2

@pytest.mark.asyncio
async def test_async_processor_integration():
    processor = AsyncProcessor(max_workers=4)
    
    # Submit multiple tasks
    futures = [processor.submit(process_item, i) for i in range(10)]
    
    # Submit async
    async_result = await processor.submit_async(process_item, 10)
    
    # Collect results
    results = [f.result() for f in futures]
    assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    assert async_result == 20
    
    processor.shutdown()