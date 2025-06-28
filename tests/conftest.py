import pytest
import os

# Set environment variables for testing
@pytest.fixture(scope="session", autouse=True)
def setup_env():
    os.environ["DATA_DIR"] = "/test/data"
    os.environ["MODELS_DIR"] = "/test/models"