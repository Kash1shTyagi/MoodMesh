import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Set environment variables for testing
@pytest.fixture(scope="session", autouse=True)
def setup_env():
    os.environ["DATA_DIR"] = "/test/data"
    os.environ["MODELS_DIR"] = "/test/models"