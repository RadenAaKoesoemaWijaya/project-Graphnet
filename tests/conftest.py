import sys
import pytest

@pytest.fixture(autouse=True)
def clean_torch_sys_modules():
    if 'torch' in sys.modules and not hasattr(sys.modules['torch'], 'Tensor'):
        sys.modules.pop('torch', None)
    yield
    if 'torch' in sys.modules and not hasattr(sys.modules['torch'], 'Tensor'):
        sys.modules.pop('torch', None)
