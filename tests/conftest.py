import os
import sys
import types

# Make the top-level modules (graph.py, TransNet_train.py) importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# torch_scatter is imported by TransNet_train.py but never used; it needs a
# compiled wheel matching the torch build, so stub it when it is not installed.
try:
    import torch_scatter  # noqa: F401
except ImportError:
    stub = types.ModuleType("torch_scatter")
    stub.scatter_add = None
    sys.modules["torch_scatter"] = stub
