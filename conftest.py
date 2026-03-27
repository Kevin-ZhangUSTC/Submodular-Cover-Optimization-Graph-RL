"""
pytest configuration: add the repository root to sys.path so that
``src`` and top-level modules (``config``) are importable from tests.
"""
import sys
import os

# Insert the repository root so that 'src' and 'config' are importable.
sys.path.insert(0, os.path.dirname(__file__))
