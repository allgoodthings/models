"""
CodeFormer architecture files.

These are copied from https://github.com/sczhou/CodeFormer/tree/master/basicsr/archs
to avoid PYTHONPATH conflicts with the pip-installed basicsr package.

Original license: MIT (see CodeFormer repo)
"""

from .vqgan_arch import VQAutoEncoder, VectorQuantizer
from .codeformer_arch import CodeFormer

__all__ = ["VQAutoEncoder", "VectorQuantizer", "CodeFormer"]
