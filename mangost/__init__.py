"""🥭 MANGOST: spatially adaptive mixtures of shared gene association networks for multi-section spatial transcriptomics."""
from . import datasets, graph, plotting as pl, tools as tl
from .trainer import MANGOST

__all__ = ["MANGOST", "datasets", "graph", "pl", "tl"]
__version__ = "0.1.0"
