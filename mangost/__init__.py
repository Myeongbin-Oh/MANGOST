"""🥭 MANGOST: multi-section spatial domains with spatially adaptive gene network mixtures."""
from . import datasets, graph, plotting as pl, tools as tl
from .trainer import MANGOST

__all__ = ["MANGOST", "datasets", "graph", "pl", "tl"]
__version__ = "0.1.0"
