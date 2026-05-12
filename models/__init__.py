"""Expose model classes at the package level.

This module makes it easy to import all supported classifiers from a single
namespace.  For example::

    from models import CAT_Net, LSTMClassifier, CRNNClassifier, TcnClassifier, BBCNN

Each of the classes implements a PyTorch ``nn.Module`` for frame‑level
overlapping speech detection.  See the individual docstrings for details.
"""

# CAT‑Net model (primary model from the paper)
from .cat_net import CAT_Net  # noqa: F401

# Baseline models
from .lstm_classifier import LSTMClassifier  # noqa: F401
from .crnn_classifier import CRNNClassifier  # noqa: F401
from .tcn_transformer_classifier import TcnClassifier  # noqa: F401
from .bbcnn import BBCNN  # noqa: F401