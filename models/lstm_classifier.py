"""Unidirectional LSTM classifier for frame‑level overlapping speech detection.

This module defines :class:`LSTMClassifier`, a simple recurrent neural
network model that processes sequences of feature frames and outputs a
logit for each frame.  The architecture consists of a single LSTM
layer followed by three fully connected (dense) layers and a final
projection layer to produce a single logit per time step.  The default
configuration uses 512 hidden units in the LSTM and dense layers of
sizes 1024, 512 and 256.

Example usage::

    from models import LSTMClassifier
    import torch

    # create model for 80‑dimensional input features
    model = LSTMClassifier(input_size=80)
    batch = torch.randn(2, 100, 80)  # (batch, time, features)
    logits = model(batch)
    # logits shape: (2, 100)
    # Use BCEWithLogitsLoss on logits

This model is a baseline for comparison and may not achieve the same
performance as the CAT‑Net architecture on challenging acoustic
conditions.
"""

from __future__ import annotations

from typing import Sequence, Optional, Tuple

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """Unidirectional LSTM with three dense layers for frame‑level classification.

    The network architecture is::

        Input → LSTM(hidden_size=512) → Dense(1024) → Dense(512) → Dense(256) → Logit(1)

    Parameters
    ----------
    input_size:
        Dimension of the input feature vector at each time step.
    lstm_hidden_size:
        Number of hidden units in the LSTM layer.  Default is 512.
    dense_sizes:
        Sizes of the intermediate dense layers.  Default is ``(1024, 512, 256)``.
    """

    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int = 512,
        dense_sizes: Sequence[int] = (1024, 512, 256),
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.dense_sizes = dense_sizes

        # 1‑layer unidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # Dense layers
        layers = []
        in_dim = lstm_hidden_size
        for dim in dense_sizes:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            in_dim = dim
        self.denses = nn.Sequential(*layers)

        # Final output: 1 logit per frame
        self.out = nn.Linear(in_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute frame‑wise logits.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, time, features)``.
        hidden:
            Optional initial hidden and cell states for the LSTM.  If provided,
            it should be a tuple ``(h0, c0)`` where each tensor has shape
            ``(1, batch, lstm_hidden_size)``.

        Returns
        -------
        torch.Tensor
            Logit tensor of shape ``(batch, time)``.  Use
            ``BCEWithLogitsLoss`` for training.
        """
        # LSTM expects (batch, time, features)
        out, _ = self.lstm(x, hidden)  # (batch, time, lstm_hidden_size)
        out = self.denses(out)  # (batch, time, last_dense_size)
        logits = self.out(out).squeeze(-1)
        return logits  # (batch, time)