"""CRNN (Convolutional Recurrent Neural Network) classifier for overlapping speech detection.

This module defines :class:`CRNNClassifier`, a lightweight neural network that
combines 2D convolutions on a time–frequency representation with an LSTM
layer.  It is intended for frame‑wise binary classification of overlapping
speech versus single‑speaker speech.  The model first applies two
convolutional layers across the time and frequency dimensions, flattens
the frequency dimension, then feeds the resulting sequences into an LSTM
followed by a linear projection to a single logit per frame.

Example usage::

    from models import CRNNClassifier
    import torch

    model = CRNNClassifier(in_channels=1, freq_dim=80)
    batch = torch.randn(4, 100, 80)  # (batch, time, freq)
    logits = model(batch)
    # logits shape: (4, 100)

This model does not include pooling or dropout; feel free to extend it as
needed for your application.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CRNNClassifier(nn.Module):
    """Simple CRNN for frame‑wise overlapping speech detection.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels.  For mel spectrograms this is 1.
    freq_dim : int, optional
        Number of frequency bins in the input spectrogram.  For example,
        80 for an 80‑band mel or gammatone spectrogram.
    conv_channels : tuple of int, optional
        Tuple specifying the number of channels for the two convolutional
        layers.  The first value is the number of output channels of the
        first convolution and the second value is the output channels of
        the second convolution.
    lstm_hidden : int, optional
        Number of hidden units in the LSTM.
    bidirectional : bool, optional
        If ``True``, a bidirectional LSTM is used.
    """

    def __init__(
        self,
        in_channels: int = 1,
        freq_dim: int = 80,
        conv_channels: tuple[int, int] = (64, 32),
        lstm_hidden: int = 128,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.freq_dim = freq_dim
        self.conv_channels = conv_channels
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        # Convolutional front‑end
        # First conv: preserves time and frequency dimensions with padding
        self.conv1 = nn.Conv2d(
            in_channels,
            conv_channels[0],
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.act1 = nn.ReLU()
        # Second conv
        self.conv2 = nn.Conv2d(
            conv_channels[0],
            conv_channels[1],
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.act2 = nn.ReLU()
        # After the conv layers, flatten the frequency dimension and
        # treat the time dimension as sequence length.
        flattened_dim = conv_channels[1] * freq_dim
        self.lstm = nn.LSTM(
            flattened_dim,
            lstm_hidden,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.linear = nn.Linear(lstm_out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, time, freq)``.

        Returns
        -------
        torch.Tensor
            Logit tensor of shape ``(batch, time)``.  Use
            ``BCEWithLogitsLoss`` for training.
        """
        # reshape to (batch, 1, time, freq)
        x = x.unsqueeze(1)
        # apply convolutions
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        # x shape: (batch, conv_channels[1], time, freq)
        batch_size, ch, t, f = x.shape
        # flatten frequency and channel into feature dimension
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch, time, ch, freq)
        x = x.view(batch_size, t, ch * f)
        # LSTM
        out, _ = self.lstm(x)
        logits = self.linear(out).squeeze(-1)
        return logits