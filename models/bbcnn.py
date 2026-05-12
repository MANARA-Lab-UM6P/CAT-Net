"""Block‑Based CNN (BBCNN) model for segment‑level overlapping speech detection.

This module implements a simple convolutional neural network inspired by
block‑based approaches.  The model operates on sequences of MFCC feature
vectors treated as segments and produces a single logit per segment.
It is intended as a baseline for comparison with more sophisticated
architectures like CAT‑Net.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization for 1‑D convolutional features.

    Input shape:  (B, C, T)
    Normalises over *both* channels and time for each sample:
        mean, var computed over (C, T) per batch element.
    """
    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        mean = x.mean(dim=(1, 2), keepdim=True)  # (B,1,1)
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)  # (B,1,1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.gamma + self.beta


class BBCNN(nn.Module):
    """Block‑Based CNN model for overlapping speech detection.

    This model is designed to operate on segments (e.g. 5‑second or
    3‑second windows) with a single label per segment.  The network
    comprises an initial projection convolution, a stack of convolutional
    blocks each followed by global layer normalisation, ReLU and
    max‑pooling, and two fully connected layers to produce a segment
    logit.

    Parameters
    ----------
    in_channels : int, optional
        Dimensionality of the input feature vector per frame (e.g. 39
        for 13‑dimensional MFCCs with first and second derivatives).
    conv_channels : int, optional
        Number of channels in the convolutional layers.  Default is 256.
    num_blocks : int, optional
        Number of convolutional blocks to apply.  Each block halves the
        temporal resolution via max‑pooling.
    kernel_size : int, optional
        Kernel size for the 1‑D convolutions.
    fc_hidden : int, optional
        Dimension of the intermediate fully connected layer.
    out_classes : int, optional
        Number of output classes.  For binary classification set to 1.
    """

    def __init__(
        self,
        in_channels: int = 39,
        conv_channels: int = 256,
        num_blocks: int = 5,
        kernel_size: int = 3,
        fc_hidden: int = 128,
        out_classes: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.fc_hidden = fc_hidden
        self.out_classes = out_classes

        # 1) Initial projection convolution: 256 channels, k=3, stride=1, same padding + ReLU
        self.proj = nn.Sequential(
            nn.Conv1d(
                in_channels,
                conv_channels,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,  # same temporal size
            ),
            nn.ReLU(),
        )

        # 2) Stack of blocks: Conv -> GlobalLayerNorm -> ReLU -> MaxPool1d
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        conv_channels,
                        conv_channels,
                        kernel_size,
                        stride=1,
                        padding=kernel_size // 2,  # keep temporal size before pooling
                    ),
                    GlobalLayerNorm(conv_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # 3) Two FC layers for the final decision
        self.fc1 = nn.Linear(conv_channels, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, T, F)`` where ``F`` equals
            ``in_channels``.  The time dimension ``T`` corresponds to the
            number of frames in the segment (e.g. 500 frames for 5 seconds
            of audio with a 10 ms frame shift).

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B,)`` if ``out_classes == 1`` or
            ``(B, out_classes)`` otherwise.
        """
        # (B, T, F) -> (B, F, T) for Conv1d
        out = x.transpose(1, 2)

        # Initial conv + ReLU
        out = self.proj(out)  # (B, conv_channels, T)

        # Blocks with Conv + GlobalLayerNorm + ReLU + MaxPool
        for block in self.blocks:
            out = block(out)  # temporal length shrinks by factor ~2 each block

        # Global mean pooling over time dimension
        out = out.mean(dim=-1)  # (B, conv_channels)

        # Two FC layers
        out = F.relu(self.fc1(out))  # (B, fc_hidden)
        out = self.fc2(out)  # (B, out_classes)

        if self.out_classes == 1:
            return out.squeeze(-1)  # (B,)
        return out  # (B, out_classes)