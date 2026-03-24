"""
Implementation of the CAT_Net model for overlapping speech detection.

This module defines all components required for the proposed
Channel‑Aware Temporal Network (CAT‑Net) architecture, including a
channel‑ and time‑sensitive squeeze‑and‑excitation layer, a stack of
temporal convolutional blocks with dilation and residual connections, and
a global layer normalisation.  The final model processes sequences of
acoustic feature frames and outputs a sequence of logits indicating the
likelihood of overlapping speech at each frame.

The architecture corresponds to the model described in

  Y. Terraf and Y. Iraqi, “CAT‑Net: A Channel and Self‑Attention TCN for
  Robust Frame‑Level Overlapping Speech Detection,” IEEE
  Transactions on Audio, Speech and Language Processing, vol. 34,
  pp. 1184–1199, 2026【697161440799525†L138-L147】.

Examples
--------

Instantiate the model and perform a forward pass on a batch of feature
sequences:

>>> import torch
>>> from catnet.models import CAT_Net
>>> batch_size, channels, frames = 4, 80, 100
>>> x = torch.randn(batch_size, channels, frames)
>>> model = CAT_Net(in_chan=channels)
>>> logits = model(x)
>>> logits.shape
torch.Size([4, 100])

The output ``logits`` tensor has shape ``(batch_size, time_frames)`` for
binary classification (``out_classes=1``).  A positive logit indicates
overlapping speech; apply a sigmoid to obtain probabilities.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class GlobalLayerNorm(nn.Module):
    """Global layer normalisation across channel and time dimensions.

    This layer normalises an input of shape ``(batch, channels, time)`` by
    computing the mean and variance across both the channel and time axes for
    each sample in the batch.  Learnable scale and bias parameters are
    applied per channel.  It is equivalent to the global layer
    normalisation used in ConvTasNet and related models.
    """

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma.view(1, -1, 1) * x_norm + self.beta.view(1, -1, 1)


def get_norm(norm_type: str) -> Callable[[int], nn.Module]:
    """Return a normalisation layer factory based on ``norm_type``.

    Parameters
    ----------
    norm_type:
        One of ``'gLN'`` for global layer normalisation or ``'bN'`` for
        batch normalisation.

    Returns
    -------
    Callable[[int], nn.Module]
        A function that takes the number of channels and returns a
        corresponding normalisation layer.
    """
    if norm_type == 'gLN':
        return lambda c: GlobalLayerNorm(c)
    elif norm_type == 'bN':
        return lambda c: nn.BatchNorm1d(c)
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")


class ChannelTimeSenseSELayer(nn.Module):
    """Multi‑scale channel‑time squeeze‑and‑excitation layer.

    This attention mechanism applies three depthwise convolutions with
    different kernel sizes to capture context at multiple temporal scales.
    The outputs are averaged over time, concatenated and passed through
    linear layers to generate channel‑wise gating weights.  The weights
    are then applied multiplicatively to the input.

    Parameters
    ----------
    num_channels:
        Number of input channels.
    reduction_ratio:
        Factor by which the channel dimension is reduced in the internal
        fully connected layer.  A typical value is 2.
    kersize:
        Tuple of three kernel sizes for the depthwise convolutions.
    subband_num:
        Number of frequency sub‑bands to split the channels into.  Set
        ``1`` to treat the entire spectrum as a single band.
    """

    def __init__(self, num_channels: int, reduction_ratio: int = 2,
                 kersize: tuple[int, int, int] = (3, 5, 10),
                 subband_num: int = 1) -> None:
        super().__init__()
        num_channels_reduced = max(1, num_channels // reduction_ratio)
        # depthwise convolutions for different kernel sizes
        self.smallConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[0],
                      padding=0, groups=num_channels // subband_num),
            nn.AdaptiveAvgPool1d(1),
            nn.ReLU(inplace=True)
        )
        self.middleConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[1],
                      padding=0, groups=num_channels // subband_num),
            nn.AdaptiveAvgPool1d(1),
            nn.ReLU(inplace=True)
        )
        self.largeConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[2],
                      padding=0, groups=num_channels // subband_num),
            nn.AdaptiveAvgPool1d(1),
            nn.ReLU(inplace=True)
        )
        # linear layers to combine multi‑scale features and produce gates
        self.feature_concate_fc = nn.Linear(3, 1, bias=True)
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # input_tensor shape: (batch, channels, time)
        small_feature = self.smallConv1d(input_tensor)
        middle_feature = self.middleConv1d(input_tensor)
        large_feature = self.largeConv1d(input_tensor)
        # concatenate along time dimension (now length 1)
        feature = torch.cat([small_feature, middle_feature, large_feature], dim=2)
        squeeze_tensor = self.feature_concate_fc(feature)[..., 0]
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        # reshape for broadcasting along the time axis
        return input_tensor * fc_out_2.unsqueeze(-1)


class Conv1DBlock(nn.Module):
    """Temporal convolutional block with an optional delta projection.

    This block performs an optional delta projection followed by a
    pointwise convolution, depthwise convolution, normalisation and
    PReLU non‑linearities.  The output of the block is a residual that
    will be added to the input outside the block.

    Parameters
    ----------
    in_chan:
        Number of input channels.
    hid_chan:
        Number of hidden channels (width of the depthwise convolution).
    kernel_size:
        Kernel size of the depthwise convolution.
    padding:
        Padding applied before the depthwise convolution.  Should be
        computed to maintain the length of the sequence.
    dilation:
        Dilation factor of the depthwise convolution.
    norm_type:
        Normalisation type: ``'gLN'`` for global layer norm or ``'bN'`` for batch norm.
    delta:
        Whether to append a delta projection along the time dimension.  This
        introduces a second branch to the residual connection and can
        improve performance for some tasks.
    """

    def __init__(self, in_chan: int, hid_chan: int, kernel_size: int,
                 padding: int, dilation: int, norm_type: str = "bN",
                 delta: bool = False) -> None:
        super().__init__()
        conv_norm = get_norm(norm_type)
        self.delta = delta
        if delta:
            # projection of the sequence along the time axis using a linear layer
            self.linear = nn.Linear(in_chan, in_chan)
            self.linear_norm = conv_norm(in_chan * 2)
            in_bottle = in_chan * 2
        else:
            in_bottle = in_chan
        # 1x1 convolution to expand channels
        self.in_conv1d = nn.Conv1d(in_bottle, hid_chan, 1)
        # depthwise convolution
        self.depth_conv1d = nn.Conv1d(hid_chan, hid_chan, kernel_size,
                                       padding=padding, dilation=dilation,
                                       groups=hid_chan)
        self.shared_block = nn.Sequential(
            self.in_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan),
            self.depth_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan)
        )
        # pointwise convolution back to input channel size
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.delta:
            # project along time dimension using a linear layer
            delta = self.linear(x.transpose(1, -1)).transpose(1, -1)
            x_cat = torch.cat((x, delta), dim=1)
            x_cat = self.linear_norm(x_cat)
        else:
            x_cat = x
        shared_out = self.shared_block(x_cat)
        res_out = self.res_conv(shared_out)
        return res_out


class SoftAttentionPooling(nn.Module):
    """Soft attention pooling over the temporal dimension.

    Given an input tensor of shape ``(batch, channels, time)``, this layer
    learns attention weights over the time dimension and produces a pooled
    representation of shape ``(batch, channels)``.  It is useful when
    reducing a variable‑length sequence to a fixed‑dimensional vector for
    classification or aggregation.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Tanh(),
            nn.Linear(channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        # compute attention over time: first transpose to (batch, time, channels)
        attn_scores = self.attention(x.permute(0, 2, 1))  # (batch, time, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, time, 1)
        pooled = torch.sum(x * attn_weights.permute(0, 2, 1), dim=2)  # (batch, channels)
        return pooled


class CAT_Net(nn.Module):
    """Channel‑Aware Temporal Network (CAT_Net) for overlapping speech detection.

    Parameters
    ----------
    in_chan:
        Number of input feature channels (e.g. 80 for gammatone features).
    n_blocks:
        Number of convolutional blocks per repeat in the TCN.
    bn_chan:
        Bottleneck channel dimension; controls the width of the TCN.
    n_repeats:
        Number of times the block stack is repeated.
    hid_chan:
        Hidden channel dimension used inside convolutional blocks.
    kernel_size:
        Size of the depthwise convolution kernel.
    norm_type:
        Normalisation type: 'gLN' for global layer norm or 'bN' for batch norm.
    out_classes:
        Number of output classes.  Set to 1 for binary classification.
    """

    def __init__(self, in_chan: int = 128, n_blocks: int = 5, bn_chan: int = 64,
                 n_repeats: int = 3, hid_chan: int = 128, kernel_size: int = 3,
                 norm_type: str = "gLN", out_classes: int = 1, **kwargs) -> None:
        super().__init__()
        self.in_chan = in_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.out_classes = out_classes
        # channel‑time attention
        self.attention = ChannelTimeSenseSELayer(in_chan)
        # bottleneck layer
        norm_layer = get_norm(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(norm_layer, bottleneck_conv)
        # temporal convolutional network (TCN)
        tcn = []
        for _ in range(n_repeats):
            for b in range(n_blocks):
                # compute padding to maintain sequence length given dilation
                padding = (kernel_size - 1) * 2 ** b // 2
                tcn.append(Conv1DBlock(bn_chan, hid_chan, kernel_size,
                                        padding=padding, dilation=2 ** b,
                                        norm_type=norm_type))
        self.TCN = nn.ModuleList(tcn)
        # output convolution: maps bn_chan to out_classes
        self.out = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(bn_chan, out_classes, 1)
        )
        # optional pooling and linear layer for sequence summarisation
        self.soft_attention = SoftAttentionPooling(bn_chan)
        self.fc_final = nn.Linear(bn_chan, out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        # apply channel‑time attention
        x = self.attention(x)
        # bottleneck projection
        x = self.bottleneck(x)
        # apply stacked dilated convolution blocks
        for block in self.TCN:
            residual = block(x)
            x = x + residual
        logits = self.out(x)  # (batch, out_classes, time)
        # squeeze channel dimension if out_classes == 1
        if logits.shape[1] == 1:
            return logits[:, 0, :]
        return logits.squeeze(1)