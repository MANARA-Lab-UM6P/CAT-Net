"""Temporal Convolutional Network (TCN) and Transformer classifier.

This module defines :class:`TcnClassifier`, which combines a stack of
dilated 1‑D convolutional blocks (inspired by Conv‑TasNet) with a
Transformer encoder to capture both local and long‑range temporal
dependencies.  The model takes a sequence of feature frames of shape
``(batch, channels, time)`` and outputs a logit for each frame
indicating the likelihood of overlapping speech.

The default hyperparameters mirror those used in the CAT‑Net paper,
although the Transformer portion can be adjusted via class arguments.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

try:
    # asteroid.masknn.norms provides global and channelwise layer norms
    from asteroid.masknn import norms

    def get_norm(norm_type: str):
        return norms.get(norm_type)

except ImportError:
    # fall back to simple implementations if asteroid is not installed
    def get_norm(norm_type: str):
        if norm_type == "gLN":
            # global LayerNorm over (channels, time)
            return lambda c: nn.LayerNorm([c, 1])
        elif norm_type == "bN":
            return lambda c: nn.BatchNorm1d(c)
        else:
            return lambda c: nn.Identity()


class Conv1DBlock(nn.Module):
    """A one‑dimensional convolutional block with residual connection.

    Structure (Conv‑TasNet‑like):
      - 1×1 pointwise convolution to expand channels
      - depthwise dilated convolution
      - normalisation + PReLU
      - residual 1×1 projection back to input dimension
    """

    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        norm_type: str = "gLN",
    ) -> None:
        super().__init__()
        conv_norm = get_norm(norm_type)

        self.in_conv = nn.Conv1d(in_chan, hid_chan, 1)
        self.depth_conv = nn.Conv1d(
            hid_chan,
            hid_chan,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hid_chan,  # depthwise
        )

        self.shared = nn.Sequential(
            self.in_conv,
            nn.PReLU(),
            conv_norm(hid_chan),
            self.depth_conv,
            nn.PReLU(),
            conv_norm(hid_chan),
        )

        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shared(x)
        residual = self.res_conv(residual)
        return residual


class TcnClassifier(nn.Module):
    """TCN + Transformer classifier for overlapping speech detection.

    Parameters
    ----------
    in_chan : int, optional
        Number of input feature channels (e.g. 80 for mel features).
    bn_chan : int, optional
        Bottleneck channel dimension (also used as Transformer ``d_model``).
    hid_chan : int, optional
        Hidden channel dimension in the convolutional blocks.
    n_blocks : int, optional
        Number of blocks per repeat in the TCN.
    n_repeats : int, optional
        Number of repeats of the block stack.
    kernel_size : int, optional
        Kernel size for the depthwise convolutions.
    norm_type : str, optional
        Normalisation type: 'gLN' for global layer norm or 'bN' for batch norm.
    n_heads : int, optional
        Number of attention heads in the Transformer encoder.
    n_transformer_layers : int, optional
        Number of Transformer encoder layers.
    dim_feedforward : int, optional
        Dimension of the feedforward network within each Transformer encoder layer.
    dropout : float, optional
        Dropout probability within the Transformer encoder.
    """

    def __init__(
        self,
        in_chan: int = 80,
        bn_chan: int = 64,
        hid_chan: int = 128,
        n_blocks: int = 5,
        n_repeats: int = 3,
        kernel_size: int = 3,
        norm_type: str = "gLN",
        n_heads: int = 4,
        n_transformer_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_chan = in_chan
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.n_heads = n_heads
        self.n_transformer_layers = n_transformer_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        # Bottleneck: normalisation + 1×1 conv
        norm_layer = get_norm(norm_type)(in_chan)
        self.bottleneck = nn.Sequential(norm_layer, nn.Conv1d(in_chan, bn_chan, 1))

        # Build TCN blocks
        blocks: List[nn.Module] = []
        for _ in range(n_repeats):
            for b in range(n_blocks):
                dilation = 2 ** b
                padding = (kernel_size - 1) * dilation // 2
                blocks.append(
                    Conv1DBlock(
                        bn_chan,
                        hid_chan,
                        kernel_size,
                        padding=padding,
                        dilation=dilation,
                        norm_type=norm_type,
                    )
                )
        self.TCN = nn.ModuleList(blocks)

        # Transformer encoder after TCN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bn_chan,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers,
        )

        # Final 1×1 convolution to output single logit per frame
        self.output = nn.Conv1d(bn_chan, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute frame‑wise logits.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, channels, time)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, time)``.
        """
        x = self.bottleneck(x)

        # TCN stack
        for block in self.TCN:
            res = block(x)
            x = x + res

        # Transformer after TCN
        # (batch, channels, time) -> (batch, time, channels)
        x_t = x.transpose(1, 2)
        x_t = self.transformer(x_t)
        # back to (batch, channels, time)
        x = x_t.transpose(1, 2)

        # Frame‑wise output logits
        logits = self.output(x)  # (batch, 1, time)
        return logits[:, 0, :]  # (batch, time)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Simple test demonstrating instantiation and forward pass
    model = TcnClassifier()
    n_params = count_parameters(model)
    print(f"Number of trainable parameters: {n_params:,}")
    # Dummy input
    batch_size = 2
    n_channels = model.in_chan
    n_frames = 100
    dummy_input = torch.randn(batch_size, n_channels, n_frames)
    logits = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {logits.shape}  # (batch, time)")