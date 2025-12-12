"""
Multi-Task CNN for Insecta Taxonomic Classification.

Architecture:
- 4 convolutional blocks (Conv → BatchNorm → ReLU → MaxPool)
- Global Average Pooling
- 2 classification heads (order, family)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block: Conv2D → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class MultiTaskCNN(nn.Module):
    """
    Multi-task CNN for Insecta taxonomic classification.

    Takes 32×32 RGB images (DNA sequences converted to pixels) and outputs
    predictions for order and family taxonomic levels.

    Architecture:
        Input: (batch, 3, 32, 32)
        Conv Block 1: 3 → 32 channels, output: (batch, 32, 16, 16)
        Conv Block 2: 32 → 64 channels, output: (batch, 64, 8, 8)
        Conv Block 3: 64 → 128 channels, output: (batch, 128, 4, 4)
        Conv Block 4: 128 → 256 channels, output: (batch, 256, 2, 2)
        Global Average Pool: (batch, 256, 1, 1) → (batch, 256)
        Dropout: 0.5
        Classification Heads:
            - order_head: 256 → num_orders
            - family_head: 256 → num_families
    """

    def __init__(
        self,
        num_orders: int = 27,
        num_families: int = 917,
        dropout: float = 0.5,
    ):
        """
        Initialize the model.

        Args:
            num_orders: Number of taxonomic orders (Insecta)
            num_families: Number of taxonomic families (Insecta)
            dropout: Dropout rate before classification heads
        """
        super().__init__()

        # Convolutional encoder
        self.conv1 = ConvBlock(3, 32)  # 32 → 16
        self.conv2 = ConvBlock(32, 64)  # 16 → 8
        self.conv3 = ConvBlock(64, 128)  # 8 → 4
        self.conv4 = ConvBlock(128, 256)  # 4 → 2

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Dropout (increased to 0.5 to reduce overfitting)
        self.dropout = nn.Dropout(dropout)

        # Classification heads (order + family only)
        self.order_head = nn.Linear(256, num_orders)
        self.family_head = nn.Linear(256, num_families)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 32, 32)

        Returns:
            Dict with 'order', 'family' logits
        """
        # Encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 256)

        # Dropout
        x = self.dropout(x)

        # Classification heads (order + family)
        return {
            "order": self.order_head(x),
            "family": self.family_head(x),
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def multi_task_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: dict[str, float] = None,
) -> torch.Tensor:
    """
    Compute weighted multi-task cross-entropy loss.

    Args:
        outputs: Dict with 'order', 'family' logits
        targets: Dict with 'order', 'family' labels
        weights: Optional dict with loss weights per level

    Returns:
        Total weighted loss
    """
    if weights is None:
        weights = {"order": 1.0, "family": 0.5}

    loss = torch.tensor(0.0, device=outputs["order"].device)

    for level, weight in weights.items():
        level_loss = nn.functional.cross_entropy(outputs[level], targets[level])
        loss = loss + weight * level_loss

    return loss


if __name__ == "__main__":
    # Test model
    model = MultiTaskCNN()
    print(f"Model parameters: {model.count_parameters():,}")

    # Test forward pass
    x = torch.randn(4, 3, 32, 32)
    outputs = model(x)

    print(f"Order output shape: {outputs['order'].shape}")  # (4, 27)
    print(f"Family output shape: {outputs['family'].shape}")  # (4, 917)
