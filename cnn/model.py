"""
Multi-Task CNN for Taxonomic Classification.

Architecture:
- 4 convolutional blocks (Conv → BatchNorm → ReLU → MaxPool)
- Global Average Pooling
- 3 classification heads (class, order, family)
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
    Multi-task CNN for taxonomic classification.

    Takes 32×32 RGB images (DNA sequences converted to pixels) and outputs
    predictions for class, order, and family taxonomic levels.

    Architecture:
        Input: (batch, 3, 32, 32)
        Conv Block 1: 3 → 32 channels, output: (batch, 32, 16, 16)
        Conv Block 2: 32 → 64 channels, output: (batch, 64, 8, 8)
        Conv Block 3: 64 → 128 channels, output: (batch, 128, 4, 4)
        Conv Block 4: 128 → 256 channels, output: (batch, 256, 2, 2)
        Global Average Pool: (batch, 256, 1, 1) → (batch, 256)
        Dropout: 0.4
        Classification Heads:
            - class_head: 256 → num_classes
            - order_head: 256 → num_orders
            - family_head: 256 → num_families
    """

    def __init__(
        self,
        num_classes: int = 19,
        num_orders: int = 130,
        num_families: int = 1989,
        dropout: float = 0.4,
    ):
        """
        Initialize the model.

        Args:
            num_classes: Number of taxonomic classes (within Arthropoda)
            num_orders: Number of taxonomic orders
            num_families: Number of taxonomic families
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

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Classification heads
        self.class_head = nn.Linear(256, num_classes)
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
            Dict with 'class', 'order', 'family' logits
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

        # Classification heads
        return {
            "class": self.class_head(x),
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
        outputs: Dict with 'class', 'order', 'family' logits
        targets: Dict with 'class', 'order', 'family' labels
        weights: Optional dict with loss weights per level

    Returns:
        Total weighted loss
    """
    if weights is None:
        weights = {"class": 1.0, "order": 1.0, "family": 0.5}

    loss = torch.tensor(0.0, device=outputs["class"].device)

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

    print(f"Class output shape: {outputs['class'].shape}")  # (4, 19)
    print(f"Order output shape: {outputs['order'].shape}")  # (4, 130)
    print(f"Family output shape: {outputs['family'].shape}")  # (4, 1989)
