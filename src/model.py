"""
Архитектура FusionResNetSEMTLCNN — двухпотоковая нейросеть для акустической
диагностики неисправностей малых БАС.

ВНИМАНИЕ: код этого модуля воспроизводит модель из notebooks/final_model.ipynb
1-в-1, чтобы загружаемые веса fusion_resnet_se_mtl_cnn_weights.pth были совместимы.

Архитектура:
    Мел-вход (B, 20, 64) → unsqueeze → (B, 1, 20, 64)
    conv1:  Conv2d(1→32, 3×3) + BN + ReLU + MaxPool2d(2,2)  → (B, 32, 10, 32)
    stage1: 2 × SEResidualBlock(32→64,  stride=1)           → (B, 64, 10, 32)
    stage2: 2 × SEResidualBlock(64→128, stride=2)           → (B, 128, 5, 16)
    stage3: 1 × SEResidualBlock(128→192, stride=2)          → (B, 192, 3, 8)
    AdaptiveAvgPool2d(1×1) → flatten → (B, 192)

    STFT-вход (B, 32) — конкатенируется к (B, 192): (B, 224)

    shared_fc: 224 → 128 → 96 → 64 (BN + ReLU + Dropout 0.25)
    fault_head:    64 → 96 → 9   (ReLU + Dropout 0.15)
    maneuver_head: 64 → 48 → 6

Источники:
    He K. et al. Deep Residual Learning for Image Recognition // CVPR 2016.
    Hu J. et al. Squeeze-and-Excitation Networks // CVPR 2018.
    Caruana R. Multitask Learning // Machine Learning 28(1), 1997.
"""
import torch
import torch.nn as nn


class SELayer(nn.Module):
    """Squeeze-and-Excitation для двумерных свёрток (Hu et al., 2018)."""

    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEResidualBlock(nn.Module):
    """Остаточный блок ResNet 2D с интегрированным SE-механизмом."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: nn.Module = None, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SELayer(out_channels, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class FusionResNetSEMTLCNN(nn.Module):
    """Двухпотоковая модель: мел-спектрограмма (2D) + STFT-статистики (1D).

    Args:
        n_fault_classes: число классов неисправностей (по умолчанию 9).
        n_maneuver_classes: число классов манёвров (по умолчанию 6).
        win_len: число временных кадров мел-спектрограммы (по умолчанию 20).
        n_mels: число мел-полос (по умолчанию 64).
        n_custom: размерность STFT-статистик (по умолчанию 32).
    """

    def __init__(
        self,
        n_fault_classes: int = 9,
        n_maneuver_classes: int = 6,
        win_len: int = 20,
        n_mels: int = 64,
        n_custom: int = 32,
    ):
        super().__init__()

        # ── Свёрточная часть мел-ветви ──────────────────────────────────────
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (20, 64) → (10, 32)
        )
        self.stage1 = self._make_stage(32, 64, 2)               # → (10, 32)
        self.stage2 = self._make_stage(64, 128, 2, stride=2)     # → (5, 16)
        self.stage3 = self._make_stage(128, 192, 1, stride=2)    # → (3, 8)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_out_dim = 192

        # ── Общие полносвязные слои (fusion) ────────────────────────────────
        self.shared_fc = nn.Sequential(
            nn.Linear(self.cnn_out_dim + n_custom, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(128, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(96, 64),
        )

        # ── Две головы (MTL) ────────────────────────────────────────────────
        self.fault_head = nn.Sequential(
            nn.Linear(64, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(96, n_fault_classes),
        )
        self.maneuver_head = nn.Sequential(
            nn.Linear(64, 48),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(48, n_maneuver_classes),
        )

        self._initialize_weights()

    def _make_stage(self, in_channels: int, out_channels: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [SEResidualBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(SEResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, mel_spec: torch.Tensor, custom_feat: torch.Tensor):
        """
        Args:
            mel_spec: тензор мел-спектрограмм (B, win_len, n_mels) или
                (B, 1, win_len, n_mels). Если 3D — будет добавлен канал.
            custom_feat: тензор STFT-статистик (B, n_custom).
        Returns:
            (fault_logits, maneuver_logits).
        """
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)

        x = self.conv1(mel_spec)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x).view(x.size(0), -1)

        combined = torch.cat([x, custom_feat], dim=1)
        shared = self.shared_fc(combined)
        return self.fault_head(shared), self.maneuver_head(shared)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = FusionResNetSEMTLCNN()
    print(f"Parameters: {count_parameters(model):,}")  # ~1,304,447
    mel = torch.randn(4, 20, 64)
    stft = torch.randn(4, 32)
    fault, maneuver = model(mel, stft)
    print(f"Fault logits: {fault.shape}")          # (4, 9)
    print(f"Maneuver logits: {maneuver.shape}")    # (4, 6)
