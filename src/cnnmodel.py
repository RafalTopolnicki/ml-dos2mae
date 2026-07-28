import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNModel(nn.Module):
    def __init__(
        self,
        input_size=36,
        n_features=1,
        target_dim=1,
        hidden_channels=32,
        dropout_prob=0.3,
        use_layernorm=True,
        n_layers=2,
    ):
        super(SimpleCNNModel, self).__init__()
        self.use_layernorm = use_layernorm
        self.input_norm = nn.LayerNorm(input_size)

        layers = []
        in_ch = input_size
        for _ in range(n_layers):
            layers += [
                nn.Conv1d(in_ch, hidden_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
            ]
            in_ch = hidden_channels
        self.convs = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_channels + n_features, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, target_dim)

    def forward(self, x, features, return_attn=False):
        if self.use_layernorm:
            x = self.input_norm(x)
        x = x.transpose(1, 2)          # [B, C, L]
        x = self.convs(x)
        x = self.pool(x).squeeze(-1)   # [B, hidden_channels]
        x = torch.cat([x, features], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        if return_attn:
            return x, {}
        return x


# ---------------------------------------------------------------------------
# ResNet 1-D
# ---------------------------------------------------------------------------

class _ResBlock1d(nn.Module):
    def __init__(self, channels, kernel_size=5, dropout_prob=0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class ResNet1dModel(nn.Module):
    def __init__(
        self,
        input_size=36,
        n_features=1,
        target_dim=1,
        hidden_channels=32,
        dropout_prob=0.3,
        use_layernorm=True,
        n_blocks=4,
    ):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.input_norm = nn.LayerNorm(input_size)

        self.stem = nn.Sequential(
            nn.Conv1d(input_size, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[_ResBlock1d(hidden_channels, dropout_prob=dropout_prob) for _ in range(n_blocks)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_channels + n_features, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, target_dim)

    def forward(self, x, features, return_attn=False):
        if self.use_layernorm:
            x = self.input_norm(x)
        x = x.transpose(1, 2)          # [B, C, L]
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)   # [B, hidden_channels]
        x = torch.cat([x, features], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        if return_attn:
            return x, {}
        return x


# ---------------------------------------------------------------------------
# DenseNet 1-D
# ---------------------------------------------------------------------------

class _DenseLayer1d(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3, dropout_prob=0.0):
        super().__init__()
        padding = kernel_size // 2
        self.bn = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv1d(in_channels, growth_rate, kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        out = self.dropout(self.conv(F.relu(self.bn(x))))
        return torch.cat([x, out], dim=1)


class _DenseBlock1d(nn.Module):
    def __init__(self, in_channels, n_layers, growth_rate, kernel_size=3, dropout_prob=0.0):
        super().__init__()
        layers = []
        c = in_channels
        for _ in range(n_layers):
            layers.append(_DenseLayer1d(c, growth_rate, kernel_size, dropout_prob))
            c += growth_rate
        self.layers = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x):
        return self.layers(x)


class _Transition1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x))))


class DenseNet1dModel(nn.Module):
    def __init__(
        self,
        input_size=36,
        n_features=1,
        target_dim=1,
        hidden_channels=32,
        dropout_prob=0.3,
        use_layernorm=True,
        growth_rate=None,       # defaults to hidden_channels // 2
        n_blocks=3,
        layers_per_block=4,
    ):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.input_norm = nn.LayerNorm(input_size)

        if growth_rate is None:
            growth_rate = max(8, hidden_channels // 2)

        self.stem = nn.Sequential(
            nn.Conv1d(input_size, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )

        blocks = []
        c = hidden_channels
        for i in range(n_blocks):
            block = _DenseBlock1d(c, layers_per_block, growth_rate, dropout_prob=dropout_prob)
            blocks.append(block)
            c = block.out_channels
            if i < n_blocks - 1:
                out_c = c // 2
                blocks.append(_Transition1d(c, out_c))
                c = out_c
        self.blocks = nn.Sequential(*blocks)

        self.final_bn = nn.BatchNorm1d(c)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(c + n_features, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, target_dim)

    def forward(self, x, features, return_attn=False):
        if self.use_layernorm:
            x = self.input_norm(x)
        x = x.transpose(1, 2)          # [B, C, L]
        x = self.stem(x)
        x = self.blocks(x)
        x = F.relu(self.final_bn(x))
        x = self.pool(x).squeeze(-1)   # [B, c]
        x = torch.cat([x, features], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        if return_attn:
            return x, {}
        return x


# ---------------------------------------------------------------------------
# VGG 1-D
# ---------------------------------------------------------------------------

# n_convs per block: 2 for the first two blocks, 3 for deeper ones (VGG convention)
_VGG_CONVS_SCHEDULE = [2, 2, 3, 3, 3]


def _vgg_block(in_channels, out_channels, n_convs, kernel_size, dropout_prob):
    layers = []
    c = in_channels
    for _ in range(n_convs):
        layers += [
            nn.Conv1d(c, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        ]
        c = out_channels
    layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
    if dropout_prob > 0:
        layers.append(nn.Dropout(dropout_prob))
    return nn.Sequential(*layers)


class VGG1dModel(nn.Module):
    def __init__(
        self,
        input_size=36,
        n_features=1,
        target_dim=1,
        hidden_channels=32,
        dropout_prob=0.3,
        use_layernorm=True,
        n_blocks=3,
    ):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.input_norm = nn.LayerNorm(input_size)

        blocks = []
        in_ch = input_size
        out_ch = hidden_channels
        for i in range(n_blocks):
            n_convs = _VGG_CONVS_SCHEDULE[min(i, len(_VGG_CONVS_SCHEDULE) - 1)]
            blocks.append(_vgg_block(in_ch, out_ch, n_convs, kernel_size=3, dropout_prob=dropout_prob))
            in_ch = out_ch
            out_ch = out_ch * 2
        self.vgg_blocks = nn.Sequential(*blocks)
        final_ch = in_ch  # channels after the last block

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(final_ch + n_features, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, target_dim)

    def forward(self, x, features, return_attn=False):
        if self.use_layernorm:
            x = self.input_norm(x)
        x = x.transpose(1, 2)              # [B, C, L]
        x = self.vgg_blocks(x)
        x = self.pool(x).squeeze(-1)       # [B, final_ch]
        x = torch.cat([x, features], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        if return_attn:
            return x, {}
        return x
