import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class ChannelAttention(nn.Module):
    def __init__(self, n_features, reduction=4, dropout_prob=0.0, l1_reg_strength=0.0):
        super().__init__()
        hidden_dim = max(1, n_features // reduction)
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_features)
        self.dropout_prob = dropout_prob
        self.l1_reg_strength = l1_reg_strength

    def forward(self, x):
        # x: [batch, seq_len, n_features]
        pooled = x.mean(dim=1)  # [batch, n_features]
        weights = torch.relu(self.fc1(pooled))
        weights = torch.sigmoid(self.fc2(weights))  # [batch, n_features]

        # Dropout on attention weights
        if self.dropout_prob > 0:
            weights = F.dropout(weights, p=self.dropout_prob, training=self.training)

        # Apply weights
        out = x * weights.unsqueeze(1)

        # Optional L1 regularization (return term for loss)
        l1_penalty = self.l1_reg_strength * weights.abs().mean() if self.l1_reg_strength > 0 else 0.0
        return out, weights, l1_penalty

class ChannelSelfAttention(nn.Module):
    def __init__(
        self,
        n_channels,
        attn_dim=None,
        dropout_prob=0.0,
        use_residual=True,
        use_layernorm=True,
        attention_bias_matrix=None,
        bias_strength=1.0,
    ):
        super().__init__()
        self.C = n_channels
        self.d = attn_dim if attn_dim is not None else max(8, n_channels // 2)
        self.bias_strength = bias_strength

        self.q_proj = nn.Linear(1, self.d, bias=False)
        self.k_proj = nn.Linear(1, self.d, bias=False)
        self.v_proj = nn.Linear(1, self.d, bias=False)

        if attention_bias_matrix is not None:
            attention_bias_matrix = torch.tensor(attention_bias_matrix, dtype=torch.float32)
            self.register_buffer("pair_bias", attention_bias_matrix)
        else:
            self.pair_bias = None

        self.dropout = nn.Dropout(dropout_prob)
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln = nn.LayerNorm(n_channels)

    def forward(self, x):
        B, T, C = x.shape
        p = x.mean(dim=1)             # [B, C]
        p = p.unsqueeze(-1)           # [B, C, 1]

        Q = self.q_proj(p)
        K = self.k_proj(p)
        V = self.v_proj(p)

        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.d)

        # 🔑 inject pair bias
        if self.pair_bias is not None:
            scores = scores + self.bias_strength * self.pair_bias

        A = torch.softmax(scores, dim=-1)
        A = self.dropout(A)

        x_mix = torch.matmul(x, A.transpose(1, 2))
        if self.use_residual:
            x_mix = x + x_mix
        if self.use_layernorm:
            x_mix = self.ln(x_mix)

        return x_mix, A

class BiGRUModel(nn.Module):
    def __init__(
        self,
        hidden_size,
        dropout_prob,
        num_layers,
        input_size=36,
        n_features=1,
        target_dim=1,
        bidirectional=True,
        use_layernorm=True,
        use_temporal_attention=True,
        use_channel_attention=False,
        use_channel_self_attention=False,
        channel_self_attn_dim=None,
        attention_bias_matrix=None,
        bias_strength=1.0,
    ):
        super(BiGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.bidirectional = bidirectional
        self.use_layernorm = use_layernorm
        self.use_temporal_attention = use_temporal_attention
        self.use_channel_attention = use_channel_attention
        self.use_channel_self_attention = use_channel_self_attention
        self.n_features = n_features
        self.target_dim = target_dim

        if self.use_channel_attention and self.use_channel_self_attention:
            raise ValueError(
                "Choose either use_channel_attention (gating) OR use_channel_self_attention (CxC), not both.")

        self.gru_output_dim = hidden_size * (2 if bidirectional else 1)
        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)

        # Channel attention
        if self.use_channel_attention:
            #self.channel_attn = ChannelAttention(input_size, l1_reg_strength=1.0) # used to be fixed to 1
            self.channel_attn = ChannelAttention(input_size, l1_reg_strength=0.0)  # used to be fixed to 1

        if self.use_channel_self_attention:
            self.channel_self_attn = ChannelSelfAttention(
                n_channels=input_size,
                attn_dim=channel_self_attn_dim,  # None -> default
                dropout_prob=dropout_prob,
                use_residual=True,
                use_layernorm=True,
                attention_bias_matrix=attention_bias_matrix,
                bias_strength=bias_strength
            )

        # BiGRU encoder
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0,
        )

        # Temporal attention projections
        if self.use_temporal_attention:
            self.q_proj = nn.Linear(self.gru_output_dim, self.gru_output_dim)
            self.k_proj = nn.Linear(self.gru_output_dim, self.gru_output_dim)
            self.v_proj = nn.Linear(self.gru_output_dim, self.gru_output_dim)

        # Optional LayerNorm after GRU
        if use_layernorm:
            self.layer_norm = nn.LayerNorm(self.gru_output_dim)

        # Classifier
        self.fc1 = nn.Linear(self.gru_output_dim + self.n_features, hidden_size // 2)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size // 2, self.target_dim)

    def forward(self, x, features, return_attn=False):
        """
        Forward pass with optional attention outputs.
        x: [batch, seq_len, input_size]
        features: [batch, n_features]
        return_attn: if True, return attention weights
        """
        attn_outputs = {}

        # Input norm
        x = self.input_norm(x)

        # Channel attention (if enabled)
        if self.use_channel_attention:
            x, channel_weights, l1_penalty = self.channel_attn(x)
            attn_outputs["channel"] = channel_weights
            attn_outputs["channel_l1"] = l1_penalty

        if self.use_channel_self_attention:
            x, channel_A = self.channel_self_attn(x)
            attn_outputs["channel_matrix"] = channel_A  # [B, C, C]

        # GRU encoder
        gru_out, _ = self.gru(x)  # [batch, seq_len, 2*hidden_size]

        if self.use_layernorm:
            gru_out = self.layer_norm(gru_out)

        # Temporal attention (if enabled)
        if self.use_temporal_attention:
            Q = self.q_proj(gru_out)
            K = self.k_proj(gru_out)
            V = self.v_proj(gru_out)

            attn_scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(Q.size(-1))
            attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, seq_len, seq_len]

            context = torch.bmm(attn_weights, V).mean(dim=1)
            attn_outputs["temporal"] = attn_weights
        else:
            context = gru_out[:, -1, :]

        # Concatenate with static features
        context_with_features = torch.cat([context, features], axis=1)

        # Feed-forward layers
        x = torch.relu(self.fc1(context_with_features))
        x = self.dropout(x)
        x = self.fc2(x)

        return (x, attn_outputs) if return_attn else x
