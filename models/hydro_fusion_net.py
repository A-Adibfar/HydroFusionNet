import torch
import torch.nn as nn


class FeatureBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(FeatureBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class DiffBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(DiffBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(AttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output.mean(dim=1)


class MonthEmbedding(nn.Module):
    def __init__(self):
        super(MonthEmbedding, self).__init__()

    def forward(self, month):
        sin_month = torch.sin(2 * torch.pi * month / 12)
        cos_month = torch.cos(2 * torch.pi * month / 12)
        month_emb = torch.stack([sin_month, cos_month], dim=-1)
        return month_emb


class HydroFusionNet(nn.Module):
    def __init__(self):
        super(HydroFusionNet, self).__init__()

        self.feature1 = FeatureBlock(input_dim=15)
        self.feature2 = FeatureBlock(input_dim=15)
        self.feature3 = FeatureBlock(input_dim=15)
        self.feature4 = FeatureBlock(input_dim=15)

        self.diff_block = DiffBlock(input_dim=6)
        self.attention_fusion = AttentionFusion(embed_dim=64)
        self.month_embed = MonthEmbedding()

        self.classifier = nn.Sequential(
            nn.Linear(64 + 32 + 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, feature1, feature2, feature3, feature4, diff_features, month):
        f1 = self.feature1(feature1)
        f2 = self.feature2(feature2)
        f3 = self.feature3(feature3)
        f4 = self.feature4(feature4)

        stack = torch.stack([f1, f2, f3, f4], dim=1)
        fused = self.attention_fusion(stack)

        diff_emb = self.diff_block(diff_features)
        month_emb = self.month_embed(month)

        combined = torch.cat([fused, diff_emb, month_emb], dim=-1)
        out = self.classifier(combined)
        return out.squeeze(1)
