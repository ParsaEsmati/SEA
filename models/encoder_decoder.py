import torch
import torch.nn as nn
from models.base_blocks import EncoderBlock, PositionalEncoding, downScaleMLP, upScaleMLP

class Encode(nn.Module):
    def __init__(self, field_groups, n_inp, MLP_hidden, num_layers, embed_dim, n_heads, max_len, src_len, dropout=0.1):
        super().__init__()
        self.field_groups = field_groups
        self.num_groups = len(field_groups)

        self.encoders_mu = nn.ModuleList([
            downScaleMLP(d_input=n_inp * len(group), d_model=embed_dim, hidden_dim=MLP_hidden)
            for group in field_groups
        ])
        self.encoders_logvar = nn.ModuleList([
            downScaleMLP(d_input=n_inp * len(group), d_model=embed_dim, hidden_dim=MLP_hidden)
            for group in field_groups
        ])

        self.spatial_pos_encoder = PositionalEncoding(self.num_groups * embed_dim, dropout)

        self.blocks = nn.ModuleList([EncoderBlock(n_heads=n_heads,
                                                  max_len=max_len,
                                                  embed_dim=self.num_groups * embed_dim,
                                                  src_len=src_len,
                                                  dropout=dropout) for _ in range(num_layers)])

        self.ln = nn.LayerNorm(self.num_groups * embed_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        B, P, F, C = x.shape

        mus, logvars, zs = [], [], []
        for i, group in enumerate(self.field_groups):
            x_group = x[:, :, group, :].permute(0, 1, 3, 2).reshape(B, P, 1, -1)
            mu = self.encoders_mu[i](x_group)
            logvar = self.encoders_logvar[i](x_group)
            z = self.reparameterize(mu, logvar)
            mus.append(mu)
            logvars.append(logvar)
            zs.append(z)

        mu = torch.cat(mus, dim=-2)
        logvar = torch.cat(logvars, dim=-2)
        z = torch.cat(zs, dim=-2)

        z = z.reshape(B, P, -1)
        z = self.spatial_pos_encoder(z)

        for block in self.blocks:
            z = block(z)
        z = self.ln(z)

        z = z.reshape(B, P, self.num_groups, -1)

        return z, mu, logvar

class Decode(nn.Module):
    def __init__(self, field_groups, n_inp, MLP_hidden, num_layers, embed_dim, n_heads, max_len, src_len, dropout=0.1):
        super().__init__()
        self.field_groups = field_groups
        self.num_groups = len(field_groups)

        self.ln = nn.LayerNorm(self.num_groups * embed_dim)

        self.blocks = nn.ModuleList([EncoderBlock(n_heads=n_heads,
                                                  max_len=max_len,
                                                  embed_dim=self.num_groups * embed_dim,
                                                  src_len=src_len,
                                                  dropout=dropout) for _ in range(num_layers)])

        self.decoders = nn.ModuleList([
            upScaleMLP(d_model=embed_dim, d_output=n_inp * len(group), hidden_dim=MLP_hidden)
            for group in field_groups
        ])
        self.spatial_pos_encoder = PositionalEncoding(self.num_groups * embed_dim, dropout)

    def forward(self, z):
        B, P, EF, D = z.shape
        z = z.reshape(B, P, -1)
        z = self.spatial_pos_encoder(z)

        for block in self.blocks:
            z = block(z)
        z = self.ln(z)

        z = z.reshape(B, P, self.num_groups, -1)

        outputs = []
        for i, group in enumerate(self.field_groups):
            z_group = z[:, :, i:i+1, :]
            x_group = self.decoders[i](z_group).reshape(B, P, -1, len(group)).permute(0, 1, 3, 2)
            outputs.append(x_group)

        output = torch.cat(outputs, dim=2)
        return output

class SpatialModel(nn.Module):
    def __init__(self, field_groups, n_inp, MLP_hidden, num_layers, embed_dim, n_heads, max_len, src_len, dropout=0.1):
        super().__init__()

        self.encode = Encode(field_groups, n_inp, MLP_hidden, num_layers, embed_dim, n_heads, max_len, src_len, dropout)
        self.decode = Decode(field_groups, n_inp, MLP_hidden, num_layers, embed_dim, n_heads, max_len, src_len, dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x):  # [B,P,F,C]
        x = self.generate_padding_mask(x)

        z, mu, logvar = self.encode(x)
        output = self.decode(z)

        return output, mu, logvar

    def generate_padding_mask(self, x, pad_idx=-9999):  # -9999 is the pad id set in data processing
        src_padding_mask = (x == pad_idx)
        x[src_padding_mask] = 0.0
        return x