import torch
import torch.nn as nn
from models.base_blocks import EncoderBlock, PositionalEncoding, downScaleMLP, upScaleMLP

class Encode(nn.Module):
    def __init__(self, field_groups, n_inp, MLP_hidden, num_layers, embed_dim, n_heads, max_len, src_len, dropout=0.1):
        super().__init__()
        self.field_groups = field_groups
        self.num_groups = len(field_groups)

        self.spatial_pos_encoder = PositionalEncoding(self.num_groups * embed_dim, dropout)

        self.blocks = nn.ModuleList([EncoderBlock(n_heads=n_heads,
                                                  max_len=max_len,
                                                  embed_dim=self.num_groups * embed_dim,
                                                  src_len=src_len,
                                                  dropout=dropout) for _ in range(num_layers)])

        self.ln = nn.LayerNorm(self.num_groups * embed_dim)

        #self.apply(self._init_weights)

        self.encoders_mu = nn.ModuleList([
            downScaleMLP(d_input=n_inp * len(group), d_model=embed_dim, hidden_dim=MLP_hidden)
            for group in field_groups
        ])
        self.encoders_logvar = nn.ModuleList([
            downScaleMLP(d_input=n_inp * len(group), d_model=embed_dim, hidden_dim=MLP_hidden)
            for group in field_groups
        ])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        B, P, F, C = x.shape

        mus, logvars, zs = [], [], []
        for i, group in enumerate(self.field_groups):
            x_group = x[:, :, group, :].reshape(B, P, 1, -1)
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


class PointwiseEncode(nn.Module):
    def __init__(self, field_groups, n_inp, MLP_hidden, num_layers, embed_dim, n_heads, max_len, src_len, dropout=0.1):
        super().__init__()
        self.field_groups = field_groups
        self.num_groups = len(field_groups)

        self.spatial_pos_encoder = PositionalEncoding(self.num_groups * embed_dim, dropout)
        self.blocks = nn.ModuleList([EncoderBlock(n_heads=n_heads,
                                                  max_len=max_len,
                                                  embed_dim=self.num_groups * embed_dim,
                                                  src_len=src_len,
                                                  dropout=dropout) for _ in range(num_layers)])

        self.ln = nn.LayerNorm(self.num_groups * embed_dim)
        self.apply(self._init_weights)

        self.encoders = nn.ModuleList([
            downScaleMLP(d_input=n_inp * len(group), d_model=embed_dim, hidden_dim=MLP_hidden)
            for group in field_groups
        ])

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        B, P, F, C = x.shape
        zs = []
        for i, group in enumerate(self.field_groups):
            x_group = x[:, :, group, :].reshape(B, P, 1, -1)
            z = self.encoders[i](x_group)
            zs.append(z)

        z = torch.cat(zs, dim=-2)
        z = z.reshape(B, P, -1)
        z = self.spatial_pos_encoder(z)

        for block in self.blocks:
            z = block(z)
        z = self.ln(z)

        z = z.reshape(B, P, self.num_groups, -1)

        return z


class Decode(nn.Module):
    def __init__(self, field_groups, n_inp, MLP_hidden, embed_dim, dropout=0.1):
        super().__init__()
        self.field_groups = field_groups
        self.num_groups = len(field_groups)

        self.decoders = nn.ModuleList([
            upScaleMLP(d_model=embed_dim, d_output=n_inp * len(group), hidden_dim=MLP_hidden)
            for group in field_groups
        ])

    def forward(self, z):
        outputs = []
        B, P, _, D = z.shape
        for i, group in enumerate(self.field_groups):
            z_group = z[:, :, i:i+1, :]
            x_group = self.decoders[i](z_group).reshape(B, P, len(group), -1)
            outputs.append(x_group)

        output = torch.cat(outputs, dim=2)
        return output


class SpatialModel(nn.Module):
    def __init__(self, field_groups, n_inp, MLP_hidden, num_layers, embed_dim, n_heads, max_len, src_len, dropout=0.1, variational=False):
        super().__init__()

        self.variational = variational
        if self.variational:
            self.encode = Encode(field_groups, n_inp, MLP_hidden, num_layers, embed_dim, n_heads, max_len, src_len, dropout)
        else:
            self.encode = PointwiseEncode(field_groups, n_inp, MLP_hidden, num_layers, embed_dim, n_heads, max_len, src_len, dropout)

        self.decode = Decode(field_groups, n_inp, MLP_hidden, embed_dim, dropout)

    def forward(self, x):  # [B,P,F,C]
        x = self.generate_padding_mask(x)

        if self.variational:
            z, mu, logvar = self.encode(x)
            output = self.decode(z)
            return output, mu, logvar
        else:
            z = self.encode(x)
            output = self.decode(z)
            return output

    def generate_padding_mask(self, x, pad_idx=-9999):
        src_padding_mask = (x == pad_idx)
        x[src_padding_mask] = 0.0
        return x
 