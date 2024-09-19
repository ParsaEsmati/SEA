import torch
import torch.nn as nn
import torch.nn.functional as FF

from models.base_blocks import PositionalEncoding, PositionalEncodingLearnable, LayerNorm, MLP, GaussianFourierProjection, MultiHeadAttention, MultiHeadCrossAttention

class BaseBlockTemporal(nn.Module):
    def __init__(self, 
                 n_heads, 
                 max_len, 
                 embed_dim, 
                 src_len, 
                 scale_ratio, 
                 num_variables, 
                 down_proj=2,
                 ib_mode='fourier', 
                 ib_addition_mode='add', 
                 ib_mlp_layers=1, 
                 dropout=0, 
                 add_info_after_cross=False):
        
        super().__init__()
        self.num_variables = num_variables
        self.ib_dim_concat = 64  # config: 64
        self.original_embed_dim = embed_dim
        self.ib_mlp_layers = ib_mlp_layers
        self.ib_addition_mode = self._validate_ib_addition_mode(ib_addition_mode)
        self.add_info_after_cross = add_info_after_cross

        self.internal_embed_dim = embed_dim + self.ib_dim_concat if self.ib_addition_mode == 'concat' else embed_dim

        if self.ib_addition_mode == 'attention':
            self.cross_attn = nn.ModuleList([
                MultiHeadCrossAttention(n_heads, self.internal_embed_dim, max_len, src_len, dropout)
                for _ in range(num_variables)
            ])

        self.ib_mode = self._validate_ib_mode(ib_mode)
        self.ib_dim = self._get_ib_dim(self.ib_addition_mode, embed_dim)
        self.ib = self._create_ib_layer(self.ib_mode, self.ib_dim, dropout, scale_ratio)

        self.down_ratio = down_proj
        self.down_dim = self.internal_embed_dim // self.down_ratio

        self.ln = nn.ModuleDict({
            'exp': nn.ModuleList([nn.ModuleList([LayerNorm(self.internal_embed_dim, bias=False) for _ in range(3)]) for _ in range(num_variables)]),
            'cross': LayerNorm(self.down_dim, bias=False)
        })

        self.attn = nn.ModuleDict({
            'self': nn.ModuleList([MultiHeadAttention(n_heads, self.internal_embed_dim, max_len, dropout) for _ in range(num_variables)])
        })

        self.mlp = nn.ModuleList([MLP(self.internal_embed_dim, dropout, scale_ratio) for _ in range(num_variables)])

        self.act = nn.GELU()
        self.pos_encoder = PositionalEncoding(self.down_dim, dropout)

        self.proj = nn.ModuleList([nn.Linear(self.internal_embed_dim, self.original_embed_dim) for _ in range(num_variables)])

    def _validate_ib_mode(self, mode):
        valid_modes = {'fourier', 'linear', 'mlp'}
        mode = mode.lower()
        if mode not in valid_modes:
            raise ValueError(f"Invalid ib_mode '{mode}'. Must be one of {valid_modes}.")
        return mode

    def _validate_ib_addition_mode(self, mode):
        valid_modes = {'add', 'concat', 'attention', 'none'}
        mode = mode.lower()
        if mode not in valid_modes:
            raise ValueError(f"Invalid ib_addition_mode '{mode}'. Must be one of {valid_modes}.")
        return mode

    def _get_ib_dim(self, addition_mode, embed_dim):
        return self.ib_dim_concat if addition_mode == 'concat' else embed_dim

    def _create_ib_layer(self, mode, dim, dropout, scale_ratio):
        if mode == 'fourier':
            return GaussianFourierProjection(9, int(dim//2)) # TODO config: 9
        elif mode == 'linear':
            return nn.Linear(9, dim) # TODO config: 9
        elif mode == 'mlp':
            return MLP(9, dropout, scale_ratio, dim, self.ib_mlp_layers) # TODO config: 9

    def _add_info(self, x, add_info, var_idx):
        ib_output = self.ib(add_info)
        if self.ib_addition_mode == 'none':
            return x
        elif self.ib_addition_mode == 'add':
            return x + ib_output
        elif self.ib_addition_mode == 'concat':
            return torch.cat([x, ib_output], dim=-1)
        elif self.ib_addition_mode == 'attention':
            return x + self.cross_attn[var_idx](x, ib_output)

    def _apply_exchange(self, x_vars):
        # This method will be overridden by subclasses
        raise NotImplementedError

    def forward(self, *x_vars, x_add):
        assert len(x_vars) == self.num_variables, f"Expected {self.num_variables} input variables, but got {len(x_vars)}"

        x_vars = list(x_vars)
        
        if not self.add_info_after_cross:
            for i in range(self.num_variables):
                x_vars[i] = self._add_info(x_vars[i], x_add, i)

        for i in range(self.num_variables):
            x_vars[i] = x_vars[i] + self.attn['self'][i](self.ln['exp'][i][0](x_vars[i]))

        x_vars = self._apply_exchange(x_vars)

        if self.add_info_after_cross: # TODO Important to set to true for CF and MP
            for i in range(self.num_variables):
                x_vars[i] = self._add_info(x_vars[i], x_add, i)

        for i in range(self.num_variables):
            x_vars[i] = x_vars[i] + self.mlp[i](self.ln['exp'][i][2](x_vars[i]))
            x_vars[i] = self.proj[i](x_vars[i])

        return tuple(x_vars)

class SEABlockTemporal(BaseBlockTemporal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cross_down = nn.ModuleList([nn.Linear(self.internal_embed_dim, self.down_dim) for _ in range(self.num_variables)])
        self.cross_up = nn.ModuleList([nn.Linear(self.down_dim, self.internal_embed_dim) for _ in range(self.num_variables)])
        self.attn['cross'] = MultiHeadCrossAttention(self.attn['self'][0].n_heads, self.down_dim, self.attn['self'][0].max_len, self.attn['self'][0].max_len, self.attn['self'][0].dropout)

    def _apply_cross_attention(self, x, others, var_idx):
        x_down = self.cross_down[var_idx](x)
        x_norm = self.ln['cross'](x_down)
        x_pos = self.pos_encoder(x_norm)

        attn_out = sum([self.attn['cross'](x_pos, self.pos_encoder(self.ln['cross'](self.cross_down[i](other))))
                        for i, other in enumerate(others) if i != var_idx])

        return self.cross_up[var_idx](self.act(attn_out))

    def _apply_exchange(self, x_vars):
        for i in range(self.num_variables):
            others = x_vars[:i] + x_vars[i+1:]
            x_vars[i] = x_vars[i] + self._apply_cross_attention(x_vars[i], others, i)
        return x_vars

# todo: check if this is correct since we also add itself 
class AddBlockTemporal(BaseBlockTemporal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var_projection = nn.ModuleList([nn.Linear(self.internal_embed_dim, self.internal_embed_dim) 
                                             for _ in range(self.num_variables)])
        
    def _apply_exchange(self, x_vars):
        projected_vars = [self.var_projection[i](x) for i, x in enumerate(x_vars)]

        results = []
        for i, x in enumerate(x_vars):
            other_vars_sum = sum(proj for j, proj in enumerate(projected_vars) if j != i)
            results.append(x + other_vars_sum)
        
        return results

class SimpleBlockTemporal(BaseBlockTemporal):
    def _apply_exchange(self, x_vars):
        return x_vars  # No exchange for simple mode

def create_block_temporal(exchange_mode, *args, **kwargs):
    if exchange_mode == 'sea':
        return SEABlockTemporal(*args, **kwargs)
    elif exchange_mode == 'addition':
        return AddBlockTemporal(*args, **kwargs)
    elif exchange_mode == 'simple':
        return SimpleBlockTemporal(*args, **kwargs)
    else:
        raise ValueError(f"Invalid exchange_mode: {exchange_mode}")

class TemporalModel(nn.Module):
    def __init__(self, 
                 num_layers, 
                 embed_dim, 
                 n_heads, 
                 max_len, 
                 scale_ratio, 
                 src_len,
                 num_variables, 
                 down_proj=2, 
                 dropout=0.0, 
                 exchange_mode='sea', 
                 pos_encoding_mode='learnable',
                 ib_mode='fourier', 
                 ib_addition_mode='add', 
                 ib_mlp_layers=1):
        
        super().__init__()
        self.num_variables = num_variables
        self.exchange_mode = self._validate_exchange_mode(exchange_mode)
        self.pos_encoding_mode = self._validate_pos_encoding_mode(pos_encoding_mode)
        self.ib_mode = ib_mode
        self.ib_addition_mode = ib_addition_mode

        self.blocks = nn.ModuleList([
            create_block_temporal(
                self.exchange_mode, n_heads=n_heads, max_len=max_len, embed_dim=embed_dim,
                src_len=src_len, down_proj=down_proj, scale_ratio=scale_ratio,
                dropout=dropout, ib_mode=self.ib_mode, ib_addition_mode=self.ib_addition_mode,
                ib_mlp_layers=ib_mlp_layers, num_variables=num_variables
            )
            for _ in range(num_layers)
        ])

        self.ln = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_variables)])
        self.pos_encoder = self._create_pos_encoder(pos_encoding_mode, embed_dim, max_len, dropout)
        self.apply(self._init_weights)

    def _validate_exchange_mode(self, mode):
        valid_modes = {'sea', 'simple', 'addition'}
        mode = mode.lower()
        if mode not in valid_modes:
            raise ValueError(f"Invalid exchange_mode '{mode}'. Must be one of {valid_modes}.")
        return mode

    def _validate_pos_encoding_mode(self, mode):
        valid_modes = {'learnable', 'fixed'}
        if mode not in valid_modes:
            raise ValueError(f"Invalid pos_encoding_mode '{mode}'. Must be one of {valid_modes}.")
        return mode
    
    def _create_pos_encoder(self, mode, embed_dim, max_len, dropout):
        if mode == 'learnable':
            return PositionalEncodingLearnable(embed_dim, max_len=5000, dropout=dropout)
        elif mode == 'fixed':
            return PositionalEncoding(embed_dim, dropout)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x, x_additional_info):
        # x shape: [batch, time, field, cell]
        assert x.shape[2] == self.num_variables, f"Expected {self.num_variables} variables, but got {x.shape[2]}"
        x_vars = [x[:, :, i, :] for i in range(self.num_variables)]
        x_vars = [self.pos_encoder(var) for var in x_vars]

        for block in self.blocks:
            x_vars = block(*x_vars, x_add=x_additional_info)

        x_vars = [self.ln[i](var) for i, var in enumerate(x_vars)]

        x = torch.stack(x_vars, dim=2)
        return x