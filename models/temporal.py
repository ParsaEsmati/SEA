import torch
import torch.nn as nn
import torch.nn.functional as FF

from models.base_blocks import (
    PositionalEncoding,
    PositionalEncodingLearnable,
    
    LayerNorm,
    AdaLN,

    MLP,
    GaussianFourierProjection,
  
    MultiHeadAttention,
    MultiHeadCrossAttention,
    MaskedMultiHeadAttention,
    MaskedMultiHeadCrossAttention,
)

class BaseBlockTemporal(nn.Module):
    def __init__(self, 
                 n_heads, 
                 max_len, 
                 embed_dim, 
                 src_len, 
                 scale_ratio, 
                 num_variables, 
                 down_proj=2,
                 ib_scale_mode='fourier', 
                 ib_addition_mode='add', 
                 ib_mlp_layers=1,
                 ib_num = 1,
                 dropout=0, 
                 add_info_after_cross=False,
                 LN_type='adaln'):
        
        super().__init__()
        self.num_variables = num_variables
        self.ib_dim_concat = 64  # config: 64
        self.original_embed_dim = embed_dim
        self.ib_mlp_layers = ib_mlp_layers
        self.ib_addition_mode = self._validate_ib_addition_mode(ib_addition_mode)
        self.ib_num = ib_num
        self.add_info_after_cross = add_info_after_cross

        self.internal_embed_dim = embed_dim + self.ib_dim_concat if self.ib_addition_mode == 'concat' else embed_dim

        if self.ib_addition_mode == 'attention':
            self.cross_attn_ib = nn.ModuleList([
                MultiHeadCrossAttention(n_heads, self.internal_embed_dim, max_len, src_len, dropout)
                for _ in range(num_variables)
            ])

        self.ib_scale_mode = self._validate_ib_mode(ib_scale_mode)
        self.ib_dim = self._get_ib_dim(self.ib_addition_mode, embed_dim)
        self.ib = self._create_ib_layer(self.ib_scale_mode, self.ib_dim, dropout, scale_ratio)

        self.down_ratio = down_proj
        self.down_dim = self.internal_embed_dim // self.down_ratio

        if LN_type.lower() == 'adaln':
            self.ln = nn.ModuleDict({
                'exp': nn.ModuleList([nn.ModuleList([AdaLN(self.internal_embed_dim, ib_num) for _ in range(3)]) for _ in range(num_variables)]),
                'cross': AdaLN(self.down_dim, ib_num)
            })
        elif LN_type.lower() == 'ln':
            self.ln = nn.ModuleDict({
                'exp': nn.ModuleList([nn.ModuleList([LayerNorm(self.internal_embed_dim, bias=False) for _ in range(3)]) for _ in range(num_variables)]),
                'cross': LayerNorm(self.down_dim, bias=False)
            })
        else:
            raise ValueError(f"Invalid LN_type: {LN_type}. Must be one of {'adaln', 'ln'}.")

        self.attn = nn.ModuleDict({
            'self': nn.ModuleList([MaskedMultiHeadAttention(n_heads, self.internal_embed_dim, max_len, src_len, dropout) for _ in range(num_variables)])
        })

        self.mlp = nn.ModuleList([MLP(self.internal_embed_dim, dropout, scale_ratio) for _ in range(num_variables)])

        self.act = nn.GELU()
        self.pos_encoder = PositionalEncoding(self.down_dim, dropout) #TODO: removed since we have RoPE implemented

        self.proj = nn.ModuleList([nn.Linear(self.internal_embed_dim, self.original_embed_dim) for _ in range(num_variables)])

    def _validate_ib_mode(self, mode):
        valid_modes = {'fourier', 'linear', 'mlp'}
        mode = mode.lower()
        if mode not in valid_modes:
            raise ValueError(f"Invalid ib_scale_mode '{mode}'. Must be one of {valid_modes}.")
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
            return GaussianFourierProjection(self.ib_num, int(dim//2))
        elif mode == 'linear':
            return nn.Linear(self.ib_num, dim)
        elif mode == 'mlp':
            return MLP(self.ib_num, dropout, scale_ratio, dim, self.ib_mlp_layers)

    def _add_info(self, x, add_info, var_idx):
        ib_output = self.ib(add_info)
        if self.ib_addition_mode == 'none':
            return x
        elif self.ib_addition_mode == 'add':
            return x + ib_output
        elif self.ib_addition_mode == 'concat':
            return torch.cat([x, ib_output], dim=-1)
        elif self.ib_addition_mode == 'attention':
            return x + self.cross_attn_ib[var_idx](x, ib_output)

    def _apply_exchange(self, x_vars, x_add):
        # This method will be overridden by subclasses
        raise NotImplementedError

    def forward(self, *x_vars, x_add):
        assert len(x_vars) == self.num_variables, f"Expected {self.num_variables} input variables, but got {len(x_vars)}"

        x_vars = list(x_vars)
        
        if not self.add_info_after_cross:
            for i in range(self.num_variables):
                x_vars[i] = self._add_info(x_vars[i], x_add, i)

        for i in range(self.num_variables):
            x_vars[i] = x_vars[i] + self.attn['self'][i](self.ln['exp'][i][0](x_vars[i], x_add))

        x_vars = self._apply_exchange(x_vars, x_add)

        if self.add_info_after_cross:
            for i in range(self.num_variables):
                x_vars[i] = self._add_info(x_vars[i], x_add, i)

        for i in range(self.num_variables):
            x_vars[i] = x_vars[i] + self.mlp[i](self.ln['exp'][i][2](x_vars[i], x_add))
            x_vars[i] = self.proj[i](x_vars[i])

        return tuple(x_vars)
"""
Original SEA implementation
"""
class SEABlockTemporal(BaseBlockTemporal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_heads = kwargs['n_heads']
        self.max_len = kwargs['max_len']
        self.src_len = kwargs['src_len']
        self.dropout = kwargs['dropout']
        self.LN_type = kwargs['LN_type']
        self.cross_down = nn.ModuleList([nn.Linear(self.internal_embed_dim, self.down_dim) for _ in range(self.num_variables)])
        self.cross_up = nn.ModuleList([nn.Linear(self.down_dim, self.internal_embed_dim) for _ in range(self.num_variables)])
        self.cross_attn = nn.ModuleList([
            nn.ModuleList([
                MaskedMultiHeadCrossAttention(self.n_heads, self.down_dim, self.max_len, self.src_len, self.dropout)
                for _ in range(self.num_variables)
            ])
            for _ in range(self.num_variables)
        ])
        if self.LN_type.lower() == 'adaln':
            self.ln_cross = nn.ModuleList([AdaLN(self.down_dim, self.ib_num) for _ in range(self.num_variables)])
        elif self.LN_type.lower() == 'ln':
            self.ln_cross = nn.ModuleList([LayerNorm(self.down_dim) for _ in range(self.num_variables)])
        else:
            raise ValueError(f"Invalid LN_type: {self.LN_type}. Must be one of {'adaln', 'ln'}.")

    def _apply_cross_attention(self, x_i, x_j, i, j, x_add):
        x_i_down = self.cross_down[i](x_i)
        x_j_down = self.cross_down[j](x_j)
        
        x_i_norm = self.ln_cross[i](x_i_down, x_add)
        x_j_norm = self.ln_cross[j](x_j_down, x_add)

        attn_out = self.cross_attn[i][j](x_i_norm, x_j_norm)
        
        return self.cross_up[i](self.act(attn_out))

    def _apply_exchange(self, x_vars, x_add):
        for i, x_i in enumerate(x_vars):
            cross_attn_sum = sum(self._apply_cross_attention(x_i, x_j, i, j, x_add)
                                for j, x_j in enumerate(x_vars) if i != j)
            x_vars[i] = x_i + cross_attn_sum
        return x_vars

"""
Gated SEA implementation with information pool
"""
class SEAPoolBlockTemporal(BaseBlockTemporal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_heads = kwargs['n_heads']
        self.max_len = kwargs['max_len']
        self.src_len = kwargs['src_len']
        self.dropout = kwargs['dropout']
        self.LN_type = kwargs['LN_type']
        self.pool_update_method = kwargs.get('pool_update_method', 'mlp')

        # Pool token
        self.pool_token = nn.Parameter(torch.randn(1, 1, self.down_dim))

        # Projection layers
        self.cross_down = nn.ModuleList([nn.Linear(self.internal_embed_dim, self.down_dim) for _ in range(self.num_variables)])
        self.cross_up = nn.ModuleList([nn.Linear(self.down_dim, self.internal_embed_dim) for _ in range(self.num_variables)])

        # Cross-attention layers
        self.cross_attn = nn.ModuleList([
            MaskedMultiHeadCrossAttention(self.n_heads, self.down_dim, self.max_len, self.src_len, self.dropout)
            for _ in range(self.num_variables)
        ])

        # Normalization layers
        if self.LN_type.lower() == 'adaln':
            self.ln_cross = nn.ModuleList([AdaLN(self.down_dim, self.ib_num) for _ in range(self.num_variables)])
            self.ln_pool = AdaLN(self.down_dim, self.ib_num)
        elif self.LN_type.lower() == 'ln':
            self.ln_cross = nn.ModuleList([LayerNorm(self.down_dim) for _ in range(self.num_variables)])
            self.ln_pool = LayerNorm(self.down_dim)
        else:
            raise ValueError(f"Invalid LN_type: {self.LN_type}. Must be one of {'adaln', 'ln'}.")

        # Pool update mechanism
        if self.pool_update_method == 'linear':
            self.pool_update = nn.Linear(self.down_dim * self.num_variables, self.down_dim)
        elif self.pool_update_method == 'mlp':
            self.pool_update = nn.Sequential(
                nn.Linear(self.down_dim * self.num_variables, self.down_dim * 2),
                nn.GELU(),
                nn.Linear(self.down_dim * 2, self.down_dim)
            )
        elif self.pool_update_method == 'gru':
            self.pool_update = nn.GRU(self.down_dim, self.down_dim, batch_first=True)
        elif self.pool_update_method == 'pooling':
            self.pool_update = nn.Parameter(torch.ones(self.num_variables) / self.num_variables)

    def _update_pool_token(self, pool_token, normalized):
        if self.pool_update_method == 'pooling':
            return torch.sum(torch.stack(normalized, dim=1) * self.pool_update.view(1, -1, 1, 1), dim=1)
        elif self.pool_update_method in ['linear', 'mlp']:
            concatenated = torch.cat(normalized, dim=-1)
            return self.pool_update(concatenated)
        else:
            raise ValueError(f"Invalid pool_update_method: {self.pool_update_method}")

    def _apply_cross_attention(self, x_i_norm, pool_token, i):
        attn_out = self.cross_attn[i](x_i_norm, pool_token)
        return attn_out

    def _apply_exchange(self, x_vars, x_add):
        batch_size = x_vars[0].shape[0]

        # Project and normalize all variables
        down_projected = [self.cross_down[i](x) for i, x in enumerate(x_vars)]
        normalized = [self.ln_cross[i](x_down, x_add) for i, x_down in enumerate(down_projected)]

        # Apply positional encoding #TODO: removed since we have RoPE implemented
        normalized = [self.pos_encoder(x_norm) for x_norm in normalized]

        # Prepare Pool token
        pool_token = self.pool_token.expand(batch_size, -1, -1)
        pool_token = self.ln_pool(pool_token, x_add)
        pool_token = self.pos_encoder(pool_token)

        # Update Pool token
        pool_token = self._update_pool_token(pool_token, normalized)

        # Perform cross-attention with Pool token
        for i, x_i in enumerate(x_vars):
            cross_attn_out = self._apply_cross_attention(normalized[i], pool_token, i)

            # Skip connection
            combined = normalized[i] + cross_attn_out
            x_vars[i] = x_i + self.cross_up[i](self.act(combined))

        return x_vars
    
class AddBlockTemporal(BaseBlockTemporal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.LN_type = kwargs['LN_type']
        self.cross_down = nn.ModuleList([nn.Linear(self.internal_embed_dim, self.down_dim) for _ in range(self.num_variables)])
        self.cross_up = nn.ModuleList([nn.Linear(self.down_dim, self.internal_embed_dim) for _ in range(self.num_variables)])

        if self.LN_type.lower() == 'adaln':
            self.ln_cross = nn.ModuleList([AdaLN(self.down_dim, self.ib_num) for _ in range(self.num_variables)])
        elif self.LN_type.lower() == 'ln':
            self.ln_cross = nn.ModuleList([LayerNorm(self.down_dim) for _ in range(self.num_variables)])
        else:
            raise ValueError(f"Invalid LN_type: {self.LN_type}. Must be one of {'adaln', 'ln'}.")

    def _apply_exchange(self, x_vars, x_add):
        down_projected = [self.cross_down[i](x) for i, x in enumerate(x_vars)]
        normalized = [self.ln_cross[i](x_down, x_add) for i, x_down in enumerate(down_projected)]
        
        for i in range(len(x_vars)):
            other_vars_sum = sum(norm for j, norm in enumerate(normalized) if j != i)
            combined = normalized[i] + other_vars_sum
            x_vars[i] = x_vars[i] + self.cross_up[i](self.act(combined))
        
        return x_vars

class SimpleBlockTemporal(BaseBlockTemporal):
    def _apply_exchange(self, x_vars, x_add):
        return x_vars  # No exchange for simple mode

def create_block_temporal(exchange_mode, *args, **kwargs):
    if exchange_mode == 'sea':
        return SEABlockTemporal(*args, **kwargs)
    elif exchange_mode == 'addition':
        return AddBlockTemporal(*args, **kwargs)
    elif exchange_mode == 'simple':
        return SimpleBlockTemporal(*args, **kwargs)
    elif exchange_mode == 'pool':
        return SEAPoolBlockTemporal(*args, **kwargs)
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
                 ib_scale_mode='fourier', 
                 ib_addition_mode='add', 
                 ib_mlp_layers=1,
                 ib_num=1,
                 add_info_after_cross=True,
                 LN_type='adaln'):
        
        super().__init__()
        self.num_variables = num_variables
        self.exchange_mode = self._validate_exchange_mode(exchange_mode)
        self.pos_encoding_mode = self._validate_pos_encoding_mode(pos_encoding_mode)
        self.ib_scale_mode = ib_scale_mode
        self.ib_addition_mode = ib_addition_mode
        self.ib_num = ib_num
        self.LN_type = LN_type

        self.blocks = nn.ModuleList([
            create_block_temporal(
                self.exchange_mode, n_heads=n_heads, max_len=max_len, embed_dim=embed_dim,
                src_len=src_len, down_proj=down_proj, scale_ratio=scale_ratio,
                dropout=dropout, ib_scale_mode=self.ib_scale_mode, ib_addition_mode=self.ib_addition_mode,
                ib_mlp_layers=ib_mlp_layers, num_variables=num_variables,
                add_info_after_cross=add_info_after_cross,
                LN_type=self.LN_type
            )
            for _ in range(num_layers)
        ])

        if self.LN_type.lower() == 'adaln':
            self.ln = nn.ModuleList([AdaLN(embed_dim, self.ib_num) for _ in range(self.num_variables)])
        elif self.LN_type.lower() == 'ln':
            self.ln = nn.ModuleList([LayerNorm(embed_dim) for _ in range(self.num_variables)])
        else:
            raise ValueError(f"Invalid LN_type: {self.LN_type}. Must be one of {'adaln', 'ln'}.")

        self.apply(self._init_weights)

    def _validate_exchange_mode(self, mode):
        valid_modes = {'sea', 'simple', 'addition', 'pool'}
        mode = mode.lower()
        if mode not in valid_modes:
            raise ValueError(f"Invalid exchange_mode '{mode}'. Must be one of {valid_modes}.")
        return mode

    def _validate_pos_encoding_mode(self, mode):
        valid_modes = {'learnable', 'fixed'}
        if mode not in valid_modes:
            raise ValueError(f"Invalid pos_encoding_mode '{mode}'. Must be one of {valid_modes}.")
        return mode
    
    def _create_pos_encoder(self, mode, embed_dim, max_len, dropout): #TODO: removed since we have RoPE implemented
        if mode == 'learnable':
            return PositionalEncodingLearnable(embed_dim, max_len=5000, dropout=dropout)
        elif mode == 'fixed':
            return PositionalEncoding(embed_dim, dropout)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, AdaLN)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)


    def forward(self, x, x_additional_info):
        # x shape: [batch, time, field, cell]
        assert x.shape[2] == self.num_variables, f"Expected {self.num_variables} variables, but got {x.shape[2]}"
        x_vars = [x[:, :, i, :] for i in range(self.num_variables)]

        for block in self.blocks:
            x_vars = block(*x_vars, x_add=x_additional_info)

        x_vars = [self.ln[i](var, x_additional_info) for i, var in enumerate(x_vars)]

        x = torch.stack(x_vars, dim=2)
        return x
    