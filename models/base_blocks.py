import torch
import torch.nn as nn
import torch.nn.functional as FF
import numpy as np
import math


class upScaleMLP(nn.Module):
    def __init__(self,d_model,d_output,hidden_dim):
        super(upScaleMLP, self).__init__()
        self.d_model = d_model
        self.d_output = d_output
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Linear(d_model,hidden_dim,bias=False)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim,d_output)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

class downScaleMLP(nn.Module):
    def __init__(self,d_input,d_model,hidden_dim):
        super(downScaleMLP, self).__init__()
        self.d_model = d_model
        self.d_input = d_input
        self.layer1 = nn.Linear(d_input,hidden_dim,bias=False)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim,d_model)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim_in, dropout, scale_ratio=4, dim_out=None, num_layers=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in

        self.residual_projection = None
        if dim_in != dim_out:
            self.residual_projection = nn.Linear(dim_in, dim_out)

        scaled_dim = max(1, int(dim_in * scale_ratio))

        layers = []
        if num_layers is None or num_layers == 1:
            layers.append(nn.Linear(dim_in, scaled_dim))
            layers.append(nn.LayerNorm(scaled_dim))
            layers.append(nn.GELU())
            layers.append(nn.Linear(scaled_dim, dim_out))
        else:
            for i in range(num_layers):
                if i == 0:
                    layers.append(nn.Linear(dim_in, scaled_dim))
                    layers.append(nn.LayerNorm(scaled_dim))
                elif i == num_layers - 1:
                    layers.append(nn.Linear(scaled_dim, dim_out))
                else:
                    layers.append(nn.Linear(scaled_dim, scaled_dim))
                    layers.append(nn.LayerNorm(scaled_dim))

                if i != num_layers - 1:  # Don't add GELU after the last layer
                    layers.append(nn.GELU())

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.dropout(x)

class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return FF.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# Not masked for encoder part
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = embed_dim

        self.k = nn.Linear(embed_dim, self.head_dim*n_heads)
        self.q = nn.Linear(embed_dim, self.head_dim*n_heads)
        self.v = nn.Linear(embed_dim, self.head_dim*n_heads)
        self.projection = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        B,T,C = x.shape
        # B,T,C
        k = self.k(x).view(B,T,self.n_heads,self.head_dim).transpose(1,2) # [B,T, H_dim*n_H] -> [B, T, n_H, H_dim] -> [B, n_H, T, H_dim]
        q = self.q(x).view(B,T,self.n_heads,self.head_dim).transpose(1,2) # [B,T, H_dim*n_H] -> [B, T, n_H, H_dim] -> [B, n_H, T, H_dim]
        v = self.v(x).view(B,T,self.n_heads,self.head_dim).transpose(1,2) # [B,T, H_dim*n_H] -> [B, T, n_H, H_dim] -> [B, n_H, T, H_dim]

        attn =(q @ k.transpose(-2,-1) ) * (self.head_dim)**-0.5 #[B, n_H, T, T]
        attn = FF.softmax(attn,dim=-1)
        attn = self.dropout(attn)

        out = attn @ v #[B, n_H, T, T] * [B, n_H, T, H_dim] = [B, n_H, T, H_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return out

class EncoderBlock(nn.Module):

    def __init__(self, n_heads, max_len, embed_dim, src_len, dropout):
        super().__init__()

        self.ln_exp1_1 = LayerNorm(embed_dim, bias=False)
        self.ln_exp1_2 = LayerNorm(embed_dim, bias=False)

        self.attn_1 = MultiHeadAttention(n_heads, embed_dim, dropout)

        self.mlp_1 = MLP(embed_dim, dropout)

    def forward(self, x):
        x = x + self.attn_1(self.ln_exp1_1(x))
        output = x + self.mlp_1(self.ln_exp1_2(x))
        return output
    

# Taken from the "Score-Based Generative Modeling Through Stochastic Differential Equations"
# https://github.com/yang-song/score_sde_pytorch/tree/main
class GaussianFourierProjection(nn.Module):
    def __init__(self, input_dim, half_dim=256, scale=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.half_dim = half_dim
        self.W = nn.Parameter(torch.randn(input_dim, half_dim) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x @ self.W * 2 * np.pi  # [B,T,input_dim] @ [input_dim,half_dim] -> [B,T,half_dim]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # [B,T,2*half_dim]


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, max_len, src_len, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.max_len = max_len
        self.head_dim = embed_dim // n_heads
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = embed_dim

        self.k = nn.Linear(embed_dim, self.head_dim*n_heads)
        self.q = nn.Linear(embed_dim, self.head_dim*n_heads)
        self.v = nn.Linear(embed_dim, self.head_dim*n_heads)
        self.projection = nn.Linear(embed_dim, embed_dim, bias=False)

        # src_len is created to allow model see the length of src if we have any src
        self.register_buffer('tril',torch.tril(torch.ones(max_len,max_len), diagonal=src_len).view(1,1,max_len,max_len))

    def forward(self, x):
        B,T,C = x.shape
        # B,T,C
        k = self.k(x).view(B,T,self.n_heads,self.head_dim).transpose(1,2) # [B,T, H_dim*n_H] -> [B, T, n_H, H_dim] -> [B, n_H, T, H_dim]
        q = self.q(x).view(B,T,self.n_heads,self.head_dim).transpose(1,2) # [B,T, H_dim*n_H] -> [B, T, n_H, H_dim] -> [B, n_H, T, H_dim]
        v = self.v(x).view(B,T,self.n_heads,self.head_dim).transpose(1,2) # [B,T, H_dim*n_H] -> [B, T, n_H, H_dim] -> [B, n_H, T, H_dim]

        attn =(q @ k.transpose(-2,-1) ) * (self.head_dim)**-0.5 #[B, n_H, T, T]

        attn = attn.masked_fill(self.tril[:, :, :T, :T]==0,float('-inf'))
        attn = FF.softmax(attn,dim=-1)
        attn = self.dropout(attn)

        out = attn @ v #[B, n_H, T, T] * [B, n_H, T, H_dim] = [B, n_H, T, H_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, max_len, src_len, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.max_len = max_len
        self.head_dim = embed_dim // n_heads
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = embed_dim

        self.k = nn.Linear(embed_dim, self.head_dim*n_heads)
        self.q = nn.Linear(embed_dim, self.head_dim*n_heads)
        self.v = nn.Linear(embed_dim, self.head_dim*n_heads)
        self.projection = nn.Linear(embed_dim, embed_dim, bias=False)

        self.register_buffer('tril',torch.tril(torch.ones(max_len,max_len), diagonal=src_len).view(1,1,max_len,max_len))

    def forward(self, x_1, x_2): # x_1 comes from decoder, x_2 comes from encoder
        """X1 And X2 have different shapes restructure q, k, v according to their own shapes"""
        B_tgt,T_tgt,C_tgt = x_1.shape # B,T,C
        B_src,T_src,C_src = x_2.shape

        q = self.q(x_1).view(B_tgt,T_tgt,self.n_heads,self.head_dim).transpose(1,2) # [B_tgt,T_tgt, H_dim*n_H] -> [B, T, n_H, H_dim] -> [B, n_H, T_tgt, H_dim]

        v = self.v(x_2).view(B_src,T_src,self.n_heads,self.head_dim).transpose(1,2) # [B,T, H_dim*n_H] -> [B, T, n_H, H_dim] -> [B, n_H, T_src, H_dim]
        k = self.k(x_2).view(B_src,T_src,self.n_heads,self.head_dim).transpose(1,2) # [B,T, H_dim*n_H] -> [B, T, n_H, H_dim] -> [B, n_H, T_src, H_dim]

        attn = (q @ k.transpose(-2,-1) ) * (self.head_dim)**-0.5 #[B, n_H, T_tgt, T_src]
        #attn = attn.masked_fill(self.tril[:, :, :T_tgt, :T_src]==0,float('-inf'))
        attn = FF.softmax(attn,dim=-1)
        attn = self.dropout(attn)

        out = attn @ v #[B, n_H, T_tgt, T_src] * [B, n_H, T_src, H_dim] = [B, n_H, T_tgt, H_dim]
        out = out.transpose(1, 2).contiguous().view(B_tgt, T_tgt, C_tgt) #[B, T, N_H, H_dim] -> [B, T, N_H*H_dim] = [B, T, C]

        return out

class MultiHeadCrossAttentionV2P(nn.Module):
    def __init__(self, n_heads, embed_dim, max_len, src_len, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.max_len = max_len
        self.head_dim = embed_dim // n_heads
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = embed_dim

        self.k = nn.Linear(embed_dim, self.head_dim*n_heads)
        self.q = nn.Linear(embed_dim, self.head_dim*n_heads)
        self.v = nn.Linear(embed_dim, self.head_dim*n_heads)
        self.projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(max_len,max_len)).view(1,1,max_len,max_len))

    def forward(self, x_1, x_2):
        B,T,C = x_1.shape # B,T,C
        q = self.q(x_1).view(B,T,self.n_heads,self.head_dim).transpose(1,2) # [B,T, H_dim*n_H] -> [B, T, n_H, H_dim] -> [B, n_H, T, H_dim]

        v = self.v(x_2).view(B,T,self.n_heads,self.head_dim).transpose(1,2) # [B,T, H_dim*n_H] -> [B, T, n_H, H_dim] -> [B, n_H, T, H_dim]
        k = self.k(x_2).view(B,T,self.n_heads,self.head_dim).transpose(1,2) # [B,T, H_dim*n_H] -> [B, T, n_H, H_dim] -> [B, n_H, T, H_dim]

        attn =(q @ k.transpose(-2,-1) ) * (self.head_dim)**-0.5 #[B, n_H, T, T]
        attn = attn.masked_fill(self.tril[:, :, :T, :T]==0,float('-inf'))
        attn = FF.softmax(attn,dim=-1)
        attn = self.dropout(attn)

        out = attn @ v #[B, n_H, T, T] * [B, n_H, T, H_dim] = [B, n_H, T, H_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return out

class MultiHeadCrossAttentionP2V(nn.Module):
    def __init__(self, n_heads, embed_dim, max_len, src_len, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.max_len = max_len
        self.head_dim = embed_dim // n_heads
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = embed_dim

        self.k = nn.Linear(embed_dim, self.head_dim*n_heads)
        self.q = nn.Linear(embed_dim, self.head_dim*n_heads)
        self.v = nn.Linear(embed_dim, self.head_dim*n_heads)
        self.projection = nn.Linear(embed_dim, embed_dim, bias=False)

        # Registered as buffer so it's saved and loaded with the model
        self.register_buffer('tril',torch.tril(torch.ones(max_len,max_len), diagonal=src_len).view(1,1,max_len,max_len))

    def forward(self, x_1, x_2): # Assume both x_1, and x_2 have the same shape
        B,T,C = x_1.shape # B,T,C
        q = self.q(x_1).view(B,T,self.n_heads,self.head_dim).transpose(1,2) # [B,T, H_dim*n_H] -> [B, T, n_H, H_dim] -> [B, n_H, T, H_dim]

        v = self.v(x_2).view(B,T,self.n_heads,self.head_dim).transpose(1,2) # [B,T, H_dim*n_H] -> [B, T, n_H, H_dim] -> [B, n_H, T, H_dim]
        k = self.k(x_2).view(B,T,self.n_heads,self.head_dim).transpose(1,2) # [B,T, H_dim*n_H] -> [B, T, n_H, H_dim] -> [B, n_H, T, H_dim]

        attn = (q @ k.transpose(-2,-1) ) * (self.head_dim)**-0.5 #[B, n_H, T, T]
        attn = attn.masked_fill(self.tril[:, :, :T, :T]==0,float('-inf'))
        attn = FF.softmax(attn,dim=-1)
        attn = self.dropout(attn)

        out = attn @ v #[B, n_H, T, T] * [B, n_H, T, H_dim] = [B, n_H, T, H_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C) #[B, T, N_H, H_dim] -> [B, T, N_H*H_dim] = [B, T, C]

        return out
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        print(f"This is pe shape: {pe.shape}")
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PositionalEncodingLearnable(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0):

        super(PositionalEncodingLearnable, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        seq_len = x.size(1)
        pos_embeddings = self.pos_embedding[:seq_len, :]
        x = x + pos_embeddings
        x = self.dropout(x)
        return x



