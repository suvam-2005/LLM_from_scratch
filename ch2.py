import torch

inputs = torch.tensor([[1, 2, 3], 
                       [4, 5, 6],
                       [7, 8, 9]])

input_query = inputs[1]  # Select the second row: [4, 5, 6]

# Compute dot products between input_query and each row in inputs
dot_products = torch.matmul(inputs, input_query) #or torch.dot(inputs, input_query)

res=0
for idx, element in enumerate(inputs[0]):
    res += input[0][idx] * input_query[idx]
    res += torch.dot(element, input_query)
print(res)

query=inputs[1]

attn_scores_2=torch.empty(inputs.shape[0])
for i , x_i in enumerate(inputs):
    attn_scores_2[i]=torch.dot(x_i, input_query)
print(attn_scores_2)

attn_scores_2_temp = attn_scores_2/attn_scores_2.sum()
print(attn_scores_2_temp)

def softmax_naive(attn_scores):
    exp_scores = torch.exp(attn_scores)
    return exp_scores / exp_scores.sum(dim=0)

attn_probs = softmax_naive(dot_products)
print(attn_probs)

# method 2
torch.softmax(attn_scores_2, dim=0)

query=inputs[1]
context_vec_2 = torch.zeros_like(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_probs[i] * x_i

print(context_vec_2)

#general method

attn_scores = torch.empty(6,6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i,j] = torch.dot(x_i, x_j)

attn_weights = torch.softmax(attn_scores, dim=1)

all_context_vecs = attn_weights @ inputs

#simplest possible for attention mechanism
attn_scores = inputs @ inputs.T
attn_weights = torch.softmax(attn_scores, dim=1)
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

#Implementing self-attention with trainable weights

x_2= inputs[2]
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)

W_query=torch.nn.Parameter(torch.randn(d_in, d_out))
W_key=torch.nn.Parameter(torch.randn(d_in, d_out))
W_value=torch.nn.Parameter(torch.randn(d_in, d_out))

query_2 = x_2 @ W_query
key_2 = inputs @ W_key
value_2 = inputs @ W_value

keys=inputs @ W_key
value=inputs @ W_value

attn_scores_2 = query_2 @ keys.T 

d_k=keys.shape[1] #dimension of the keys

attn_weights_2 = torch.softmax(attn_scores_2/d_k**0.5, dim=-1)

context_vec_2 = attn_weights_2 @ value

#Multi-head Self-Attention
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query=torch.nn.Parameter(torch.randn(d_in, d_out))
        self.W_key=torch.nn.Parameter(torch.randn(d_in, d_out))
        self.W_value=torch.nn.Parameter(torch.randn(d_in, d_out))


    def forward(self, x):
        queries = inputs @ W_query
        keys = inputs @ W_key
        values = inputs @ W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores/d_k**0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec
    
torch.manual_seed(123)
sa_v1=SelfAttention_v1(d_in, d_out)

#Better implementation with nn.Linear

import torch.nn as nn

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query=torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key=torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value=torch.nn.Linear(d_in, d_out, bias=qkv_bias)


    def forward(self, x):
        queries = self.W_query(inputs)
        keys = self.W_key(inputs)
        values = self.W_value(inputs)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores/d_k**0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec
    
sa_v2 = SelfAttention_v2(d_in, d_out)
    
#Casual attention mask

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
values = sa_v2.W_value(inputs)

attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores/d_k**0.5, dim=-1)
        
#method 1 "NOT GOOD"

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))

masked_simple = attn_weights * mask_simple

rows_sum = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / rows_sum

#method2 for simpler method 1

mask = torch.tril(torch.ones(context_length, context_length)).bool()
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)

torch.exp(torch.tensor(float('-inf')))

attn_weights = torch.softmax(masked/d_k**0.5, dim=-1)

#Masking additional attention weights for dropout

torch.manual_seed(123)

layers=nn.Dropout(p=0.5)

#Implementating a compact casual self-attention class

import torch.nn as nn

inputs = torch.tensor([[0.4300, 0.1500, 0.8900], 
                       [0.5500, 0.8700, 0.6600],
                       [0.5700, 0.8500, 0.6400],
                       [0.2200, 0.5800, 0.3300],
                       [0.7700, 0.2500, 0.1000],
                       [0.0500, 0.8000, 0.5500]])

batch=torch.stack((inputs, inputs),dim=0)

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out,context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query=torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key=torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value=torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout=nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones( context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
            )
        attn_weights = torch.softmax(
            attn_scores/keys.shape[-1]**0.5, dim=-1
            )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values

        return context_vec
    
torch.manual_seed(789)
context_length=batch.shape[1]
ca = CausalAttention(d_in=3, d_out=2, context_length=context_length, dropout=0.1)
ca(batch)

#Multi-head Causal Self-Attention

class MultiHeadCausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads=2, qkv_bias=False):
        super().__init__()
        self.heads= nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
torch.manual_seed(789)

context_length=batch.shape[1]
d_in,d_out=batch.shape[0],2

mha= MultiHeadCausalAttention(d_in=d_in, d_out=d_out, context_length=context_length, dropout=0.1, num_heads=2)
mha(batch)

#simplified version using nn.MultiheadAttention

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False
    ):
        super().__init__()

        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Causal mask (upper triangular)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Linear projections
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        # Split into heads
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose for attention
        # (b, num_heads, num_tokens, head_dim)
        keys    = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values  = values.transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = queries @ keys.transpose(2, 3)

        # Apply causal mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)

        # Softmax
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5,
            dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec

