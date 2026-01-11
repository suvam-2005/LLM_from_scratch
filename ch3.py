GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "num_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.position_embedding = nn.Embedding(cfg["content_length"],cfg["emd_dim"])  # assuming max seq length of 512
        self.drop_emb = nn.Dropout(cfg["emb_dropout"])

        self.trf_blocks=nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["num_layers"])])

        self.final_norm = DummyGPTModel(cfg["emb_dim"])
        self.output_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.size()
        tok_embeds = self.token_embedding(in_idx)  # (B, T, d_model)
        pos_embeds = self.position_embedding(
            torch.arange(seq_len, device=in_idx.device)
        ).unsqueeze(0)  # (1, T, d_model)
        x = tok_embeds(x) + pos_embeds(x)  # (B, T, d_model)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)  # (B, T, d_model)
        x = self.final_norm(x)  # (B, T, d_model)
        logits = self.output_head(x)  # (B, T, vocab_size)
        return logits
    
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        # Placeholder for actual forward pass
        return x
    
class DummyLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
    
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

batch=[]

txt1="kjdbnclksdn cncwjk jkfbnc"
txt2="nckjdbnc jkfbnc kjdbnclksdn cncwjk"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))  
batch=torch.stack(batch, dim=0)  # (B, T)
print(batch)

torch.manual_seed
model=DummyGPTModel(cfg=GPT_CONFIG_124M)
logits=model(batch)
print(logits.shape)  # (B, T, vocab_size)
print(logits)

torch.manual_seed(789)
batch_example=torch.randn(2,5)  # (B, T, d_in)\

layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
out=layer(batch_example)
mean=out.mean(dim=-1, keepdim=True)

var=out.var(dim=-1, keepdim=True)

normed=((out-mean)/torch.sqrt(var))
normed.var(dim=-1, keepdim=True)

torch.set_printoptions(sci_mode=False)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        normed = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * normed + self.shift

ln=LayerNorm(6)
output_normed=ln(out)
output_normed.mean(dim=-1, keepdim=True)
output_normed.var(dim=-1, keepdim=True)

#Implementing a feed_forward network

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"]),
        )

        def forward(self, x):
            return self.layers(x)

ffn = FeedForward(GPT_CONFIG_124M)

x=torch.randn(2,3,768)  # (B, T, d_model)
ffn(x).shape  # (B, T, d_model) B stands for batch size, T for sequence length and d_model for embedding dimension


#Implementing shortcut connections

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcuts):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcuts and x.shape == layer_output.shape:
                x = x + layer_output  # Shortcut connection
            else:
                x=layer_output
        return x

def print_gradients(model, x):
    output=model(x)
    targeet=torch.tensor([[0.]])        

    loss=nn.MSELoss()
    loss=loss(output, targeet)
    loss.backward()
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

layer_sizes=[3,3,3,3,3,1]
sample_input=torch.tensor([[1.,0.,-1.]])

torch.manual_seed(42)
model_without_shortcuts=ExampleDeepNeuralNetwork(layer_sizes, use_shortcuts=False)
print(model_without_shortcuts, sample_input)

