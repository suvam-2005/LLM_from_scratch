import os
import urllib.request
import re

if not os.path.exists('the-verdict.txt'):
    url=("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)
    
with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

#converting data to tokens

result = re.split(r'(?<=[.!?]) +', raw_text)
result = [item.strip() for item in result if item.strip()]
print(result)
preprocessed = result

# converting tokens to token ids

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

vocab = {token: i for i, token in enumerate(all_words)}

class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'(?<=[.!?]) +', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = ' '.join([self.int_to_str[i] for i in ids])
        text = re.sub(r' ([.!?])', r'\1', text)
        return text
    
tokenizer = SimpleTokenizer(vocab)

# Byte pair encoding (BPE) implementation
# pip install tiktoken 0.9.0

import tiktoken 

tokenizer = tiktoken.get_encoding("gpt2")
tokens = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})

# Data sampling for training

with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})

context_size = 5
x = enc_text[:context_size]
y = enc_text[1:context_size + 1]

#visualizing the data

for i in range(1, context_size + 1):
    context = enc_text[:i]
    desired = enc_text[i]
    print(context, "->", desired)
    print(tokenizer.decode(context), "->", tokenizer.decode([desired]))

import torch

from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt, tokenizer, max_length=256, stride=128, batch_size=4, shuffel=True, drop_last=True, num_workers=0):

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffel, drop_last=drop_last, num_workers=num_workers)
    
    return dataloader

with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, tokenizer, max_length=256, stride=128, batch_size=4, shuffel=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

# Creating token embeddings

import torch.nn as nn

input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
vocab_size = 10
embedding_dim = 8
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

print(embedding_layer.weight)

# Encoding word dimensions

vocab_size = 50257
embedding_dim = 256

token_embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

max_length = 4
dataloader = create_dataloader_v1(raw_text, max_length=4, stride=2, batch_size=2, shuffel=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Token IDs:\n", inputs)
print("\nInputs shape:", inputs.shape)

