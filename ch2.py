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

