import torch
from transformers import BartTokenizer, BartForConditionalGeneration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BartTokenizer.from_pretrained("../04_bartBlankPrompt/model/bart-blank")
model = BartForConditionalGeneration.from_pretrained("../04_bartBlankPrompt/model/bart-blank")

with open('data/test.txt', 'r') as f:
    data = [line for line in f]

with open('data/completions.txt', 'r') as f2:
    completions = [line for line in f2]

for i in range(100):
    text = data[i]
    completion = completions[i]
    
    encoded_input = tokenizer(text, return_tensors = 'pt')
    labels = tokenizer(completion, return_tensors = 'pt')
    
    output = model.forward(encoded_input['input_ids'], labels=labels['input_ids'])
    
    print(output.loss.item())

f.close()
f2.close()
