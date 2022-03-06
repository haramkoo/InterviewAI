import torch
from transformers import BartTokenizer, BartForConditionalGeneration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BartTokenizer.from_pretrained("../03_bart20000/model/output")
model = BartForConditionalGeneration.from_pretrained("../03_bart20000/model/output")

with open('test.txt', 'r') as f:
    data = [obj for obj in f]

for i in data:
    encoded_input = tokenizer(i, return_tensors = 'pt')
    output = model.generate(encoded_input['input_ids'], num_beams=4, max_length=64)

    print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output][0])

f.close()
