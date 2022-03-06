import torch
from transformers import BigBirdTokenizer, BigBirdForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BigBirdTokenizer.from_pretrained("./model/bigbird-clm")
model = BigBirdForCausalLM.from_pretrained("./model/bigbird-clm", is_decoder="True")

input = "<host> Hello! This is Calvin News Weekly. Today, we have Advait Scaria on the show! Hello, Mr. Scaria. <guest> Hello. <host>"
encoded_input = tokenizer(input, return_tensors = 'pt')
output = model.forward(encoded_input['input_ids'])

print([tokenizer.decode(item) for item in output.logits.topk(5).indices[0]])
