import torch
import random
from transformers import BigBirdTokenizer, BigBirdForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BigBirdTokenizer.from_pretrained("./model/bigbird-clm")
model = BigBirdForCausalLM.from_pretrained("./model/bigbird-clm", is_decoder="True")

input = "<host> Hello! This is Calvin News Weekly. Today, we have Advait Scaria on the show! Hello, Mr. Scaria. <guest> Hello. <host>"

next_word = ''
for i in range(20):
	
	encoded_input = tokenizer(input, return_tensors = 'pt')
	output = model.forward(encoded_input['input_ids'])
	
	potential_words = (output.logits[0, -1].topk(5).indices).tolist()
	next_word = tokenizer.decode(potential_words[random.randint(0, len(potential_words)-1)])
	input += next_word
	print(input)
