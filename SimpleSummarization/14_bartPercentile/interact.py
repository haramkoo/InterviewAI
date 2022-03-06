import torch
from transformers import BartTokenizer, BartForConditionalGeneration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

modelpath = './model/bart_completion_percentile'

tokenizer = BartTokenizer.from_pretrained(modelpath)
model = BartForConditionalGeneration.from_pretrained(modelpath)

print('\nWelcome to the BART Interactive Interview AI!')
print('Tell us an interesting story, or type \'END\' to stop.')
prompt = input('> ')

while (prompt != 'END'):
    encoded_input = tokenizer(prompt, return_tensors = 'pt')
    output = model.generate(encoded_input['input_ids'], num_beams=4, max_length=64)

    print('\nAI:', end="")
    print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output][0])
    prompt = input('> ')

print('\nGoodbye!\n')
