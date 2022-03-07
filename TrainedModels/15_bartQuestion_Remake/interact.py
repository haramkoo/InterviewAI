import torch
from transformers import BartTokenizer, BartForConditionalGeneration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BartTokenizer.from_pretrained("./model/bart_question_remake")
model = BartForConditionalGeneration.from_pretrained("./model/bart_question_remake")
valid_response = [1, 2, 3, 4, 5]

print('\nWelcome to the BART Interactive Interview AI (with diverse beam search)!')
print('Tell us an interesting story, or type \'END\' to stop.')
prompt = input('> ')

while (prompt != 'END'):
    encoded_input = tokenizer(prompt, return_tensors = 'pt')
    output = model.generate(encoded_input['input_ids'], num_beams=10, max_length=64, num_return_sequences=10, diversity_penalty=1.0, num_beam_groups=2)
    decoded_beams = []
    num = '0'

    print('\nPotential AI Outputs:')
    for i in range(5):
        print(str(i + 1) + '.', end="")
        decoded_beams.append([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output][i])
        print(decoded_beams[i])

    while int(num) not in valid_response:
        print('\nPlease select which question you want to use (1 to 5)')
        num = input('> ')
    
    print('\nYou have chosen: ' + num + '.' + decoded_beams[int(num) - 1])
    print('\nPlease provide a response:')
    prompt = input('> ')

#print('\nGoodbye!\n')
