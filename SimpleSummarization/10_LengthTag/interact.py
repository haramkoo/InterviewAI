import torch
from transformers import BartTokenizer, BartForConditionalGeneration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BartTokenizer.from_pretrained("./model/bart-length-tagged")
model = BartForConditionalGeneration.from_pretrained("./model/bart-length-tagged")

print('\nWelcome to the BART Interactive Interview AI with Length Tagging!')
print('Tell us an interesting story, or type \'END\' to stop.')
prompt = input('> ')

while (prompt != 'END'):
    tag = int(input('\n\tHow long should the response be?\n\tPlease enter a multiple of 10, up to 100: '))
    decoded_output = []    

    if (tag == 10):
        prompt += ' <TEN>'
    elif (tag == 20):
        prompt += ' <TWENTY>'
    elif (tag == 30):
        prompt += ' <THIRTY>'
    elif (tag == 40):
        prompt += ' <FORTY>'
    elif (tag == 50):
        prompt += ' <FIFTY>'
    elif (tag == 60):
        prompt += ' <SIXTY>'
    elif (tag == 70):
        prompt += ' <SEVENTY>'
    elif (tag == 80):
        prompt += ' <EIGHTY>'
    elif (tag == 90):
        prompt += ' <NINETY>'
    elif (tag == 100):
        prompt += ' <HUNDRED>'

    encoded_input = tokenizer(prompt, return_tensors = 'pt')
    output = model.generate(encoded_input['input_ids'], num_beams=4, max_length=1024)
    

    print('\nAI:', end="")
    decoded_output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output][0]
    word_list = decoded_output.split()
    print(decoded_output)
    print('Response Length: ' + str(len(word_list)))
    prompt = input('> ')

print('\nGoodbye!\n')
