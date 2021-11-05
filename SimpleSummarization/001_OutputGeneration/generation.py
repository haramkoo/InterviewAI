import random
import json_lines
import os
import openai
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

openai.api_key = "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"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BartTokenizer.from_pretrained("../output")
model = BartForConditionalGeneration.from_pretrained("../output")

with json_lines.open('transformers_output.json', 'r') as jsonl_f:
    data = [obj for obj in jsonl_f]

prompts = []
completions = []
openft = []
summ = []

for i in range(100):
    num = random.randint(20001, 79000)
    text = data[num]['prompt']
    complete = data[num]['completion']
    prompts.append(text)
    completions.append(complete)

    encoded_input = tokenizer(text, return_tensors = 'pt')
    output = model.generate(encoded_input['input_ids'], num_beams=4, max_length=64)

    summ.append([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in output][0])

    openft.append(openai.Completion.create(
        model="curie:ft-calvin-university-data-science-2021-09-23-19-56-19",
        prompt=text,
        max_tokens=64,
        stop="END"
    )['choices'][0]['text'])

for i in range(len(prompts)):
    print(prompts[i] + "---" + completions[i] + "---" + openft[i] + "---" + summ[i])

jsonl_f.close()
