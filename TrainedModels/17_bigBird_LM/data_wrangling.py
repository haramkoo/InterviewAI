from transformers import PegasusTokenizer, BigBirdPegasusModel
import jsonlines

tokenizer = PegasusTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")

with jsonlines.open('data/valid_long.json', 'r') as f:
	with open('data/valid_lm.txt', 'w', encoding='utf-8') as writer:
		current_str = ''
		for obj in f:
			if obj['prompt'][:20].encode('utf-8').decode('utf-8') != current_str[:20] and current_str != '':
				writer.write(current_str + '\n')
			current_str = obj['prompt'] + obj['completion']
		writer.write(current_str + '\n')
