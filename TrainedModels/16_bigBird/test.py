from transformers import PegasusTokenizer, BigBirdPegasusModel
import jsonlines

tokenizer = PegasusTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")

with jsonlines.open('data/valid_long.json', 'r') as f:
	with jsonlines.open('data/valid_trunc.json', mode='w') as writer:
		for obj in f:
			if len(obj['prompt']) < 3000:
				writer.write(obj)

