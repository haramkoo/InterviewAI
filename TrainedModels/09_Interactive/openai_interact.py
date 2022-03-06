import os
import openai

print('\nWelcome to the GPT-3 Interactive Interview AI!')
print('Tell us an interesting story, or type \'END\' to stop.')
prompt = input('> ')

while (prompt != 'END'):
    print('\nAI:', end="")
    print(openai.Completion.create(
        model="curie:ft-calvin-university-data-science-2021-09-23-19-56-19",
        prompt=prompt + "\n\n###\n\n",
        max_tokens=64,
        stop="END"
    )['choices'][0]['text'])
    prompt = input('> ')

print('\nGoodbye!\n')
