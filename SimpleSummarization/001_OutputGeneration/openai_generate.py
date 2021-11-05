import os
import openai

openai.api_key = "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"
with open('OpenAI_Test.txt', 'r') as f:
    data = [obj for obj in f]

for i in data:
    print("{" + openai.Completion.create(
        model="curie:ft-calvin-university-data-science-2021-09-23-19-56-19",
        prompt=i + "\n\n###\n\n",
        max_tokens=64,
        stop="END"
    )['choices'][0]['text'] + "}")

f.close()
