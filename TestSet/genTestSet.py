contextsFile = open('contexts.txt', 'r')
questionsFile = open('questions.txt', 'r')

contexts = contextsFile.read().splitlines()
questions = questionsFile.read().splitlines()

with open('testSet.json', 'w') as file:
    for i in range(100):
        file.write('{\"prompt\": \"' + contexts[i] + '\", \"completion\": \" ' + questions[i] + '\"}\n')

contextsFile.close()
questionsFile.close()
