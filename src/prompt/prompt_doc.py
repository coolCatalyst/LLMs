prompt_template = """You have to generate 10 diverse question-answer pairs based on following given context. 

context:
{context}


Here are the requirements:
1. Try not to repeat the verb for each question to maximize diversity.
2. The language used for the question also should be diverse. For example, you should combine questions with imperative instrucitons.
3. The type of questions should be diverse. The list should include diverse types of tasks like open-ended generation, classification, editing, etc.
4. An AI language model should be able to answer the question. For example, do not ask the assistant to create any visual or audio output. For another example, do not ask the assistant to wake you up at 5pm or set a reminder because it cannot perform any action.
5. The questions should be in English.
6. The questions should be 1 to 2 sentences long. Either an imperative sentence or a question is permitted.
7. The answer should be an appropriate response to the question. Make sure the answer is less than 100 words.
8. Questions should be independent each other. 

Output must be following type:
###
1. Question: <question 1>
1. Answer: <answer 1>
###
2. Question: <question 2>
2. Answer: <answer 2>

...

###
10. Question: <question 10>
10. Answer: <answer 10>


List of 10 tasks:
"""