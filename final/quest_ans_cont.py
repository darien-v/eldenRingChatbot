import openai
import csv
import random
from transformers import GPT2Tokenizer
import sys
import csv


csv.field_size_limit(1000000)
context_number = 1

# Remember to remove OpenAI API key
api_key = ""
client = openai.OpenAI(api_key=api_key)

def ask_gpt(precise_prompt, model_name = "gpt-3.5-turbo-0125"):
  message = [{"role":"system", "content":"Think critically about the following."},
             {"role":"user", "content": precise_prompt}]

  completion = client.chat.completions.create(
        model=model_name,
        messages=message
  )
  ret = completion.choices[0].message.content
  return ret

# initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def token_count(text):
    # tokenize the text and count the tokens
    tokens = tokenizer.tokenize(text)
    token_count = len(tokens)
    return token_count  # a basic approximation of tokens

def process_context(context, file_number):
    if token_count(context) > 16000:
        return  # skip large contexts to avoid exceeding the token limit
        
    question_types = ["what", "how", "why", "explain, who, when, where, describe"]
    questions = []
    answers = []

    for _ in range(5):
        question_type = random.choice(question_types)
        question_prompt = f"Given the following CONTEXT, generate a {question_type} question someone might ask related to the CONTEXT. Just return the question and do not add anything like Question: followed by the question. Only the question:\nCONTEXT\n{context}\n"
        if token_count(question_prompt) > 16000:
            continue
        
        question = ask_gpt(question_prompt)
        questions.append(question)
    
        answer_prompt = f"Given the following CONTEXT and QUESTION, generate an answer to the QUESTION using only information from the CONTEXT, and return only the answer. Keep the answer detailed but concise. Keep the answer confined to a single line. Do not place parts of the same answer on different line, only use a single line. If there are multiple parts, use commas instead of newline characters to separate them. Do not make lists, if you are listing multiple of something, just put them on a single line by putting the contents of the list between commas. Newline characters should NOT be used at all.:\nCONTEXT\n{context}QUESTION\n{question}\n"
        if token_count(answer_prompt) > 16000:
            continue
        answer = ask_gpt(answer_prompt)
        answers.append(answer)

    # write to files
    for index, (question, answer) in enumerate(zip(questions, answers)):
        filename = f"./tmp/{file_number + index}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"CONTEXT\n{context}\n\nQUESTION\n{question}\n\nANSWER\n{answer}\n")
    
file_number = 1

with open('../eldenRingWikiText.csv', 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        print(f"{context_number}/3714")
        context_number += 1
        if len(row) > 2:
            context = row[2]
            process_context(context, file_number)
            file_number += 5  # increment by 5 as we generate 5 sets of Q&A per context

print("Finished processing and writing to files.")