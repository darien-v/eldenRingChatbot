import prototype as Prototype
import openai
import sys
#Remember to remove key before pushing to repo
key = ""
client = openai.OpenAI(api_key=key)

def ask_gpt(precise_prompt, model_name = "gpt-3.5-turbo-0125"):
  message = [{"role":"system", "content":"Think critically about the following."},
             {"role":"user", "content": precise_prompt}]

  completion = client.chat.completions.create(
        model=model_name,
        messages=message
  )
  ret = completion.choices[0].message.content
  return ret
def wikiGPTAsk(question):
  prompt = "Given the following QUESTION and the CONTEXT necessary to answer the QUESTION, answer the QUESTION using only information from the CONTEXT, and return only the answer. Keep the answer detailed but concise:\n"
  context = Prototype.findTopic(question)
  prompt += "QUESTION\n"+question + "\n" + "CONTEXT\n" +context + "\n"
  response = ask_gpt(prompt)
  return response

#print(wikiGPTAsk("How do I beat Margit?")
question = input("Ask any question: ")
print(wikiGPTAsk(question))