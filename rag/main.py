import os
from openai import OpenAI
from .retriever import Retriever


class QueryAnswerBot:

    PROMPT = """
The following is a conversation with an AI assistant. 
The assistant is focused, concise, and uses simple language to deliver answers directly and clearly.

Rules:
- Use **only** the provided context to answer the question.
- If the context does not contain the answer, reply only with: "I don't know 🤷".
- Avoid adding extra or irrelevant information.
- Ensure answers are simple, clear, and provide only essential details.
- Don't use constuctions like this "According to the provided context".
"""

    def __init__(self):
        self.retriever = Retriever()


    def set_api_key(self, api_key):
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )    

    def get_answer(self, question: str):

        context, urls = self.retriever.get_docs(question)
        
        messages = [
                {"role": "system", "content": self.PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\nQuestion:\n{question}"
                }
            ]

        completion = self.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages
        )

        bot_answer = completion.choices[0].message.content
        references = "\n".join([f"- {url}" for url in urls])
        result = f"{bot_answer}\n\nReferences:\n{references}"
        return result



if __name__ == "__main__":
    question = "The Kubernetes network model is based on what?"
    bot = QueryAnswerBot()
    answer = bot.get_answer(question)

    print("Question:", question)
    print("Answer:", answer)