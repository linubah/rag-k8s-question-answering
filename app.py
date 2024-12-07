import gradio as gr
import pandas as pd
from rag.main import QueryAnswerBot

bot = QueryAnswerBot()

def answer_question(question: str, api_key: str):
    bot.set_api_key(api_key)
    return bot.get_answer(question)

docs_links = pd.read_csv("data/data_links.csv")
links = docs_links["link"].tolist()
links_markdown = "\n".join([f"- [{link}]({link})" for link in links])

description = """
<h1 style="text-align: center;">AI-Powered Kubernetes Documentation Assistant</h1>
<p style="font-size: 14px;">This tool uses <b>Retrieval-Augmented Generation (RAG)</b> to provide answers to your questions based on Kubernetes (K8s) documentation.  
Simply type your question, and the AI will provide a detailed answer!</p>

<h5>&#128679; <b>Important:</b> You need to provide a valid <b>GROQ_API_KEY</b> to use this tool. You can get your API key by signing up at <a href="https://console.groq.com/keys">GroqCloud</a>.</h5>
<h5>&#128680; <b>Warning:</b> Remember that it's a beta-RAG and its using small database :)</h5>

#### Topics Overview:
Click the toggle below to view the full list of available topics.
"""

example_questions = [
    "What is a Kubernetes?",
    "What main components there are in k8s?",
    "The Kubernetes network model is based on what?"
]

with gr.Blocks() as demo:
    gr.Markdown(description)
    question = gr.Textbox(label="Enter your question here:", lines=3, placeholder="Ask a question about Kubernetes...")
    token = gr.Textbox(label="GROQ_API_KEY:", placeholder="Enter your GROQ_API_KEY here", type="password")
    with gr.Accordion("Availiable topics:", open=False):
        gr.Markdown(links_markdown)
    
    interface = gr.Interface(fn=answer_question, 
                    inputs=[
                        question, token
                    ],
                    outputs=gr.Textbox(label="AI-Powered Answer:", lines=10),
    )
    gr.Examples(examples=example_questions, inputs=[question])

demo.launch()
