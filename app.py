import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os

def process_pdf_and_answer_question(pdf, user_question, openai_key):
    if pdf is None or user_question == "":
        return "Please upload a PDF and enter a question.", None

    if not pdf.name.lower().endswith('.pdf'):
        return "Please upload a PDF file.", None

    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Set OpenAI API key from the provided input
    os.environ["OPENAI_API_KEY"] = openai_key

    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    docs = knowledge_base.similarity_search(user_question)
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
        print(cb)

    return response

# Define the Gradio Interface
gradio_app = gr.Interface(
    fn=process_pdf_and_answer_question,
    inputs=[
        gr.File(label="Upload a PDF"),
        gr.Textbox(label="Ask a question about this PDF:"),
        gr.Textbox(label="Enter your OpenAI API Key:")
    ],
    outputs=gr.Textbox(label="Response")
)

if __name__ == "__main__":
    gradio_app.launch()
