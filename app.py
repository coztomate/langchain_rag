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
        return "Please upload a PDF and enter a question."

    if not pdf.name.lower().endswith('.pdf'):
        return "Please upload a PDF file."

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

with gr.Blocks() as demo:
    with gr.Accordion("Enter OpenAI API Key for the app to work"):
        openai_key = gr.Textbox(label="API Key", placeholder="Enter your OpenAI API Key here")

    gr.Markdown("Q&A on your PDF")
    with gr.Row():
        pdf = gr.File(label="Upload a PDF")
        user_question = gr.Textbox(label="Ask a question about this PDF:")
        answer_button = gr.Button("Answer")
    response = gr.Textbox(label="Response", interactive=False)

    answer_button.click(
        fn=process_pdf_and_answer_question,
        inputs=[pdf, user_question, openai_key],
        outputs=response
    )

demo.launch(share=True)
