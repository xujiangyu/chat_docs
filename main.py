from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.llms import LlamaCpp
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pdf2image import convert_from_path, convert_from_bytes
import tempfile
from langchain.schema import Document
from baidu_trans import *

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Load environment variables
load_dotenv()


def image_to_base64(image):
    import base64
    import io

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()
    return image_base64


def pdf_to_images(file_bytes):
    with tempfile.TemporaryDirectory() as temp_dir:
        # 将上传的PDF文件转换为图像
        images = convert_from_bytes(file_bytes, output_folder=temp_dir)
    return images


def pdf_to_text(pdf_loader):
    texts = []
    text_str = ""

    pdf_reader = PdfReader(pdf_loader)
    for page in pdf_reader.pages:
        text = page.extract_text()
        texts.append(text)
        text_str += text

    return texts, text_str


def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=2000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    knowledgeBase = FAISS.from_texts(chunks, embeddings)

    return knowledgeBase


def main():
    # 仅支持cpu，使用GPU请修改参数
    llm = LlamaCpp(
        model_path="your model path",
        top_p=2,
        n_ctx=5096,
        # f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        verbose=True,
    )
    chain = load_qa_chain(llm, chain_type="stuff")
    st.title("Chat with your PDF 💬")
    global images, texts
    images = None
    texts = None
    text_str = None
    query = st.text_input("Ask a question to the PDF")
    cancel_button = st.button("Cancel")

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload your PDF Document", type="pdf")
        if uploaded_file:
            texts, text_str = pdf_to_text(uploaded_file)

            images = convert_from_bytes(uploaded_file.getvalue())

            # 在Streamlit中展示图像
            for i, image in enumerate(images):
                st.image(image, caption=f"第 {i + 1} 页", use_column_width=True)

    if text_str:
        knowledgeBase = process_text(text_str)
        selected_image_index = st.selectbox("select specific page", range(len(images)))
        genre = st.radio("结果是否翻译成中文？", options=["是", "否"])

    if images:
        print("selected_image_index", selected_image_index)

        if cancel_button:
            st.stop()
        if st.button("执行"):
            if selected_image_index > 0:
                select_text = texts[selected_image_index - 1]
                response = chain.run(
                    input_documents=[Document(page_content=select_text)],
                    question=query,
                )
                if genre == "是":
                    response = baidu_translate(response, app_id, secret_key)

                st.write(response)

            else:
                docs = knowledgeBase.similarity_search(query)
                response = chain.run(input_documents=docs, question=query)

                if genre == "是":
                    response = baidu_translate(response, app_id, secret_key)

                st.write(response)


if __name__ == "__main__":
    main()
