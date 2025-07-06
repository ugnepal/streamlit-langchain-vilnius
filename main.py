import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader


load_dotenv()

token = os.getenv("SECRET") # Replace with your actual token
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"



loader1 = WebBaseLoader("https://lt.wikipedia.org/wiki/Vilnius")
docs1 = loader1.load()
for doc in docs1:
    doc.metadata["source"] = "Wikipedia"


loader2 = WebBaseLoader("https://www.govilnius.lt/")
docs2 = loader2.load()
for doc in docs2:
    doc.metadata["source"] = "www.govilnius.lt"

from langchain_community.document_loaders import PyPDFLoader
pdf_loader = PyPDFLoader("Vilnius_vilnius.pdf")
docs3 = pdf_loader.load()
for doc in docs3:
    doc.metadata["source"] = "Vilnius PDF"

docs = docs1 + docs2 + docs3

text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
splits = text_spliter.split_documents(docs)


embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url="https://models.inference.ai.azure.com",
    api_key=token,
)

vectorstore = Chroma(embedding_function=embedding)

# Ingest documents in batches to avoid exceeding token limits
batch_size = 100
for i in range(0, len(splits), batch_size):
    batch = splits[i:i+batch_size]
    vectorstore.add_documents(batch)





retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
#prompt = hub.pull("rlm/rag-prompt")
prompt = hub.pull("rlm/rag-prompt").partial(
    instructions=(
        "Before answering, determine if the question is ONLY about Vilnius based on the following three sources: "
        "1) Wikipedia article at https://lt.wikipedia.org/wiki/Vilnius, "
        "2) website https://govilnius.lt, "
        "3) PDF Vilnius_vilnius.pdf. "
        "If the question is NOT about Vilnius or the answer is NOT found exactly in these sources, "
        "do NOT answer or attempt to generate any information. "
        "Respond ONLY with this exact sentence: 'Your question is not valid. I can only answer about Vilnius.' "
        "Do NOT add anything else. "
        "If the question is about Vilnius, answer ONLY using these sources and clearly cite each piece of information's source."
    )
)

def format_docs(docs):
    print(docs)
    return "\n\n".join(doc.page_content for doc in docs)

#st.title("Streamlit Langchain Vilnius Chat Demo")
st.image("Vilnius.jpg", use_container_width=True)
st.title("Streamlit Langchain Vilnius Chat Demo")



def generate_response(input_text):
    llm = ChatOpenAI(base_url=endpoint, temperature=0.7, api_key=token, model=model)

    fetched_docs = retriever.invoke(input_text)


    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    ) 

    result = rag_chain.invoke(input_text)     
    st.markdown(
    f"<div style='background-color:#1e3a5f; color:#ffffff; padding:15px; border-radius:8px; margin-bottom:20px;'>"
    f"{result}</div>", 
    unsafe_allow_html=True
    )


    st.subheader("📚 Sources")
    for i, doc in enumerate(fetched_docs, 1):
        source = doc.metadata.get("source", "Unknown source")
        with st.expander(f"Source {i} ({source})"):
            st.markdown(
                f"<div style='background-color:#e3f2fd; color:#000000; padding:15px; border-radius:8px; margin-bottom:20px;'>"
                f"{result}</div>", 
                unsafe_allow_html=True
    )




with st.form("my_form"):
    text = st.text_area(
        "I am Vilnius expert:",
        "Ask me anything about Vilnius, Lithuania.",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)
