import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Hugging Face model
checkpoint = "MBZUAI/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype=torch.float32)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Load and process documents
persist_directory = "db"
loader = TextLoader('ThripitakaQuest_BackEnd\\app\statics\\tipitaka.txt', encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Create embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store
db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
db.persist()
db = None

def llm_pipeline():
    pipe = pipeline(
        task='text2text-generation',
        model=model,
        tokenizer=tokenizer,
        max_length=250,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

def qa_llm():
    llm = llm_pipeline()
    retriever = Chroma(persist_directory="db", embedding_function=embeddings).as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer


from flask import Blueprint, jsonify, request
from flask_cors import CORS

chat_bot = Blueprint('chat_bot', __name__)
CORS(chat_bot)

@chat_bot.route('/ask', methods=["POST"])
def ask():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'success'})
        response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
        response.headers['Access-Control-Allow-Methods'] = 'POST'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    try:
        data = request.get_json()
        user_input = data.get('query', '')
        if not user_input:
            return jsonify({'response': "Please provide a question"}),400
        
        answer = process_answer({'query': user_input})
       
        response = jsonify({'response': answer})
        response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
        print(answer)
        return response

    except Exception as e:
        error_response = jsonify({'error': str(e)})
        error_response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
        return error_response
    

  