import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import os
from langchain_openai import ChatOpenAI
import random
from streamlit.components.v1 import html

# Set up Streamlit page configuration
st.set_page_config(page_title="Mentoring Chatbot", page_icon="🤖", layout="wide")

# Add social media icons and links
st.markdown("""
    <div style="text-align: center; padding-bottom: 10px;">
        <a href="https://www.youtube.com/@TechProEducationUS" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg" width="30" style="margin-right: 10px;"></a>
        <a href="https://api.whatsapp.com/send/?phone=%2B15853042959&text&type=phone_number&app_absent=0" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg" width="30" style="margin-right: 10px;"></a>
        <a href="https://t.me/joinchat/HH2qRvA-ulh4OWbb" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/8/82/Telegram_logo.svg" width="30" style="margin-right: 10px;"></a>
        <a href="https://www.instagram.com/techproeducation/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Instagram_icon.png" width="30" style="margin-right: 10px;"></a>
        <a href="https://www.facebook.com/techproeducation" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg" width="30" style="margin-right: 10px;"></a>
        <a href="https://x.com/techproedu" target="_blank"><img src="https://abs.twimg.com/icons/apple-touch-icon-192x192.png" width="30" style="margin-right: 10px;"></a>
        <a href="https://linkedin.com/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/commons/0/01/LinkedIn_Logo.svg" width="30" style="margin-right: 10px;"></a>    
    </div>
""", unsafe_allow_html=True)

# Display the image at the top of the page with a clickable link
st.markdown("""
    <div style="text-align: center;">
        <a href="https://www.techproeducation.com/" target="_blank">
            <img src="https://media.licdn.com/dms/image/v2/D4D3DAQE9q_OhttkyPw/image-scale_191_1128/image-scale_191_1128/0/1713367311427/techproeducationtr_cover?e=1732885200&v=beta&t=5-uB4D7yBL8JFKvdJeCw8AjMCwcDEMo7R9d52RI1Ho8" 
            alt="Techpro Education Cover" width="100%" style="border-radius: 10px;"/>
        </a>
    </div>
""", unsafe_allow_html=True)

st.title("Chat with Techpro Education 💬")

 # Add sidebar with information
with st.sidebar:
    st.title("About")
    st.markdown("""
    Company:    
    
    Techproeducation provides quality online IT courses and coding bootcamps with reasonable prices to prepare individuals for next-generation jobs from beginners to IT professionals. 
    We offer cutting-edge programs used today by leading corporations.

    Contact:
    
    +1 585 304 29 59   
    info@techproeducation.com    
    New York City, NY USA
                
    Programs:
    
    FREE ONLINE IT COURSES                
    AUTOMATION ENGINEER                
    SOFTWARE DEVELOPMENT                
    CLOUD ENGINEERING & SECURITY                
    DATA SCIENCE                
    DIGITAL MARKETING
    """)

    # Upload Excel file
excel_file = "Cleaned_Mentoring_data.xlsx"
data = pd.read_excel(excel_file)

# Convert questions and answers to a list
questions = data['Questions'].tolist()  
answers = data['Answer'].tolist() 

# Create document objects
documents = [Document(page_content=f"{row['Questions']}\n{row['Answer']}") for _, row in data.iterrows()]

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Determine embedding model
model_name = "BAAI/bge-base-en"  
encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity

# Establish embeddings model
bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs
)

# Create database
persist_directory = 'db'
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)  

# Establish vector database (vectorstore)
vectordb = Chroma.from_documents(documents=texts,
                                 collection_name="rag-chroma",
                                 embedding=bge_embeddings,
                                 persist_directory=None)

retriever = vectordb.as_retriever()

# Initialize message history
if "messages" not in st.session_state:  # Initialize session state if not already initialized
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Data Science!"}]
    
# Wrap the prompt in a function
def prompt_fn(query: str, context: str) -> str:
    return f"""
    You are a Data Science Instructor. 
    If the user's query matches any question from the database, return the corresponding answer directly.
    Otherwise, answer the user's question using the information from the context below. 
    If you don't find the answer in the context, respond with "Konu dışı sorduğunuz sorulara cevap veremiyorum."

    Context: {context}
    
    User's question: {query}"""

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
import streamlit as st

# .env dosyasını yükleyin
load_dotenv()

# Ortam değişkenlerinden değerleri alın
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("MODEL_NAME")  # Varsayılan değer
temperature = float(os.getenv("TEMPERATURE"))  # Varsayılan: 0
max_tokens = int(os.getenv("MAX_TOKENS"))  # Varsayılan: 500

print(f"Model Name: {model_name}")
print(f"Temperature: {temperature}")
print(f"Max Tokens: {max_tokens}")

# LLM model
@st.cache_resource
def load_llm():
    return ChatOpenAI(model_name= model_name, temperature= temperature, max_tokens= max_tokens)

llm = load_llm()

@st.cache_resource
def create_rag_chain():
    from langchain_core.runnables import RunnableLambda
    prompt_runnable = RunnableLambda(lambda inputs: prompt_fn(inputs["query"], inputs["context"]))
    return prompt_runnable | llm | StrOutputParser()

rag_chain = create_rag_chain()

# Generate response
def generate_response(query):
    # Search the vector database for the most relevant answer
    results = retriever.get_relevant_documents(query)[:3]  # Top 3 results
    context = "\n".join([doc.page_content for doc in results])
    inputs = {"query": query, "context": context}
    response = rag_chain.invoke(inputs)

    # Suggest three random related questions
    related_questions = random.sample(questions, k=3)  # Randomly select 3 questions
    suggestions = "\n".join([f"- {q}" for q in related_questions])

    return response, suggestions

# Display chat messages from session state
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user query input
if query := st.chat_input("Your question"):
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, suggestions = generate_response(query)
            st.write(response)
            st.write("Şu soruları sorabilirsiniz: ")
            st.markdown(suggestions)  # Removed link formatting

            st.session_state["messages"].append({"role": "assistant", "content": response})

# Kullanıcıdan bir soru sorulması
user_question = "Yapay zeka nedir?"  # Burayı istediğiniz bir soru ile değiştirebilirsiniz

# Cevap üret
response, suggestions = generate_response(user_question)

# Çıktıları görüntüle
print("Cevap:")
print(response)

# Add robot avatar to the right of chat input with the name "Techie"
avatar_html = """
<style>
.robot-avatar {
    position: fixed;
    right: 30px;
    bottom: 50px;
    width: 80px;
    height: 80px;
    background: linear-gradient(45deg, #32CD32, #FFFFFF);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: float 2s ease-in-out infinite;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
    text-align: center;
    font-family: Arial, sans-serif;
}

.robot-avatar img {
    width: 50px;
    height: 50px;
    border-radius: 50%;
}

.robot-name {
    position: absolute;
    top: -25px;
    left: 10px;
    font-size: 16px;
    font-weight: bold;
    color: #32CD32;
    background-color: white;
    padding: 2px 10px;
    border-radius: 50px;    
    text-align: center;
}

@keyframes float {
    0%, 100% {
        transform: translateY(-5px);
    }
    50% {
        transform: translateY(5px);
    }
}
</style>
<div class="robot-avatar">
    <div class="robot-name">Techie </div>
    <img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png" alt="Robot Avatar">
</div>
"""
html(avatar_html, height=200)
