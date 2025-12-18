# MedBRAIN AI - Medical Assistant Chatbot
# Powered by LangChain + Groq + Open Source RAG

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="MedBRAIN AI - Medical Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .disclaimer {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    h1 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🧠 MedBRAIN AI")
st.caption("Educational Medical Assistant powered by Open-Source RAG + Groq")

# Medical disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ Medical Disclaimer:</strong> MedBRAIN AI is for educational purposes only.
    It does NOT provide medical advice, diagnoses, or treatment recommendations.
    Always consult a licensed healthcare professional for medical decisions.
    In emergencies, contact your local emergency services immediately.
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Enter your Groq API key"
    )
    
    model_name = st.selectbox(
        "Select Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        help="Choose the LLM model"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Lower = more focused, Higher = more creative"
    )
    
    st.divider()
    
    st.markdown("### 💡 Example Questions")
    example_questions = [
        "What is asthma?",
        "Explain MRI to a layperson",
        "Risk factors for hypertension?",
        "What causes diabetes?"
    ]
    
    for q in example_questions:
        if st.button(q, key=q):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()
    
    st.divider()
    st.markdown("""   
    **Built with:**
    - 🦙 LangChain
    - ⚡ Groq
    - 📚 Gale Medical Encyclopedia
    - 🎨 Streamlit
    """)

# Initialize RAG chain
@st.cache_resource
def init_rag_chain(_api_key, _model_name, _temperature):
    try:
        # Initialize LLM
        llm = ChatGroq(
            model=_model_name,
            api_key=_api_key,
            temperature=_temperature,
            max_tokens=1024
        )
        
        # Load vector database
        vectordb = Chroma(
            persist_directory="chroma_db",
            embedding_function=None
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        
        # System prompt
        SYSTEM_PROMPT = """You are MedBRAIN AI, an educational medical knowledge assistant.
        
IMPORTANT RULES:
        1. Use ONLY the provided context from the Gale Encyclopedia of Medicine
        2. NEVER make diagnoses or prescribe treatments
        3. ALWAYS remind users to consult licensed healthcare professionals
        4. If information is insufficient, clearly state this
        5. Use clear, layperson-friendly language with medical terms explained
        6. Structure your response with headings and bullet points when helpful
        
Context from medical encyclopedia:
        {context}
        
User question: {question}"""
        
        prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        
        def format_docs(docs):
            return "\n\n".join([f"Source {i+1}:\n{d.page_content}" for i, d in enumerate(docs)])
        
        # Build RAG chain
        rag_chain = (
            RunnableParallel(
                context=retriever | format_docs,
                question=RunnablePassthrough()
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain, "✅ RAG chain initialized successfully!"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# Initialize chain if API key provided
if groq_api_key:
    if st.session_state.rag_chain is None:
        with st.spinner("🔧 Initializing MedBRAIN AI..."):
            chain, status = init_rag_chain(groq_api_key, model_name, temperature)
            st.session_state.rag_chain = chain
            if chain:
                st.success(status)
            else:
                st.error(status)
else:
    st.warning("⚠️ Please enter your Groq API key in the sidebar to start.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about medical topics (for education only)..."):
    if not groq_api_key:
        st.error("Please provide Groq API key in the sidebar!")
    elif st.session_state.rag_chain is None:
        st.error("RAG chain not initialized. Check your API key and try again.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🧠 MedBRAIN AI is thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
