# 🧠 MedBRAIN AI

> **Powerful Medical Assistant Chatbot** | RAG-based Medical Encyclopedia Q&A using LangChain + Groq + Open Source

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

MedBRAIN AI is an educational medical assistant chatbot that uses **Retrieval-Augmented Generation (RAG)** to provide accurate medical information from the Gale Encyclopedia of Medicine (3rd Edition). Built entirely with **open-source** technologies and powered by **Groq's fast inference**, this chatbot demonstrates production-ready RAG architecture.

### ⚠️ Medical Disclaimer

**MedBRAIN AI is for educational purposes ONLY**. It does NOT provide medical advice, diagnoses, or treatment. Always consult licensed healthcare professionals for medical decisions.

## ✨ Features

- 📚 **Medical Encyclopedia RAG**: Retrieves relevant information from 3rd Edition Gale Medical Encyclopedia
- ⚡ **Lightning Fast**: Powered by Groq's LLM inference (llama-3.1-8b-instant)
- 🎨 **Beautiful UI**: Modern Streamlit interface with gradient design
- 🔒 **Privacy-First**: Runs locally, no data sent to third parties
- 💰 **100% Free**: Uses open-source models and free Groq API tier
- 🛡️ **Safety Built-In**: Strong medical disclaimers and educational focus

## 🛠️ Tech Stack

- **LLM**: Groq (llama-3.1-8b-instant)
- **Framework**: LangChain
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector DB**: Chroma (local)
- **UI**: Streamlit
- **Data**: Gale Medical Encyclopedia (3rd Edition, 64.4 MB PDF)

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Groq API Key (free from [console.groq.com](https://console.groq.com/keys))

### Installation

```bash
# Clone the repository
git clone https://github.com/amalsp220/MedBRAIN-AI.git
cd MedBRAIN-AI

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Download Medical Encyclopedia PDF

Download the Gale Medical Encyclopedia PDF from Hugging Face:

```bash
mkdir data
# Download from: https://huggingface.co/datasets/amalsp/MEDICAL_ENCYCLOPEDIA/blob/main/The-Gale-Encyclopedia-of-Medicine-3rd-Edition-staibabussalamsula.ac_.id_.pdf
# Save as: data/gale_medical_encyclopedia.pdf
```

### Ingest PDF & Create Vector Database

```bash
python ingest_pdf.py
```

This will:
- Load the PDF
- Split into chunks
- Create embeddings
- Store in Chroma vector database (`chroma_db/`)

### Run the Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

## 📚 Project Structure

```
MedBRAIN-AI/
├── app.py                  # Streamlit UI application
├── ingest_pdf.py           # PDF ingestion script
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
├── data/
│   └── gale_medical_encyclopedia.pdf
├── chroma_db/              # Vector database (generated)
├── README.md
└── LICENSE
```

## 🔧 Configuration

Edit `.env` file:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key: [https://console.groq.com/keys](https://console.groq.com/keys)

## 💡 Usage Examples

Ask MedBRAIN AI questions like:

- "What is diabetes and what are its symptoms?"
- "Explain MRI procedure to a layperson"
- "What are the risk factors for hypertension?"
- "How does asthma affect the respiratory system?"

## 🏆 Resume/Portfolio Highlights

This project demonstrates:

✅ **Production-Ready RAG Architecture** using LangChain  
✅ **Vector Database Implementation** with Chroma  
✅ **LLM Integration** with Groq's high-performance API  
✅ **Open-Source Embeddings** (sentence-transformers)  
✅ **Professional UI/UX** with Streamlit  
✅ **Secure Environment Management** with python-dotenv  
✅ **Medical Domain Application** with safety guardrails  

## 🚀 Deployment

### Streamlit Cloud

1. Push to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add GROQ_API_KEY to Secrets
4. Deploy!

### Railway / Render

1. Add `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT
   ```
2. Add environment variables
3. Deploy

## 📝 License

MIT License - feel free to use for learning and portfolio projects!

## 👤 Author

**Amal SP**  
- GitHub: [@amalsp220](https://github.com/amalsp220)

## ⭐ Star This Repo

If this project helps you, give it a ⭐!

---

**Built with ❤️ using open-source technologies**
