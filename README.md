---
title: Conversational RAG With PDF
emoji: 📄
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.43.2
app_file: app.py
pinned: false
---

# 📄 Conversational RAG With PDF Uploads and Chat History

> Upload any PDF and chat with its content using AI — powered by **Groq LLM**, **LangChain RAG**, and **HuggingFace Embeddings**. Completely free to use!

---

## 📌 What Does This App Do?

This app lets you:
1. 📤 **Upload one or multiple PDF files**
2. 💬 **Ask questions** about the content of those PDFs
3. 🧠 **Maintains chat history** — the AI remembers previous questions in the same session
4. 🔁 **Multiple sessions** — switch between different session IDs to keep conversations separate

---

## 🛠️ How It Works

```
User uploads PDF(s)
        │
        ▼
PyPDFLoader loads the PDF content
        │
        ▼
RecursiveCharacterTextSplitter splits into chunks
        │
        ▼
HuggingFace Embeddings converts chunks to vectors
        │
        ▼
ChromaDB stores vectors as a retriever
        │
        ▼
User asks a question
        │
        ▼
History-Aware Retriever reformulates question using chat history
        │
        ▼
RAG Chain retrieves relevant chunks + generates answer
        │
        ▼
Groq LLM (llama-3.1-8b-instant) returns the final answer
```

---

## ⚙️ Tech Stack

| Technology | Purpose | Cost |
|------------|---------|------|
| [Groq](https://console.groq.com) | LLM inference (llama-3.1-8b-instant) | Free |
| [LangChain 1.x](https://python.langchain.com) | RAG pipeline and chat history | Free |
| [HuggingFace Embeddings](https://huggingface.co) | Convert text to vectors locally | Free |
| [ChromaDB](https://www.trychroma.com) | In-memory vector store | Free |
| [Streamlit](https://streamlit.io) | Web UI | Free |
| [PyPDFLoader](https://pypi.org/project/pypdf/) | Load and parse PDF files | Free |

> ✅ **No OpenAI API key required!**

---

## 🚀 How to Use

### On HuggingFace Spaces:
1. Open the app
2. Enter your **free Groq API key** ([get one here](https://console.groq.com))
3. Enter a **Session ID** (default is `default_session`)
4. Upload one or more **PDF files**
5. Type your question and get answers!

### Run Locally:

**1. Clone the repo:**
```bash
git clone https://huggingface.co/spaces/dev-2106/conversational-rag-pdf
cd conversational-rag-pdf
```

**2. Install dependencies:**
```bash
pip install streamlit langchain langchain-groq langchain-chroma langchain-huggingface langchain-community langchain-core pypdf chromadb
```

**3. Create `.env` file:**
```env
HF_TOKEN=your-huggingface-token
```

**4. Run the app:**
```bash
streamlit run app.py
```

---

## 🔑 API Keys Required

| Key | Where to Get | Cost |
|-----|-------------|------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | Free |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Free |

---

## 💡 Features

- ✅ Upload **multiple PDFs** at once
- ✅ **Session-based chat history** — each session ID maintains its own history
- ✅ **Context-aware questions** — the AI reformulates questions based on previous chat
- ✅ **No OpenAI needed** — uses Groq (free) + HuggingFace embeddings (local)
- ✅ **Fast responses** — Groq runs llama-3.1-8b-instant at blazing speed

---

## 📋 Requirements

```txt
streamlit
langchain
langchain-groq
langchain-chroma
langchain-huggingface
langchain-community
langchain-core
langchain-text-splitters
pypdf
chromadb
python-dotenv
```

---

## 🐛 Common Issues & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: langchain.chains` | Old LangChain version | Upgrade to `langchain>=1.0.0` |
| `No module named 'langchain_classic'` | Wrong import | Use `from langchain.chains import ...` |
| `ChromaDB PanicException` | Corrupted ChromaDB files | Run `rm -rf ~/.local/share/chroma` |
| `GROQ 429 rate limit` | Too many requests | Wait 60 seconds and retry |
| `PDF not loading` | File saved to same temp path | Each PDF now gets a unique temp filename |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">Built with ❤️ using LangChain, Groq and Streamlit</p>
