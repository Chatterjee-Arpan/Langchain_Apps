# Langchain_Apps
End-to-end LangChain projects with RAG pipelines, vector databases (Pinecone, FAISS, Astra DB), and interactive Streamlit UIs.

This repo currently includes two demo apps:  

- **Celebrity Finder** → Multi-step LLMChain pipeline to fetch celebrity details, date of birth, and major world events around that time.  
- **Conversational Q&A Chatbot** → A memory-enabled chatbot that acts as a helpful fact-checking assistant.
- **PDF Reader with RAG & Astra DB** → A *Retrieval-Augmented Generation (RAG)* pipeline built with LangChain to query PDFs.
- **Q&A over Documents with Pinecone** → A *document question-answering pipeline* using LangChain, OpenAI embeddings, and *Pinecone vector database*.
- **Querying PDFs with FAISS** → A *PDF-based Q&A pipeline* built with LangChain, OpenAI embeddings, and *FAISS* as the vector store.

All apps feature **Streamlit UIs** for easy interaction.  

---

## 🚀 Applications  

### 1. Celebrity Finder (`Celeb_finder.py`)  
- Uses **LangChain Sequential Chains** with memory buffers.  
- Workflow:  
  1. Ask about a celebrity.  
  2. Extract their date of birth.  
  3. Retrieve 5 major world events around that time.  
- Built with `ChatOpenAI (gpt-3.5-turbo)`.  
- Streamlit expands into sections to show chat history and major events.  

**Run it:**  
```bash
streamlit run Celeb_finder.py
```

### 2. Conversational Q&A Chatbot (`Simple_Chatbot.py`)
- A **fact-checking conversational assistant** powered by `ChatOpenAI (gpt-3.5-turbo)`.
- Maintains **conversation flow** using LangChain `SystemMessage`, `HumanMessage`, and `AIMessage`.
- Session memory is handled with `st.session_state` so the chatbot remembers the full dialogue.
- Streamlit UI provides a clean text input box and button to submit questions interactively.

**Run it:**
```bash
streamlit run Simple_Chatbot.py
```

### 3. PDF Reader with RAG & Astra DB (`Pdf_Reader_app_RAG_Astra_DB.ipynb`)
- A **Retrieval-Augmented Generation (RAG) pipeline** built with LangChain to query PDFs.  
- Loads PDF content using **PyPDF2**, extracts text, and embeds it with **OpenAI embeddings**.  
- Stores and retrieves embeddings via **Astra DB (Cassandra-based vector store)**.  
- Integrates with `ChatOpenAI (gpt-3.5-turbo)` for answering natural language queries over the document.  
- Demonstrates a scalable pipeline for document Q&A powered by vector databases.  

**Run it:**
```bash
jupyter notebook Pdf_Reader_app_RAG_Astra_DB.ipynb
```
### 4. Q&A over Documents with Pinecone (`Q_nd_A_app_Pinecone.ipynb`)
- A **document question-answering pipeline** using LangChain, OpenAI embeddings, and **Pinecone vector database**.  
- Loads PDFs from a directory with `PyPDFDirectoryLoader`, then splits them into chunks via **RecursiveCharacterTextSplitter**.  
- Embeds text chunks using **OpenAI embeddings** and stores them in **Pinecone** for semantic search.  
- Queries are matched against the vector store, and results are fed into `ChatOpenAI` to generate precise answers.  
- Demonstrates a scalable RAG workflow for **multi-document semantic search and Q&A**.  

**Run it:**
```bash
jupyter notebook Q_nd_A_app_Pinecone.ipynb
```

### 5. Querying PDFs with FAISS (`querying_pdfs.ipynb`)
- A **PDF-based Q&A pipeline** built with LangChain, OpenAI embeddings, and **FAISS** as the vector store.  
- Loads PDF documents with **PyPDF2** and extracts the raw text.  
- Splits text into manageable chunks using **CharacterTextSplitter** to ensure efficient token usage.  
- Creates embeddings via **OpenAI embeddings** and stores them in a **FAISS index** for fast similarity search.  
- Enables natural language queries over the PDF content by retrieving relevant chunks and passing them to the LLM.  
- Demonstrates a lightweight local RAG workflow without external database dependencies.  

**Run it:**
```bash
jupyter notebook querying_pdfs.ipynb
```

### Future Work

- Add more LangChain apps using RAG pipelines and vector databases (Pinecone, FAISS, Astra DB).
- Extend Streamlit dashboards for better visualization.
- Include support for multiple LLM providers.
