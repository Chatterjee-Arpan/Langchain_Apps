# Langchain_Apps
End-to-end LangChain projects with RAG pipelines, vector databases (Pinecone, FAISS, Astra DB), and interactive Streamlit UIs.

This repo currently includes two demo apps:  

- **Celebrity Finder** → Multi-step LLMChain pipeline to fetch celebrity details, date of birth, and major world events around that time.  
- **Conversational Q&A Chatbot** → A memory-enabled chatbot that acts as a helpful fact-checking assistant.
- **PDF Reader with RAG & Astra DB** → A *Retrieval-Augmented Generation (RAG)* pipeline built with LangChain to query PDFs.

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

### 2. Conversational Q&A Chatbot (`Chatbot.py`)
- A **fact-checking conversational assistant** powered by `ChatOpenAI (gpt-3.5-turbo)`.
- Maintains **conversation flow** using LangChain `SystemMessage`, `HumanMessage`, and `AIMessage`.
- Session memory is handled with `st.session_state` so the chatbot remembers the full dialogue.
- Streamlit UI provides a clean text input box and button to submit questions interactively.

**Run it:**
```bash
streamlit run Simple_Chatbot.py

### 3. PDF Reader with RAG & Astra DB (`Pdf_Reader_app_RAG_Astra_DB.ipynb`)
- A **Retrieval-Augmented Generation (RAG) pipeline** built with LangChain to query PDFs.  
- Loads PDF content using **PyPDF2**, extracts text, and embeds it with **OpenAI embeddings**.  
- Stores and retrieves embeddings via **Astra DB (Cassandra-based vector store)**.  
- Integrates with `ChatOpenAI (gpt-3.5-turbo)` for answering natural language queries over the document.  
- Demonstrates a scalable pipeline for document Q&A powered by vector databases.  

**Run it:**
```bash
jupyter notebook Pdf_Reader_app_RAG_Astra_DB.ipynb

### Future Work

- Add more LangChain apps using RAG pipelines and vector databases (Pinecone, FAISS, Astra DB).
- Extend Streamlit dashboards for better visualization.
- Include support for multiple LLM providers.
