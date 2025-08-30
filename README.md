# DocuChat AI - Multi-Document RAG Chatbot (https://amit9785-multiple-documnt-rag-system-main-zn3jjy.streamlit.app/)


A powerful Retrieval-Augmented Generation (RAG) chatbot that allows you to upload and query multiple documents (PDF, DOCX, PowerPoint, TXT) using natural language. Built with modern AI technologies including LangChain, FAISS vector search, and Streamlit for an intuitive web interface.

![SYSTEM ARCHITECT OF DOCUCHAT AI](https://github.com/user-attachments/assets/9942ef46-e5b3-4157-8e64-1e10511a12ab)

## 🚀 Features

- **Multi-Format Document Support**: Upload PDF, DOCX, PPTX, and TXT files
- **Intelligent Text Processing**: Automatic text extraction and chunking
- **Semantic Search**: FAISS vector database for efficient similarity search
- **Interactive Chat Interface**: Real-time conversation with document context
- **Conversation History**: Maintains chat history throughout the session
- **Modern AI Stack**: Built with LangChain, Groq LLM, and Sentence Transformers

## 🏗️ Architecture

The system follows a modern RAG architecture:

1. **Document Processing**: Unstructured library extracts text from various formats
2. **Text Chunking**: RecursiveCharacterTextSplitter with token-based splitting
3. **Vector Embeddings**: Sentence Transformers for semantic representation
4. **Vector Storage**: FAISS for efficient similarity search
5. **RAG Chain**: LangChain-based retrieval and generation pipeline
6. **Web Interface**: Streamlit for user interaction

## 📋 Prerequisites

- Python 3.8 or higher
- Groq API key (free tier available)
- Internet connection for model downloads

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd RAG-MultiDoc-Chatbot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   GORQ_API=your_groq_api_key_here
   ```
   
   Get your free Groq API key from: https://console.groq.com/

## 🚀 Usage

1. **Start the application**
   ```bash
   streamlit run main.py
   ```

2. **Access the web interface**
   - Open your browser and go to `http://localhost:8501`
   - The application will load with a clean chat interface

3. **Upload documents**
   - Use the sidebar file uploader
   - Select multiple files (PDF, DOCX, PPTX, TXT)
   - Wait for the success message

4. **Start chatting**
   - Type your questions in the chat input
   - The AI will search through your documents and provide relevant answers
   - Conversation history is maintained throughout the session

## 🔧 Configuration

### Model Settings

The application uses the following models by default:

- **LLM**: Groq's `openai/gpt-oss-20b` (fast and cost-effective)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local)
- **Text Chunking**: 1000 tokens with 300 token overlap

### Customization

You can modify these settings in `main.py`:

```python
# Change LLM model
llm = ChatGroq(
    model="llama3-8b-8192",  # Alternative model
    temperature=0.0,
    api_key=os.getenv("GORQ_API")
)

# Change embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  # Alternative model
    model_kwargs={'device': 'cpu'}
)

# Adjust chunking parameters
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1500,    # Larger chunks
    chunk_overlap=400,  # More overlap
)
```


These documents provide rich content for testing the RAG capabilities.

## 🔍 How It Works

### 1. Document Processing
```python
def load_documents(file_paths):
    # Uses Unstructured library to extract text from various formats
    elements = partition(filename=file)
    text_elements = [element.text for element in elements]
```

### 2. Text Chunking
```python
def split_text(text: str):
    # Splits text into manageable chunks with overlap
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=300,
    )
```

### 3. Vector Embeddings
```python
def get_vectorstore(chunks):
    # Creates semantic embeddings and stores in FAISS
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
```

### 4. RAG Chain
```python
def rag_chain(vectorstore, question):
    # Retrieves relevant chunks and generates answers
    qa_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
```



### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **LangChain**: For the RAG framework
- **Groq**: For fast and cost-effective LLM access
- **Hugging Face**: For embedding models
- **Streamlit**: For the web interface
- **FAISS**: For vector similarity search

## Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the code comments for implementation details
3. Open an issue on GitHub

**Made with ❤️ using modern AI technologies**
