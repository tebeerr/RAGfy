# import streamlit as st
# import os
# import tempfile
# import json
# import logging
# from dotenv import load_dotenv
# from langchain_community.vectorstores import FAISS
# # Fixed import - use the new package
# try:
#     from langchain_huggingface import HuggingFaceEmbeddings
# except ImportError:
#     from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# from langchain_groq import ChatGroq
# from langchain.prompts import PromptTemplate
# import warnings
# from functools import lru_cache
# import time

# # Load environment variables
# load_dotenv()

# # Suppress deprecation warnings to reduce noise
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(_name_)

# # Page configuration
# st.set_page_config(
#     page_title="PDF Chatbot",
#     page_icon="📚",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Initialize session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "vectorstore" not in st.session_state:
#     st.session_state.vectorstore = None
# if "embedding_model" not in st.session_state:
#     st.session_state.embedding_model = None

# # Initialize embedding model once at startup
# @lru_cache(maxsize=1)
# def get_embedding_model():
#     """Get or create the embedding model (cached)"""
#     if st.session_state.embedding_model is None:
#         with st.spinner("Initializing embedding model..."):
#             logger.info("Initializing embedding model...")
#             st.session_state.embedding_model = HuggingFaceEmbeddings(
#                 model_name="sentence-transformers/all-MiniLM-L6-v2",
#                 model_kwargs={'device': 'cpu'},
#                 encode_kwargs={'normalize_embeddings': True}
#             )
#             logger.info("Embedding model initialized")
#     return st.session_state.embedding_model

# def process_single_pdf(pdf_file, temp_dir):
#     """Process a single PDF file efficiently"""
#     try:
#         temp_path = os.path.join(temp_dir, f"temp_{pdf_file.name}")
#         with open(temp_path, "wb") as f:
#             f.write(pdf_file.read())
        
#         loader = PyPDFLoader(temp_path)
#         pages = loader.load()
        
#         # Clean up immediately after loading
#         try:
#             os.remove(temp_path)
#         except:
#             pass
            
#         logger.info(f"Processed {pdf_file.name}: {len(pages)} pages")
#         return pages
#     except Exception as e:
#         logger.error(f"Error processing {pdf_file.name}: {e}")
#         st.error(f"Error processing {pdf_file.name}: {e}")
#         return []

# def create_vectorstore_optimized(all_pages):
#     """Create vectorstore with optimizations"""
#     try:
#         # Use smaller chunks for faster processing
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=500,  # Reduced chunk size for faster processing
#             chunk_overlap=50,  # Reduced overlap
#             separators=["\n\n", "\n", ". ", " ", ""]
#         )
#         docs = text_splitter.split_documents(all_pages)
        
#         # Limit number of chunks to prevent slowdown
#         if len(docs) > 100:  # Limit chunks for faster processing
#             docs = docs[:100]
#             logger.info(f"Limited to first 100 chunks for performance")
        
#         embedding_function = get_embedding_model()
#         vectorstore = FAISS.from_documents(docs, embedding_function)
#         logger.info(f"Created vectorstore with {len(docs)} document chunks")
#         return vectorstore
#     except Exception as e:
#         logger.error(f"Error creating vectorstore: {e}")
#         st.error(f"Error creating vectorstore: {e}")
#         return None

# def process_pdfs(pdf_files):
#     """Process uploaded PDF files"""
#     if not pdf_files:
#         return None
    
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     try:
#         # Limit number of PDFs for performance
#         if len(pdf_files) > 3:
#             pdf_files = pdf_files[:3]
#             st.warning("Limited to first 3 PDFs for performance")
        
#         status_text.text(f"Processing {len(pdf_files)} PDF files...")
        
#         # Create temporary directory
#         with tempfile.TemporaryDirectory() as temp_dir:
#             all_pages = []
            
#             for i, pdf in enumerate(pdf_files):
#                 if pdf is not None:
#                     status_text.text(f"Processing {pdf.name}...")
#                     progress_bar.progress((i + 1) / len(pdf_files))
                    
#                     pages = process_single_pdf(pdf, temp_dir)
#                     all_pages.extend(pages)
                    
#                     # Limit total pages for performance
#                     if len(all_pages) > 50:  # Limit total pages
#                         all_pages = all_pages[:50]
#                         st.warning("Limited to first 50 pages for performance")
#                         break
            
#             if all_pages:
#                 status_text.text("Creating vector store...")
#                 logger.info(f"Total pages loaded: {len(all_pages)}")
#                 vectorstore = create_vectorstore_optimized(all_pages)
                
#                 progress_bar.progress(1.0)
#                 status_text.text("✅ PDFs processed successfully!")
#                 time.sleep(1)
#                 status_text.empty()
#                 progress_bar.empty()
                
#                 return vectorstore
#             else:
#                 status_text.text("❌ No pages could be processed")
#                 return None
                
#     except Exception as e:
#         logger.error(f"Error processing PDFs: {e}")
#         st.error(f"Error processing PDFs: {e}")
#         return None

# def get_chat_response(message, vectorstore=None, temperature=0.7):
#     """Generate chat response"""
#     # Check for required environment variables
#     groq_api_key = os.environ.get('GROQ_API_KEY')
#     if not groq_api_key:
#         st.error("GROQ_API_KEY not found in environment variables")
#         return None, []
        
#     model_name = os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')
    
#     # Initialize LLM with faster settings
#     try:
#         groq_llm = ChatGroq(
#             groq_api_key=groq_api_key,
#             model_name=model_name, 
#             temperature=temperature,
#             max_tokens=1024,  # Reduced from 2048 for faster response
#             timeout=20  # Reduced timeout
#         )
#         logger.info(f"Initialized Groq LLM with model: {model_name}")
#     except Exception as e:
#         logger.error(f"Error initializing Groq LLM: {e}")
#         st.error(f"Error initializing Groq LLM: {e}")
#         return None, []
    
#     # Generate response
#     if not vectorstore:
#         # Chat without documents
#         try:
#             conversation_messages = []
#             # Add recent messages from session state (last 5)
#             for msg in st.session_state.messages[-5:]:
#                 if msg["role"] in ["user", "assistant"]:
#                     conversation_messages.append({
#                         "role": msg["role"],
#                         "content": msg["content"][:500]
#                     })
            
#             conversation_messages.append({"role": "user", "content": message})
#             response = groq_llm.invoke(conversation_messages)
#             answer = response.content if hasattr(response, 'content') else str(response)
#             sources = []
#             logger.info("Generated response without documents")
#             return answer, sources
#         except Exception as e:
#             logger.error(f"Error in chat completion: {e}")
#             st.error(f"Error in chat completion: {e}")
#             return None, []
#     else:
#         # Use RetrievalQA with documents
#         try:
#             # Simplified context window
#             context_window = "\n".join([
#                 f"{m['role']}: {m['content'][:200]}"
#                 for m in st.session_state.messages[-3:]  # Only last 3 messages
#                 if m["role"] in ["user", "assistant"]
#             ])
            
#             # Simplified prompt template
#             prompt_template = PromptTemplate(
#                 template=(
#                     "Use the document context and conversation history to answer the question.\n\n"
#                     f"Recent conversation:\n{context_window}\n\n"
#                     "Context: {context}\n\n"
#                     "Question: {question}\n\n"
#                     "Answer:"
#                 ),
#                 input_variables=["context", "question"]
#             )
            
#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=groq_llm,
#                 chain_type="stuff",
#                 retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # Reduced from 5
#                 return_source_documents=True,
#                 chain_type_kwargs={"prompt": prompt_template}
#             )
            
#             response = qa_chain.invoke({"query": message})
#             answer = response.get('result', 'No answer generated')
            
#             # Extract sources with better formatting
#             source_docs = response.get('source_documents', [])
#             sources = []
#             for i, doc in enumerate(source_docs[:2]):  # Reduced from 3 to 2
#                 content = doc.page_content.strip()
#                 if len(content) > 100:  # Reduced from 150
#                     content = content[:100] + "..."
#                 sources.append(f"Source {i+1}: {content}")
            
#             logger.info("Generated response with document context")
#             return answer, sources
                
#         except Exception as e:
#             logger.error(f"Error in retrieval QA: {e}")
#             st.error(f"Error in retrieval QA: {e}")
#             return None, []

# def main():
#     """Main Streamlit app"""
#     st.title("📚 PDF Chatbot")
#     st.write("Upload PDF files and chat with their content using AI")
    
#     # Sidebar for file upload and settings
#     with st.sidebar:
#         st.header("📁 Upload PDFs")
#         uploaded_files = st.file_uploader(
#             "Choose PDF files",
#             type="pdf",
#             accept_multiple_files=True,
#             help="Upload up to 3 PDF files for optimal performance"
#         )
        
#         if uploaded_files:
#             if st.button("Process PDFs", type="primary"):
#                 with st.spinner("Processing PDFs..."):
#                     st.session_state.vectorstore = process_pdfs(uploaded_files)
#                     if st.session_state.vectorstore:
#                         st.success(f"✅ Processed {len(uploaded_files)} PDF(s) successfully!")
#                     else:
#                         st.error("❌ Failed to process PDFs")
        
#         st.header("⚙️ Settings")
#         temperature = st.slider(
#             "Temperature",
#             min_value=0.0,
#             max_value=1.0,
#             value=0.7,
#             step=0.1,
#             help="Higher values make the output more random"
#         )
        
#         # Model selection
#         model_options = [
#             'llama-3.3-70b-versatile',
#             'mixtral-8x7b-32768',
#             'gemma2-9b-it'
#         ]
#         selected_model = st.selectbox(
#             "Model",
#             model_options,
#             index=0,
#             help="Choose the AI model to use"
#         )
        
#         # Set model in environment
#         os.environ['GROQ_MODEL'] = selected_model
        
#         # Status
#         if st.session_state.vectorstore:
#             st.success("📄 PDFs loaded and ready")
#         else:
#             st.info("💡 Upload PDFs to chat with documents")
        
#         # Clear chat button
#         if st.button("🗑️ Clear Chat"):
#             st.session_state.messages = []
#             st.rerun()
    
#     # Main chat interface
#     st.header("💬 Chat")
    
#     # Display chat messages
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
            
#             # Show sources if available
#             if message.get("sources"):
#                 with st.expander("📖 Sources"):
#                     for source in message["sources"]:
#                         st.text(source)
    
#     # Chat input
#     if prompt := st.chat_input("Ask a question about your PDFs or chat generally..."):
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         # Display user message
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         # Generate and display assistant response
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 response, sources = get_chat_response(
#                     prompt, 
#                     st.session_state.vectorstore, 
#                     temperature
#                 )
            
#             if response:
#                 st.markdown(response)
                
#                 # Show sources if available
#                 if sources:
#                     with st.expander("📖 Sources"):
#                         for source in sources:
#                             st.text(source)
                
#                 # Add assistant response to chat history
#                 st.session_state.messages.append({
#                     "role": "assistant", 
#                     "content": response,
#                     "sources": sources
#                 })
#             else:
#                 st.error("Failed to generate response. Please try again.")
    
#     # Footer
#     st.markdown("---")
#     st.markdown("Powered by Groq, LangChain, and Streamlit")

# if _name_ == "_main_":
#     # Initialize embedding model at startup
#     logger.info("Pre-loading embedding model...")
#     get_embedding_model()
    
#     # Check for required environment variables
#     if not os.environ.get('GROQ_API_KEY'):
#         st.error("⚠️ GROQ_API_KEY not found in environment variables")
#         st.info("Please set your GROQ_API_KEY in the .env file")
#         st.stop()
    
#     main()

import streamlit as st
import os
import tempfile
import json
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
# Fixed import - use the new package
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import warnings
from functools import lru_cache
import time

# Load environment variables
load_dotenv()

# Suppress deprecation warnings to reduce noise
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "debug_info" not in st.session_state:
    st.session_state.debug_info = []
if "document_count" not in st.session_state:
    st.session_state.document_count = 0

def get_embedding_model():
    """Get or create the embedding model"""
    try:
        if st.session_state.embedding_model is None:
            st.session_state.debug_info.append("🔧 Initializing embedding model...")
            
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            st.session_state.embedding_model = embedding_model
            st.session_state.debug_info.append("✅ Embedding model initialized successfully")
        
        return st.session_state.embedding_model
    except Exception as e:
        error_msg = f"❌ Error initializing embedding model: {e}"
        st.session_state.debug_info.append(error_msg)
        st.error(error_msg)
        return None

def process_single_pdf(pdf_file, temp_dir):
    """Process a single PDF file with better error handling"""
    try:
        st.session_state.debug_info.append(f"📄 Processing PDF: {pdf_file.name}")
        
        # Reset file pointer to beginning
        pdf_file.seek(0)
        
        # Save uploaded file to temp directory
        temp_path = os.path.join(temp_dir, f"temp_{pdf_file.name}")
        with open(temp_path, "wb") as f:
            f.write(pdf_file.read())
        
        st.session_state.debug_info.append(f"💾 Saved to: {temp_path}")
        
        # Check if file was saved properly
        if not os.path.exists(temp_path):
            st.session_state.debug_info.append("❌ File was not saved properly")
            return []
        
        file_size = os.path.getsize(temp_path)
        st.session_state.debug_info.append(f"📊 File size: {file_size} bytes")
        
        # Load PDF with better error handling
        try:
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
        except Exception as pdf_error:
            st.session_state.debug_info.append(f"❌ PyPDFLoader failed: {pdf_error}")
            # Try alternative loader
            try:
                from langchain_community.document_loaders import UnstructuredPDFLoader
                loader = UnstructuredPDFLoader(temp_path)
                pages = loader.load()
                st.session_state.debug_info.append("✅ Used UnstructuredPDFLoader as fallback")
            except Exception as alt_error:
                st.session_state.debug_info.append(f"❌ Alternative loader also failed: {alt_error}")
                return []
        
        # Debug: Show extracted content
        if pages:
            total_chars = sum(len(page.page_content) for page in pages)
            st.session_state.debug_info.append(f"📖 Extracted {len(pages)} pages, {total_chars} characters")
            
            # Show first 200 characters of first page for debugging
            if pages[0].page_content and len(pages[0].page_content.strip()) > 0:
                preview = pages[0].page_content[:200].replace('\n', ' ').strip()
                st.session_state.debug_info.append(f"📝 First page preview: {preview}...")
            else:
                st.session_state.debug_info.append("⚠️ First page appears to be empty")
                
            # Filter out empty pages
            pages = [page for page in pages if page.page_content.strip()]
            st.session_state.debug_info.append(f"📄 After filtering empty pages: {len(pages)} pages")
        else:
            st.session_state.debug_info.append("❌ No pages extracted from PDF")
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
            
        return pages
    except Exception as e:
        error_msg = f"❌ Error processing {pdf_file.name}: {e}"
        st.session_state.debug_info.append(error_msg)
        st.error(error_msg)
        return []

def create_vectorstore_optimized(all_pages):
    """Create vectorstore with better validation"""
    try:
        st.session_state.debug_info.append("🔨 Creating vector store...")
        
        if not all_pages:
            st.session_state.debug_info.append("❌ No pages provided for vector store creation")
            return None
        
        # Text splitting with better settings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased for better context
            chunk_overlap=100,  # Increased overlap
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        docs = text_splitter.split_documents(all_pages)
        
        # Filter out very short chunks
        docs = [doc for doc in docs if len(doc.page_content.strip()) > 50]
        
        st.session_state.debug_info.append(f"✂️ Created {len(docs)} meaningful chunks")
        
        if not docs:
            st.session_state.debug_info.append("❌ No meaningful chunks created")
            return None
        
        # Show sample chunks for debugging
        if docs:
            for i, doc in enumerate(docs[:3]):  # Show first 3 chunks
                sample = doc.page_content[:100].replace('\n', ' ').strip()
                st.session_state.debug_info.append(f"📄 Chunk {i+1}: {sample}...")
        
        # Get embedding model
        embedding_function = get_embedding_model()
        if not embedding_function:
            st.session_state.debug_info.append("❌ Failed to get embedding model")
            return None
        
        # Create vector store
        st.session_state.debug_info.append("🧮 Creating FAISS vector store...")
        vectorstore = FAISS.from_documents(docs, embedding_function)
        
        # Test the vector store
        test_query = "test"
        test_results = vectorstore.similarity_search(test_query, k=1)
        st.session_state.debug_info.append(f"🧪 Vector store test: {len(test_results)} results")
        
        st.session_state.debug_info.append(f"✅ Vector store created successfully with {len(docs)} chunks")
        st.session_state.document_count = len(docs)
        
        return vectorstore
    except Exception as e:
        error_msg = f"❌ Error creating vectorstore: {e}"
        st.session_state.debug_info.append(error_msg)
        st.error(error_msg)
        return None

def process_pdfs(pdf_files):
    """Process uploaded PDF files with comprehensive error handling"""
    if not pdf_files:
        st.session_state.debug_info.append("❌ No PDF files provided")
        return None
    
    # Clear previous debug info
    st.session_state.debug_info = []
    st.session_state.document_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        st.session_state.debug_info.append(f"🚀 Starting to process {len(pdf_files)} PDF files...")
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            all_pages = []
            
            for i, pdf in enumerate(pdf_files):
                if pdf is not None:
                    status_text.text(f"Processing {pdf.name}...")
                    progress_bar.progress((i + 1) / len(pdf_files))
                    
                    pages = process_single_pdf(pdf, temp_dir)
                    all_pages.extend(pages)
            
            st.session_state.debug_info.append(f"📚 Total pages collected: {len(all_pages)}")
            
            if all_pages:
                status_text.text("Creating vector store...")
                vectorstore = create_vectorstore_optimized(all_pages)
                
                if vectorstore:
                    progress_bar.progress(1.0)
                    status_text.text("✅ PDFs processed successfully!")
                    st.session_state.pdf_processed = True
                    st.session_state.debug_info.append("🎉 PDF processing completed successfully!")
                    
                    time.sleep(1)
                    status_text.empty()
                    progress_bar.empty()
                    
                    return vectorstore
                else:
                    st.session_state.debug_info.append("❌ Failed to create vector store")
                    status_text.text("❌ Failed to create vector store")
                    return None
            else:
                st.session_state.debug_info.append("❌ No content extracted from any PDF")
                status_text.text("❌ No content could be extracted")
                return None
                
    except Exception as e:
        error_msg = f"❌ Error in PDF processing pipeline: {e}"
        st.session_state.debug_info.append(error_msg)
        st.error(error_msg)
        return None

def get_chat_response(message, vectorstore=None, temperature=0.7):
    """Generate chat response with proper document context checking"""
    # Check for required environment variables
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables")
        return None, []
        
    model_name = os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')
    
    # Initialize LLM
    try:
        groq_llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name, 
            temperature=temperature,
            max_tokens=1024,
            timeout=30
        )
    except Exception as e:
        error_msg = f"❌ Error initializing Groq LLM: {e}"
        st.session_state.debug_info.append(error_msg)
        st.error(error_msg)
        return None, []
    
    # Check if we have a valid vector store
    if vectorstore is None:
        st.session_state.debug_info.append("💬 No vector store available - using general chat")
        try:
            response = groq_llm.invoke([{"role": "user", "content": message}])
            answer = response.content if hasattr(response, 'content') else str(response)
            return answer, []
        except Exception as e:
            error_msg = f"❌ Error in general chat: {e}"
            st.session_state.debug_info.append(error_msg)
            return None, []
    
    # Use document-based chat
    try:
        st.session_state.debug_info.append(f"🤖 Using documents to answer: {message[:50]}...")
        
        # Test retrieval first
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Get more documents
        )
        
        retrieved_docs = retriever.get_relevant_documents(message)
        st.session_state.debug_info.append(f"🔍 Retrieved {len(retrieved_docs)} relevant documents")
        
        if not retrieved_docs:
            st.session_state.debug_info.append("⚠️ No relevant documents found - falling back to general chat")
            response = groq_llm.invoke([{"role": "user", "content": message}])
            answer = response.content if hasattr(response, 'content') else str(response)
            return answer, []
        
        # Show what was retrieved for debugging
        for i, doc in enumerate(retrieved_docs[:3]):
            preview = doc.page_content[:150].replace('\n', ' ').strip()
            st.session_state.debug_info.append(f"📄 Retrieved doc {i+1}: {preview}...")
        
        # Create a more explicit prompt
        prompt_template = PromptTemplate(
            template=(
                "You are an AI assistant that answers questions based on provided documents. "
                "Use ONLY the information from the documents below to answer the question. "
                "If the answer is not in the documents, say 'I cannot find that information in the provided documents.'\n\n"
                "DOCUMENTS:\n{context}\n\n"
                "QUESTION: {question}\n\n"
                "ANSWER (based only on the documents above):"
            ),
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=groq_llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        
        response = qa_chain.invoke({"query": message})
        answer = response.get('result', 'No answer generated')
        
        # Format sources
        source_docs = response.get('source_documents', [])
        sources = []
        for i, doc in enumerate(source_docs[:3]):
            content = doc.page_content.strip()
            if len(content) > 200:
                content = content[:200] + "..."
            sources.append(f"Source {i+1}: {content}")
        
        st.session_state.debug_info.append("✅ Generated response using document context")
        return answer, sources
        
    except Exception as e:
        error_msg = f"❌ Error in document-based chat: {e}"
        st.session_state.debug_info.append(error_msg)
        st.error(error_msg)
        return None, []

def main():
    """Main Streamlit app"""
    st.title("📚 PDF Chatbot")
    st.write("Upload PDF files and chat with their content using AI")
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("📁 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload PDF files to chat with their content"
        )
        
        if uploaded_files:
            if st.button("Process PDFs", type="primary"):
                with st.spinner("Processing PDFs..."):
                    st.session_state.vectorstore = process_pdfs(uploaded_files)
                    if st.session_state.vectorstore:
                        st.success(f"✅ Processed {len(uploaded_files)} PDF(s) successfully!")
                        st.info(f"📄 {st.session_state.document_count} text chunks ready for Q&A")
                    else:
                        st.error("❌ Failed to process PDFs - check debug info below")
        
        st.header("⚙️ Settings")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make the output more random"
        )
        
        # Model selection
        model_options = [
            'llama-3.3-70b-versatile',
            'mixtral-8x7b-32768',
            'gemma2-9b-it'
        ]
        selected_model = st.selectbox(
            "Model",
            model_options,
            index=0,
            help="Choose the AI model to use"
        )
        
        os.environ['GROQ_MODEL'] = selected_model
        
        # Status display
        if st.session_state.vectorstore:
            st.success("📄 PDFs loaded and ready")
            st.info(f"📊 {st.session_state.document_count} chunks available")
        else:
            st.info("💡 Upload and process PDFs to enable document chat")
        
        # Debug info
        show_debug = st.checkbox("Show Debug Info", value=True)
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
            
        if st.button("🔄 Reset All"):
            st.session_state.messages = []
            st.session_state.vectorstore = None
            st.session_state.pdf_processed = False
            st.session_state.debug_info = []
            st.session_state.document_count = 0
            st.rerun()
    
    # Debug Info Section
    if show_debug and st.session_state.debug_info:
        st.header("🔧 Debug Information")
        with st.expander("Processing Log", expanded=False):
            for info in st.session_state.debug_info:
                st.text(info)
    
    # Main chat interface
    st.header("💬 Chat")
    
    # Show status
    if st.session_state.vectorstore:
        st.success(f"🤖 Ready to answer questions about your documents ({st.session_state.document_count} chunks loaded)")
    else:
        st.info("🤖 Ready for general chat. Upload PDFs to enable document Q&A.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message.get("sources"):
                with st.expander("📖 Sources"):
                    for source in message["sources"]:
                        st.text(source)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your PDFs or chat generally..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, sources = get_chat_response(
                    prompt, 
                    st.session_state.vectorstore, 
                    temperature
                )
            
            if response:
                st.markdown(response)
                
                # Show sources if available
                if sources:
                    with st.expander("📖 Sources"):
                        for source in sources:
                            st.text(source)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources
                })
            else:
                st.error("Failed to generate response. Please try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("Powered by Groq, LangChain, and Streamlit")

if __name__ == "__main__":
    # Check for required environment variables
    if not os.environ.get('GROQ_API_KEY'):
        st.error("⚠️ GROQ_API_KEY not found in environment variables")
        st.info("Please set your GROQ_API_KEY in the .env file")
        st.stop()
    
    main()