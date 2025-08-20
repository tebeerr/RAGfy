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
from PIL import Image
import cv2
import numpy as np
from transformers import pipeline
import io

# Load environment variables
load_dotenv()

# Suppress deprecation warnings to reduce noise
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAGfy",
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
if "uploaded_files_hash" not in st.session_state:
    st.session_state.uploaded_files_hash = None
if "editing_message_id" not in st.session_state:
    st.session_state.editing_message_id = None
if "edited_message" not in st.session_state:
    st.session_state.edited_message = ""
if "image_analyzer" not in st.session_state:
    st.session_state.image_analyzer = None
if "video_analyzer" not in st.session_state:
    st.session_state.video_analyzer = None
if "media_processed" not in st.session_state:
    st.session_state.media_processed = False

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

def get_files_hash(files):
    """Create a hash of uploaded files to detect changes"""
    if not files:
        return None
    file_info = [(f.name, f.size, f.last_modified) for f in files if f]
    return hash(str(sorted(file_info)))

def get_chat_response(message, vectorstore=None, temperature=0.9):
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

def get_image_analyzer():
    """Get or create the image analysis model"""
    try:
        if st.session_state.image_analyzer is None:
            st.session_state.debug_info.append("🖼️ Initializing image analyzer...")
            
            # Use a lightweight image captioning model
            image_analyzer = pipeline(
                "image-to-text",
                model="nlpconnect/vit-gpt2-image-captioning",
                device=-1  # Use CPU for compatibility
            )
            
            st.session_state.image_analyzer = image_analyzer
            st.session_state.debug_info.append("✅ Image analyzer initialized successfully")
        
        return st.session_state.image_analyzer
    except Exception as e:
        error_msg = f"❌ Error initializing image analyzer: {e}"
        st.session_state.debug_info.append(error_msg)
        st.error(error_msg)
        return None

def analyze_single_image(image_file):
    """Analyze a single image and describe its content"""
    try:
        st.session_state.debug_info.append(f"🖼️ Analyzing image: {image_file.name}")
        
        # Reset file pointer
        image_file.seek(0)
        
        # Load image with PIL
        image = Image.open(image_file)
        
        # Get image info
        width, height = image.size
        file_size = len(image_file.read())
        image_file.seek(0)  # Reset pointer again
        
        st.session_state.debug_info.append(f"📊 Image size: {width}x{height} pixels, {file_size} bytes")
        
        # Get analyzer
        analyzer = get_image_analyzer()
        if not analyzer:
            return f"Error: Could not initialize image analyzer. Image appears to be {width}x{height} pixels."
        
        # Analyze image
        with st.spinner("🔍 Analyzing image content..."):
            result = analyzer(image)
            
        if result and len(result) > 0:
            caption = result[0]['generated_text']
            st.session_state.debug_info.append(f"✅ Image analysis complete: {caption}")
            return caption
        else:
            return f"Image appears to be {width}x{height} pixels, but content analysis failed."
            
    except Exception as e:
        error_msg = f"❌ Error analyzing image {image_file.name}: {e}"
        st.session_state.debug_info.append(error_msg)
        return f"Error analyzing image: {str(e)}"

def analyze_single_video(video_file):
    """Analyze a single video and describe its content"""
    try:
        st.session_state.debug_info.append(f"🎥 Analyzing video: {video_file.name}")
        
        # Reset file pointer
        video_file.seek(0)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_file.read())
            temp_path = temp_file.name
        
        try:
            # Get video properties
            cap = cv2.VideoCapture(temp_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            st.session_state.debug_info.append(f"📊 Video: {width}x{height}, {duration:.2f}s, {fps:.2f} FPS, {frame_count} frames")
            
            # Analyze key frames
            frames_to_analyze = [0, frame_count//4, frame_count//2, 3*frame_count//4, frame_count-1]
            frame_descriptions = []
            
            analyzer = get_image_analyzer()
            if not analyzer:
                return f"Error: Could not initialize analyzer. Video is {width}x{height}, {duration:.2f}s long."
            
            for i, frame_idx in enumerate(frames_to_analyze):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    
                    # Analyze frame
                    with st.spinner(f"🔍 Analyzing frame {i+1}/5..."):
                        result = analyzer(frame_pil)
                    
                    if result and len(result) > 0:
                        frame_desc = result[0]['generated_text']
                        frame_descriptions.append(f"Frame {frame_idx}: {frame_desc}")
                    else:
                        frame_descriptions.append(f"Frame {frame_idx}: Analysis failed")
            
            cap.release()
            
            # Create summary
            if frame_descriptions:
                summary = f"Video analysis complete. Duration: {duration:.2f}s, Resolution: {width}x{height}. Key frames: {'; '.join(frame_descriptions)}"
                st.session_state.debug_info.append(f"✅ Video analysis complete: {len(frame_descriptions)} frames analyzed")
                return summary
            else:
                return f"Video analysis failed. Video is {width}x{height}, {duration:.2f}s long."
                
        finally:
            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass
                
    except Exception as e:
        error_msg = f"❌ Error analyzing video {video_file.name}: {e}"
        st.session_state.debug_info.append(error_msg)
        return f"Error analyzing video: {str(e)}"

def process_media_files(image_files, video_files):
    """Process uploaded media files automatically"""
    if not image_files and not video_files:
        return
    
    # Clear previous media state
    st.session_state.media_processed = False
    st.session_state.debug_info.append("🚀 Starting media analysis...")
    
    all_media_results = []
    
    # Process images
    if image_files:
        st.session_state.debug_info.append(f"🖼️ Processing {len(image_files)} image(s)...")
        for image_file in image_files:
            if image_file:
                result = analyze_single_image(image_file)
                all_media_results.append(f"🖼️ **{image_file.name}**: {result}")
    
    # Process videos
    if video_files:
        st.session_state.debug_info.append(f"🎥 Processing {len(video_files)} video(s)...")
        for video_file in video_files:
            if video_file:
                result = analyze_single_video(video_file)
                all_media_results.append(f"🎥 **{video_file.name}**: {result}")
    
    if all_media_results:
        st.session_state.media_processed = True
        st.session_state.debug_info.append("🎉 Media analysis completed successfully!")
        return all_media_results
    
    return None

def main():
    """Main Streamlit app"""
    st.title("📚 RAGfy")
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
        
        # Add media upload section
        st.header("🖼️ Upload Media")
        uploaded_images = st.file_uploader(
            "Choose images",
            type=["png", "jpg", "jpeg", "gif", "bmp"],
            accept_multiple_files=True,
            help="Upload images to analyze their content"
        )
        
        uploaded_videos = st.file_uploader(
            "Choose videos",
            type=["mp4", "avi", "mov", "mkv", "wmv"],
            accept_multiple_files=True,
            help="Upload videos to analyze their content"
        )
        
        # Auto-process media when uploaded
        if (uploaded_images or uploaded_videos) and not st.session_state.media_processed:
            with st.spinner("🔄 Automatically analyzing media..."):
                media_results = process_media_files(uploaded_images, uploaded_videos)
                if media_results:
                    st.success("✅ Media analysis complete!")
                    st.info(f"📊 Analyzed {len(media_results)} media files")
                else:
                    st.error("❌ Media analysis failed")
        
        # Show media processing status
        if st.session_state.media_processed:
            st.success("🖼️ Media analyzed and ready")
        elif uploaded_images or uploaded_videos:
            st.info("⏳ Analyzing media automatically...")
        
        # AUTOMATIC PROCESSING - Remove the manual button
        if uploaded_files and not st.session_state.pdf_processed:
            # Auto-process when files are uploaded
            with st.spinner("🔄 Automatically processing PDFs..."):
                st.session_state.vectorstore = process_pdfs(uploaded_files)
                if st.session_state.vectorstore:
                    st.success(f"✅ Auto-processed {len(uploaded_files)} PDF(s)!")
                    st.info(f"📄 {st.session_state.document_count} text chunks ready for Q&A")
                else:
                    st.error("❌ Auto-processing failed - check debug info below")
        
        # Show processing status
        if st.session_state.pdf_processed:
            st.success("📄 PDFs loaded and ready")
            st.info(f"📊 {st.session_state.document_count} chunks available")
        elif uploaded_files:
            st.info("⏳ Processing PDFs automatically...")
        else:
            st.info("💡 Upload PDFs to enable document chat")
        
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
        ]
        selected_model = st.selectbox(
            "Model",
            model_options,
            index=0,
            help="Choose the AI model to use"
        )
        
        os.environ['GROQ_MODEL'] = selected_model
        
        # Add a manual reprocess button for when users want to refresh
        if uploaded_files and st.session_state.pdf_processed:
            if st.button("🔄 Reprocess PDFs", type="secondary"):
                with st.spinner("Reprocessing PDFs..."):
                    st.session_state.pdf_processed = False  # Reset flag
                    st.session_state.vectorstore = process_pdfs(uploaded_files)
                    if st.session_state.vectorstore:
                        st.success("✅ PDFs reprocessed successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Reprocessing failed")
        
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
    
    # Enhanced Debug Info Section
    if show_debug and st.session_state.debug_info:
        st.header("🔧 Debug Information")
        with st.expander("Processing Log", expanded=False):
            for info in st.session_state.debug_info:
                st.text(info)
    
    # Media Results Display
    if st.session_state.media_processed and (uploaded_images or uploaded_videos):
        st.header("🖼️ Media Analysis Results")
        
        # Get media results
        media_results = process_media_files(uploaded_images, uploaded_videos)
        if media_results:
            for result in media_results:
                st.markdown(result)
                st.markdown("---")
    
    # Main chat interface
    st.header("💬 Chat")
    
    # Show status
    if st.session_state.vectorstore:
        st.success(f"🤖 Ready to answer questions about your documents ({st.session_state.document_count} chunks loaded)")
    else:
        st.info("🤖 Ready for general chat. Upload PDFs to enable document Q&A.")
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # For user messages, add edit functionality
            if message["role"] == "user":
                # Check if this message is being edited
                if st.session_state.editing_message_id == i:
                    # Show edit form with better styling
                    st.markdown("**✏️ Editing your question:**")
                    edited_text = st.text_area(
                        "Edit your question:",
                        value=message["content"],
                        key=f"edit_text_{i}",
                        height=120,
                        placeholder="Type your edited question here..."
                    )
                    
                    # Action buttons
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        if st.button("✅ Submit", key=f"submit_edit_{i}", type="primary"):
                            # Update the message content
                            st.session_state.messages[i]["content"] = edited_text
                            st.session_state.editing_message_id = None
                            st.session_state.edited_message = ""
                            
                            # Remove the old assistant response
                            if i + 1 < len(st.session_state.messages) and st.session_state.messages[i + 1]["role"] == "assistant":
                                st.session_state.messages.pop(i + 1)
                            
                            # Regenerate response for edited question
                            with st.spinner("🔄 Regenerating answer for your edited question..."):
                                response, sources = get_chat_response(
                                    edited_text, 
                                    st.session_state.vectorstore, 
                                    temperature
                                )
                            
                            if response:
                                # Add new assistant response
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": response,
                                    "sources": sources
                                })
                                
                                st.success("✅ Answer regenerated successfully!")
                            
                            st.rerun()
                    
                    with col2:
                        if st.button("❌ Cancel", key=f"cancel_edit_{i}"):
                            st.session_state.editing_message_id = None
                            st.session_state.edited_message = ""
                            st.rerun()
                    
                    with col3:
                        st.info(" Edit your question above and click Submit to regenerate the answer")
                
                else:
                    # Show normal message with edit button
                    st.markdown(message["content"])
                    
                    # Add edit button with better styling
                    if st.button("✏️ Edit Question", key=f"edit_btn_{i}", help="Click to edit this question"):
                        st.session_state.editing_message_id = i
                        st.session_state.edited_message = message["content"]
                        st.rerun()
            
            else:
                # For assistant messages, just display content
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