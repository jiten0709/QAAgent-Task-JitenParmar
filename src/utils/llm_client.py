"""
This complete implementation provides:

Multi-LLM support - OpenAI GPT models with configurable parameters
Dual vector stores - Both FAISS and ChromaDB support
RAG capabilities - Retrieval-augmented generation for context-aware responses
Document processing - Convert video chunks to searchable documents
QA chains - Question answering with source documents
Conversational chains - Memory-enabled chat functionality
Cost tracking - OpenAI usage and cost monitoring
Context generation - Smart context extraction for test generation
Video analysis - Automated content analysis and insights
Error handling - Comprehensive error management and logging
"""

import os
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path

# LangChain imports
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks.manager import get_openai_callback

from dotenv import load_dotenv
load_dotenv()

# ChromaDB import
import chromadb
from chromadb.config import Settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global constants
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_TEMPERATURE = 0.1
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_RETRIEVER_K = 5
DEFAULT_FAISS_STORE_NAME = "qa_agent_faiss"
DEFAULT_CHROMA_COLLECTION = "qa_agent_collection"
DATA_DIR = "src/data"
VECTOR_STORE_SUBDIR = "vector_stores"
CHROMA_DB_SUBDIR = "chroma_db"
QA_CHAIN_TYPE = "stuff"
MEMORY_KEY = "chat_history"
MEMORY_OUTPUT_KEY = "answer"

# Prompt templates
QA_TEMPLATE = """Use the following pieces of context from video transcript to answer the question. 
Focus on actionable steps and user interface elements mentioned in the context.

Context: {context}

Question: {question}

Answer: Provide a detailed, step-by-step answer based on the video content. 
Include specific actions, UI elements, and any relevant details from the transcript."""

class LLMClient:
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 model_name: str = MODEL_NAME,
                 temperature: float = DEFAULT_TEMPERATURE):
        """Initialize LLM Client with OpenAI and vector stores"""
        
        # Set OpenAI API key
        self.api_key = openai_api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=self.api_key
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            length_function=len,
        )
        
        # Vector stores
        self.faiss_store = None
        self.chroma_store = None
        self.retriever = None
        
        # Memory for conversational chains
        self.memory = ConversationBufferMemory(
            memory_key=MEMORY_KEY,
            return_messages=True,
            output_key=MEMORY_OUTPUT_KEY
        )
        
        # Data directories
        self.data_dir = Path(DATA_DIR)
        self.vector_store_dir = self.data_dir / VECTOR_STORE_SUBDIR
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"LLM Client initialized with model: {model_name}")
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts"""
        try:
            with get_openai_callback() as cb:
                embeddings = self.embeddings.embed_documents(texts)
                logger.info(f"Created embeddings for {len(texts)} texts. Cost: ${cb.total_cost:.4f}")
                return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            return []
    
    def setup_faiss_vectorstore(self, documents: List[Document], 
                               store_name: str = DEFAULT_FAISS_STORE_NAME) -> bool:
        """Setup FAISS vector store from documents"""
        try:
            if not documents:
                logger.warning("No documents provided for FAISS store")
                return False
            
            # Create FAISS vector store
            self.faiss_store = FAISS.from_documents(documents, self.embeddings)
            
            # Save to disk
            faiss_path = self.vector_store_dir / store_name
            self.faiss_store.save_local(str(faiss_path))
            
            # Setup retriever
            self.retriever = self.faiss_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": DEFAULT_RETRIEVER_K}
            )
            
            logger.info(f"FAISS vector store created with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up FAISS store: {str(e)}")
            return False
    
    def setup_chroma_vectorstore(self, documents: List[Document], 
                                collection_name: str = DEFAULT_CHROMA_COLLECTION) -> bool:
        """Setup ChromaDB vector store from documents"""
        try:
            if not documents:
                logger.warning("No documents provided for Chroma store")
                return False
            
            # Setup ChromaDB client
            chroma_path = self.vector_store_dir / CHROMA_DB_SUBDIR
            chroma_path.mkdir(exist_ok=True)
            
            client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection
            try:
                collection = client.get_collection(collection_name)
                client.delete_collection(collection_name)  # Reset if exists
            except:
                pass
            
            collection = client.create_collection(collection_name)
            
            # Create Chroma vector store
            self.chroma_store = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(chroma_path)
            )
            
            # Add documents
            self.chroma_store.add_documents(documents)
            
            # Setup retriever
            self.retriever = self.chroma_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": DEFAULT_RETRIEVER_K}
            )
            
            logger.info(f"ChromaDB vector store created with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up Chroma store: {str(e)}")
            return False
    
    def load_existing_vectorstore(self, store_type: str = "faiss", 
                                 store_name: str = DEFAULT_FAISS_STORE_NAME) -> bool:
        """Load existing vector store"""
        try:
            if store_type == "faiss":
                faiss_path = self.vector_store_dir / store_name
                if faiss_path.exists():
                    self.faiss_store = FAISS.load_local(str(faiss_path), self.embeddings)
                    self.retriever = self.faiss_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": DEFAULT_RETRIEVER_K}
                    )
                    logger.info(f"Loaded existing FAISS store from {faiss_path}")
                    return True
            
            elif store_type == "chroma":
                chroma_path = self.vector_store_dir / CHROMA_DB_SUBDIR
                if chroma_path.exists():
                    client = chromadb.PersistentClient(
                        path=str(chroma_path),
                        settings=Settings(anonymized_telemetry=False)
                    )
                    
                    self.chroma_store = Chroma(
                        client=client,
                        collection_name=store_name,
                        embedding_function=self.embeddings,
                        persist_directory=str(chroma_path)
                    )
                    
                    self.retriever = self.chroma_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": DEFAULT_RETRIEVER_K}
                    )
                    logger.info(f"Loaded existing Chroma store from {chroma_path}")
                    return True
            
            logger.warning(f"No existing {store_type} store found")
            return False
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def create_documents_from_chunks(self, chunks: List[Dict]) -> List[Document]:
        """Convert processed chunks to LangChain Documents"""
        documents = []
        
        for chunk in chunks:
            doc = Document(
                page_content=chunk.get("text", ""),
                metadata={
                    "chunk_id": chunk.get("chunk_id", 0),
                    "start_time": chunk.get("start_time", 0),
                    "end_time": chunk.get("end_time", 0),
                    "actions": chunk.get("actions", []),
                    "ui_elements": chunk.get("ui_elements", []),
                    "step_type": chunk.get("step_type", "unknown"),
                    "confidence": chunk.get("confidence", 0.0)
                }
            )
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} documents from chunks")
        return documents
    
    def similarity_search(self, query: str, k: int = DEFAULT_RETRIEVER_K) -> List[Dict]:
        """Perform similarity search on vector store"""
        if not self.retriever:
            logger.warning("No retriever available. Setup vector store first.")
            return []
        
        try:
            # Get relevant documents
            docs = self.retriever.get_relevant_documents(query)
            
            # Format results
            results = []
            for doc in docs[:k]:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": getattr(doc, 'relevance_score', 0.0)
                })
            
            logger.info(f"Found {len(results)} relevant documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def setup_qa_chain(self, chain_type: str = QA_CHAIN_TYPE) -> Optional[RetrievalQA]:
        """Setup QA chain for question answering"""
        if not self.retriever:
            logger.warning("No retriever available. Setup vector store first.")
            return None
        
        try:
            # Create custom prompt template
            QA_CHAIN_PROMPT = PromptTemplate(
                input_variables=["context", "question"],
                template=QA_TEMPLATE
            )
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type=chain_type,
                retriever=self.retriever,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                return_source_documents=True
            )
            
            logger.info("QA chain setup completed")
            return qa_chain
            
        except Exception as e:
            logger.error(f"Error setting up QA chain: {str(e)}")
            return None
    
    def setup_conversational_chain(self) -> Optional[ConversationalRetrievalChain]:
        """Setup conversational retrieval chain"""
        if not self.retriever:
            logger.warning("No retriever available. Setup vector store first.")
            return None
        
        try:
            # Create conversational chain
            conv_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )
            
            logger.info("Conversational chain setup completed")
            return conv_chain
            
        except Exception as e:
            logger.error(f"Error setting up conversational chain: {str(e)}")
            return None
    
    def generate_response(self, prompt: str, context: str = "") -> Dict:
        """Generate response using LLM with optional context"""
        try:
            messages = []
            
            if context:
                messages.append(SystemMessage(content=f"Context: {context}"))
            
            messages.append(HumanMessage(content=prompt))
            
            with get_openai_callback() as cb:
                response = self.llm(messages)
                
                result = {
                    "success": True,
                    "response": response.content,
                    "usage": {
                        "total_tokens": cb.total_tokens,
                        "total_cost": cb.total_cost
                    }
                }
                
                logger.info(f"Generated response. Tokens: {cb.total_tokens}, Cost: ${cb.total_cost:.4f}")
                return result
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def generate_test_context(self, user_flow: str) -> str:
        """Generate context for test generation from video content"""
        if not self.retriever:
            return ""
        
        try:
            # Search for relevant content
            relevant_docs = self.similarity_search(user_flow, k=3)
            
            if not relevant_docs:
                return ""
            
            # Combine relevant content
            context_parts = []
            for doc in relevant_docs:
                context_parts.append(f"Step: {doc['content']}")
                
                metadata = doc['metadata']
                if metadata.get('actions'):
                    context_parts.append(f"Actions: {', '.join(metadata['actions'])}")
                if metadata.get('ui_elements'):
                    context_parts.append(f"UI Elements: {', '.join(metadata['ui_elements'])}")
                
                context_parts.append("---")
            
            context = "\n".join(context_parts)
            logger.info(f"Generated test context from {len(relevant_docs)} relevant documents")
            
            return context
            
        except Exception as e:
            logger.error(f"Error generating test context: {str(e)}")
            return ""
    
    def analyze_video_content(self, video_data: Dict) -> Dict:
        """Analyze video content and extract insights"""
        try:
            segments = video_data.get("segments", [])
            flows = video_data.get("flows", [])
            
            # Prepare analysis prompt
            analysis_prompt = f"""
            Analyze the following video content and provide insights for test case generation:
            
            Video has {len(segments)} segments and {len(flows)} user flows.
            
            Key flows identified:
            {json.dumps([flow.get('flow_name', 'Unknown') for flow in flows[:5]], indent=2)}
            
            Provide:
            1. Main functionality being demonstrated
            2. Critical user journeys
            3. Key UI elements and interactions
            4. Potential edge cases to test
            5. Recommended test priorities
            """
            
            response = self.generate_response(analysis_prompt)
            
            if response["success"]:
                return {
                    "success": True,
                    "analysis": response["response"],
                    "insights": {
                        "total_segments": len(segments),
                        "total_flows": len(flows),
                        "key_flows": [flow.get('flow_name', 'Unknown') for flow in flows[:5]]
                    }
                }
            else:
                return response
                
        except Exception as e:
            logger.error(f"Error analyzing video content: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_usage_stats(self) -> Dict:
        """Get usage statistics"""
        return {
            "faiss_store_active": self.faiss_store is not None,
            "chroma_store_active": self.chroma_store is not None,
            "retriever_active": self.retriever is not None,
            "memory_messages": len(self.memory.chat_memory.messages) if self.memory else 0,
            "vector_store_path": str(self.vector_store_dir)
        }
    