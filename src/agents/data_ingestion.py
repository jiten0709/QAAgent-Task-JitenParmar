"""
This implementation provides:

Complete video processing pipeline - Download → Transcript → Chunk → Vectorize
Intelligent chunking - Based on actions, content size, and topic changes
Fallback mechanisms - YouTube API → Whisper for transcripts
Vector storage - FAISS with metadata preservation
Search capabilities - Semantic search with similarity scores
Error handling - Comprehensive logging and error management
Data persistence - Saves processed data and vector stores
"""

import json
from pathlib import Path
from typing import List, Dict
import logging

# Video processing imports
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import whisper

# LangChain imports
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestionAgent:
    def __init__(self, data_dir: str = "src/data"):
        """Initialize the Data Ingestion Agent"""
        self.data_dir = Path(data_dir)
        self.videos_dir = self.data_dir / "videos"
        self.transcripts_dir = self.data_dir / "transcripts"
        
        # Create directories
        self._create_directories()
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Action keywords for intelligent chunking
        self.action_keywords = [
            "click", "select", "enter", "type", "navigate", "open", 
            "close", "save", "upload", "download", "scroll", "drag",
            "button", "menu", "form", "field", "login", "signup"
        ]
    
    def _create_directories(self):
        """Create necessary directories"""
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directories: {self.videos_dir}, {self.transcripts_dir}")
    
    def process_video_content(self, video_url: str) -> Dict:
        """Process video and store in vector database"""
        try:
            logger.info(f"Processing video: {video_url}")
            
            # Step 1: Download video
            download_result = self._download_video(video_url)
            if not download_result["success"]:
                return download_result
            
            # Step 2: Extract transcript
            video_id = download_result["video_id"]
            transcript_result = self._extract_transcript(video_id, download_result["video_path"])
            if not transcript_result["success"]:
                return transcript_result
            
            # Step 3: Chunk and vectorize
            chunks = self._intelligent_chunking(transcript_result["transcript"])
            vectorize_result = self.chunk_and_vectorize(chunks)
            
            # Step 4: Save processed data
            output_file = self._save_processed_data(video_id, {
                "video_info": download_result,
                "transcript": transcript_result["transcript"],
                "chunks": chunks,
                "vector_store_info": vectorize_result
            })
            
            return {
                "success": True,
                "video_id": video_id,
                "video_info": download_result,
                "chunks_count": len(chunks),
                "output_file": str(output_file),
                "vector_store": self.vector_store
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _download_video(self, video_url: str) -> Dict:
        """Download video using pytube"""
        try:
            yt = YouTube(video_url)
            video_id = yt.video_id
            safe_title = "".join(c for c in yt.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            
            # Download video
            stream = yt.streams.get_highest_resolution()
            video_path = stream.download(
                output_path=str(self.videos_dir),
                filename=f"{safe_title}_{video_id}.mp4"
            )
            
            logger.info(f"Downloaded video: {yt.title}")
            return {
                "success": True,
                "video_path": video_path,
                "title": yt.title,
                "duration": yt.length,
                "video_id": video_id,
                "description": yt.description
            }
            
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            return {"success": False, "error": f"Download failed: {str(e)}"}
    
    def _extract_transcript(self, video_id: str, video_path: str) -> Dict:
        """Extract transcript using YouTube API or Whisper"""
        # Try YouTube Transcript API first
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            logger.info("Transcript extracted using YouTube API")
            return {"success": True, "transcript": transcript, "method": "youtube_api"}
        except Exception as e:
            logger.warning(f"YouTube API failed: {str(e)}, trying Whisper...")
        
        # Fallback to Whisper
        try:
            model = whisper.load_model("base")
            result = model.transcribe(video_path)
            
            # Convert Whisper format to YouTube API format
            transcript = []
            for segment in result["segments"]:
                transcript.append({
                    "text": segment["text"].strip(),
                    "start": segment["start"],
                    "duration": segment["end"] - segment["start"]
                })
            
            logger.info("Transcript extracted using Whisper")
            return {"success": True, "transcript": transcript, "method": "whisper"}
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {str(e)}")
            return {"success": False, "error": f"Transcription failed: {str(e)}"}
    
    def _intelligent_chunking(self, transcript: List[Dict]) -> List[Dict]:
        """Intelligent chunking based on content and actions"""
        chunks = []
        current_chunk = {
            "text": "",
            "start_time": 0,
            "end_time": 0,
            "actions": [],
            "chunk_type": "content"
        }
        
        for i, segment in enumerate(transcript):
            text = segment["text"].lower()
            
            # Check for action keywords
            actions_found = [keyword for keyword in self.action_keywords if keyword in text]
            
            # Determine if this should start a new chunk
            should_split = (
                len(current_chunk["text"]) > 800 or  # Size-based split
                (actions_found and current_chunk["text"]) or  # Action-based split
                self._is_topic_change(current_chunk["text"], segment["text"])  # Topic change
            )
            
            if should_split and current_chunk["text"]:
                # Finalize current chunk
                current_chunk["chunk_id"] = len(chunks)
                current_chunk["end_time"] = transcript[i-1]["start"] + transcript[i-1].get("duration", 0)
                chunks.append(current_chunk.copy())
                
                # Start new chunk
                current_chunk = {
                    "text": segment["text"],
                    "start_time": segment["start"],
                    "end_time": segment["start"] + segment.get("duration", 0),
                    "actions": actions_found,
                    "chunk_type": "action" if actions_found else "content"
                }
            else:
                # Add to current chunk
                if not current_chunk["text"]:
                    current_chunk["start_time"] = segment["start"]
                current_chunk["text"] += " " + segment["text"]
                current_chunk["actions"].extend(actions_found)
                current_chunk["end_time"] = segment["start"] + segment.get("duration", 0)
        
        # Add final chunk
        if current_chunk["text"]:
            current_chunk["chunk_id"] = len(chunks)
            chunks.append(current_chunk)
        
        logger.info(f"Created {len(chunks)} intelligent chunks")
        return chunks
    
    def _is_topic_change(self, previous_text: str, current_text: str) -> bool:
        """Simple topic change detection"""
        topic_indicators = [
            "now let's", "next step", "moving on", "let's switch",
            "another way", "alternatively", "on the other hand"
        ]
        return any(indicator in current_text.lower() for indicator in topic_indicators)
    
    def chunk_and_vectorize(self, chunks: List[Dict]) -> Dict:
        """Chunk content and create embeddings"""
        try:
            # Prepare documents for vectorization
            documents = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk["text"],
                    metadata={
                        "chunk_id": chunk["chunk_id"],
                        "start_time": chunk["start_time"],
                        "end_time": chunk["end_time"],
                        "actions": chunk["actions"],
                        "chunk_type": chunk["chunk_type"]
                    }
                )
                documents.append(doc)
            
            # Create vector store
            if documents:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                logger.info(f"Created vector store with {len(documents)} documents")
                
                # Save vector store
                vector_store_path = self.data_dir / "vector_store"
                vector_store_path.mkdir(exist_ok=True)
                self.vector_store.save_local(str(vector_store_path))
                
                return {
                    "success": True,
                    "documents_count": len(documents),
                    "vector_store_path": str(vector_store_path)
                }
            else:
                return {"success": False, "error": "No documents to vectorize"}
                
        except Exception as e:
            logger.error(f"Error in vectorization: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def setup_retrieval_chain(self):
        """Setup RAG retrieval chain"""
        if self.vector_store is None:
            # Try to load existing vector store
            vector_store_path = self.data_dir / "vector_store"
            if vector_store_path.exists():
                try:
                    self.vector_store = FAISS.load_local(
                        str(vector_store_path), 
                        self.embeddings
                    )
                    logger.info("Loaded existing vector store")
                except Exception as e:
                    logger.error(f"Failed to load vector store: {str(e)}")
                    return None
            else:
                logger.warning("No vector store found. Process a video first.")
                return None
        
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        return retriever
    
    def _save_processed_data(self, video_id: str, data: Dict) -> Path:
        """Save processed data to JSON file"""
        output_file = self.transcripts_dir / f"{video_id}_processed.json"
        
        # Remove non-serializable objects
        save_data = data.copy()
        if "vector_store" in save_data:
            del save_data["vector_store"]
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved processed data to: {output_file}")
        return output_file
    
    def search_content(self, query: str, k: int = 5) -> List[Dict]:
        """Search content using vector similarity"""
        if self.vector_store is None:
            retriever = self.setup_retrieval_chain()
            if retriever is None:
                return []
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in content search: {str(e)}")
            return []