"""
Utils Module - Utility functions and helper classes

This module provides utility functions for:
- Video processing and analysis
- LLM integration and RAG operations
- Playwright script conversion
- Data processing and validation
"""

from .video_processor import VideoProcessor
from .llm_client import LLMClient
from .playwright_converter import PlaywrightConverter

__all__ = [
    "VideoProcessor",
    "LLMClient", 
    "PlaywrightConverter"
]

# Utility configuration
DEFAULT_CHUNK_SIZE = 30  # seconds
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "auto"]
