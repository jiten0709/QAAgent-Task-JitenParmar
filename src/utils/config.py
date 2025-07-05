import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

def load_environment():
    """Load environment variables from .env file"""
    # Load from project root .env file
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    
    # Also try from src/.env
    src_env_path = Path(__file__).parent.parent / ".env"
    if src_env_path.exists():
        load_dotenv(src_env_path)

def get_openai_api_key():
    """Get OpenAI API key from environment or session state"""
    # Check session state first (user input)
    if hasattr(st, 'session_state') and 'openai_api_key' in st.session_state and st.session_state.openai_api_key:
        return st.session_state.openai_api_key
    
    # Fallback to environment variable
    return os.getenv('OPENAI_API_KEY')

def set_openai_api_key(api_key: str):
    """Set OpenAI API key in session state and environment"""
    if hasattr(st, 'session_state'):
        st.session_state.openai_api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key