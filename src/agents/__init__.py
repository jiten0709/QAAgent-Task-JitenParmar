"""
Agents Module - Core AI agents for test generation and execution

This module contains the intelligent agents that power the QA automation:
- DataIngestionAgent: Processes videos and documents
- TestGeneratorAgent: Generates test cases using RAG and LLM
- TestExecutorAgent: Executes tests using Playwright
"""

from .data_ingestion import DataIngestionAgent
from .test_generator import TestGeneratorAgent  
from .test_executor import TestExecutorAgent

__all__ = [
    "DataIngestionAgent",
    "TestGeneratorAgent", 
    "TestExecutorAgent"
]

# Agent version tracking
AGENT_VERSION = "1.0.0"
SUPPORTED_FRAMEWORKS = ["playwright", "selenium", "cypress"]
SUPPORTED_FORMATS = ["json", "yaml", "markdown"]
