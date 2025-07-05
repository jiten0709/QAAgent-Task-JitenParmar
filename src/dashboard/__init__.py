"""
Dashboard Module - Streamlit web interface

This module contains the web dashboard for the QA Agent platform:
- Main application entry point
- Page components for test generation, execution, and results
- Reusable UI components and charts
- Report generation utilities
"""

# Dashboard configuration
DASHBOARD_TITLE = "QAgenie - AI-Powered QA Testing Platform"
DASHBOARD_ICON = "🤖"
DEFAULT_LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# Theme colors
COLORS = {
    "primary": "#007bff",
    "success": "#28a745", 
    "warning": "#ffc107",
    "danger": "#dc3545",
    "info": "#17a2b8"
}

# Page configurations
PAGES = {
    "test_generation": {
        "title": "Test Generation",
        "icon": "🎯",
        "description": "Generate intelligent test cases from videos and requirements"
    },
    "test_execution": {
        "title": "Test Execution", 
        "icon": "🚀",
        "description": "Execute and monitor automated test runs"
    },
    "results": {
        "title": "Results & Reports",
        "icon": "📊", 
        "description": "Analyze results and generate comprehensive reports"
    }
}

__all__ = [
    "DASHBOARD_TITLE",
    "DASHBOARD_ICON", 
    "COLORS",
    "PAGES"
]
