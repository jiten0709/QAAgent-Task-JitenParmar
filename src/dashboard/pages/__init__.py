"""
Dashboard Pages Module - Main application pages

This module contains the main pages of the Streamlit dashboard:
- Test Generation page: Video upload, processing, and test generation
- Test Execution page: Test configuration, execution, and monitoring  
- Results page: Results analysis, reporting, and historical data
"""

# Page metadata
PAGES_INFO = {
    "test_generation": {
        "file": "1_🎯_Test_Generation.py",
        "title": "Test Generation",
        "description": "Generate test cases from videos, documentation, and manual input",
        "features": [
            "Video upload and processing",
            "Manual test scenario definition", 
            "AI-powered test generation",
            "Test case management"
        ]
    },
    "test_execution": {
        "file": "2_🚀_Test_Execution.py", 
        "title": "Test Execution",
        "description": "Execute and monitor automated test runs",
        "features": [
            "Test selection and filtering",
            "Execution configuration",
            "Real-time monitoring",
            "Multi-browser testing"
        ]
    },
    "results": {
        "file": "3_📊_Results.py",
        "title": "Results & Reports", 
        "description": "Analyze results and generate comprehensive reports",
        "features": [
            "Results visualization",
            "Trend analysis",
            "Report generation", 
            "Historical comparison"
        ]
    }
}

# Navigation configuration
NAVIGATION_ORDER = ["test_generation", "test_execution", "results"]

# Page settings
DEFAULT_PAGE_CONFIG = {
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "menu_items": {
        'Get Help': 'https://github.com/your-repo/wiki',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': '# QAgenie v1.0.0\nAI-powered QA testing platform'
    }
}

__all__ = [
    "PAGES_INFO",
    "NAVIGATION_ORDER", 
    "DEFAULT_PAGE_CONFIG"
]
