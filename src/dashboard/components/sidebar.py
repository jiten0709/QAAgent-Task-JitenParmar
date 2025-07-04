"""
This comprehensive sidebar component provides:

Main Navigation - Clean menu with icons and proper styling
System Status - Real-time status indicators for key components
Quick Stats - Essential metrics display with delta values
Settings Panel - API key configuration and data management
Progress Indicators - For multi-step processes
Notifications System - Centralized notification management
File Browser - Simple file exploration capability
Responsive Design - Works well on different screen sizes
"""

import streamlit as st
from streamlit_option_menu import option_menu
import os
from pathlib import Path
from typing import Dict, List, Optional

# Global constants
SIDEBAR_TITLE = "🧩 QAgenie"
SIDEBAR_SUBTITLE = "AI QA Agent"
LOGO_URL = "https://via.placeholder.com/200x100/1f77b4/white?text=QAgenie"

# Navigation menu items
MENU_ITEMS = {
    "home": {"icon": "🏠", "label": "Dashboard"},
    "generation": {"icon": "🎯", "label": "Test Generation"},
    "execution": {"icon": "🚀", "label": "Test Execution"},
    "results": {"icon": "📊", "label": "Results & Reports"},
    "settings": {"icon": "⚙️", "label": "Settings"}
}

# Status indicators
STATUS_COLORS = {
    "success": "#28a745",
    "warning": "#ffc107", 
    "error": "#dc3545",
    "info": "#17a2b8",
    "inactive": "#6c757d"
}

def render_sidebar() -> str:
    """
    Render the main sidebar navigation
    Returns the selected menu item
    """
    with st.sidebar:
        # Logo and title
        st.image(LOGO_URL, width=200)
        st.markdown(f"### {SIDEBAR_TITLE}")
        st.markdown(f"*{SIDEBAR_SUBTITLE}*")
        st.divider()
        
        # Main navigation menu
        selected = option_menu(
            menu_title=None,
            options=list(MENU_ITEMS.keys()),
            icons=[item["icon"] for item in MENU_ITEMS.values()],
            menu_icon="cast",
            default_index=0,
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "#0e1117"},
                "icon": {"color": "#fafafa", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#262730",
                },
                "nav-link-selected": {"background-color": "#1f77b4"},
            }
        )
        
        st.divider()
        
        # System status section
        render_system_status()
        
        st.divider()
        
        # Quick stats
        render_quick_stats()
        
        st.divider()
        
        # Settings and info
        render_settings_section()
        
    return selected

def render_system_status():
    """Render system status indicators"""
    st.markdown("#### 🔍 System Status")
    
    # Check various system components
    status_items = _get_system_status()
    
    for item, status in status_items.items():
        color = STATUS_COLORS.get(status["level"], STATUS_COLORS["inactive"])
        
        # Create status indicator
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="width: 12px; height: 12px; border-radius: 50%; 
                           background-color: {color}; margin-right: 10px;"></div>
                <span style="font-size: 14px;">{item}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

def render_quick_stats():
    """Render quick statistics"""
    st.markdown("#### 📈 Quick Stats")
    
    # Get statistics from session state or default values
    stats = _get_quick_stats()
    
    # Display stats in a compact format
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Tests Generated",
            value=stats.get("tests_generated", 0),
            delta=stats.get("tests_delta", 0)
        )
        
        st.metric(
            label="Success Rate",
            value=f"{stats.get('success_rate', 0)}%",
            delta=f"{stats.get('success_delta', 0)}%"
        )
    
    with col2:
        st.metric(
            label="Tests Executed",
            value=stats.get("tests_executed", 0),
            delta=stats.get("execution_delta", 0)
        )
        
        st.metric(
            label="Avg Duration",
            value=f"{stats.get('avg_duration', 0)}s",
            delta=f"{stats.get('duration_delta', 0)}s"
        )

def render_settings_section():
    """Render settings and configuration section"""
    st.markdown("#### ⚙️ Settings")
    
    # API Key status
    api_key_status = _check_api_key_status()
    
    if api_key_status["configured"]:
        st.success("✅ OpenAI API Key configured")
    else:
        st.error("❌ OpenAI API Key not configured")
        with st.expander("Configure API Key"):
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-..."
            )
            if st.button("Save API Key"):
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key
                    st.success("API Key saved!")
                    st.rerun()
    
    # Theme toggle
    if st.button("🌙 Toggle Theme"):
        # Toggle theme logic would go here
        st.info("Theme toggle functionality")
    
    # Data management
    with st.expander("🗂️ Data Management"):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Cache"):
                st.cache_data.clear()
                st.success("Cache cleared!")
        
        with col2:
            if st.button("Export Data"):
                # Export functionality would go here
                st.info("Export functionality")
    
    # About section
    with st.expander("ℹ️ About QAgenie"):
        st.markdown("""
        **QAgenie v1.0**
        
        An AI-powered QA automation platform that generates and executes 
        test cases from video tutorials and documentation.
        
        **Features:**
        - 🎥 Video-based test generation
        - 🤖 AI-powered test creation
        - 🎭 Playwright automation
        - 📊 Comprehensive reporting
        
        **Support:** 
        - 📧 support@qagenie.com
        - 📚 [Documentation](https://docs.qagenie.com)
        - 🐛 [Report Issues](https://github.com/qagenie/issues)
        """)

def render_progress_indicator(current_step: int, total_steps: int, step_names: List[str]):
    """Render a progress indicator for multi-step processes"""
    st.markdown("#### 🚀 Progress")
    
    # Calculate progress percentage
    progress = min(current_step / total_steps, 1.0)
    
    # Display progress bar
    st.progress(progress)
    
    # Display current step
    if current_step <= len(step_names):
        current_step_name = step_names[current_step - 1]
        st.markdown(f"**Step {current_step}/{total_steps}:** {current_step_name}")
    
    # Display step list
    for i, step_name in enumerate(step_names, 1):
        if i < current_step:
            st.markdown(f"✅ {step_name}")
        elif i == current_step:
            st.markdown(f"🔄 {step_name}")
        else:
            st.markdown(f"⏳ {step_name}")

def render_notifications():
    """Render notification panel"""
    st.markdown("#### 🔔 Notifications")
    
    notifications = _get_notifications()
    
    if not notifications:
        st.info("No new notifications")
        return
    
    for notification in notifications[-5:]:  # Show last 5 notifications
        level = notification.get("level", "info")
        message = notification.get("message", "")
        timestamp = notification.get("timestamp", "")
        
        if level == "success":
            st.success(f"{message} - {timestamp}")
        elif level == "warning":
            st.warning(f"{message} - {timestamp}")
        elif level == "error":
            st.error(f"{message} - {timestamp}")
        else:
            st.info(f"{message} - {timestamp}")

def render_file_browser(directory: str = "src/data", file_types: List[str] = None):
    """Render a simple file browser"""
    st.markdown(f"#### 📁 Files in {directory}")
    
    if file_types is None:
        file_types = [".mp4", ".json", ".md", ".py"]
    
    try:
        directory_path = Path(directory)
        if not directory_path.exists():
            st.warning(f"Directory {directory} does not exist")
            return
        
        files = []
        for file_type in file_types:
            files.extend(directory_path.rglob(f"*{file_type}"))
        
        if not files:
            st.info("No files found")
            return
        
        # Display files
        for file_path in sorted(files)[-10:]:  # Show last 10 files
            relative_path = file_path.relative_to(directory_path)
            file_size = file_path.stat().st_size
            size_mb = file_size / (1024 * 1024)
            
            st.markdown(f"📄 {relative_path} ({size_mb:.2f} MB)")
    
    except Exception as e:
        st.error(f"Error browsing files: {str(e)}")

def _get_system_status() -> Dict[str, Dict]:
    """Get current system status"""
    status = {}
    
    # Check OpenAI API
    api_key = os.getenv("OPENAI_API_KEY")
    status["OpenAI API"] = {
        "level": "success" if api_key else "error"
    }
    
    # Check data directories
    data_dir = Path("src/data")
    status["Data Directory"] = {
        "level": "success" if data_dir.exists() else "warning"
    }
    
    # Check test directory
    test_dir = Path("src/tests")
    status["Test Directory"] = {
        "level": "success" if test_dir.exists() else "warning"
    }
    
    # Check if there are any test cases
    test_cases_dir = Path("src/data/test_cases")
    has_test_cases = test_cases_dir.exists() and any(test_cases_dir.iterdir())
    status["Test Cases"] = {
        "level": "success" if has_test_cases else "inactive"
    }
    
    return status

def _get_quick_stats() -> Dict:
    """Get quick statistics for display"""
    # These would typically come from session state or database
    # For now, return default/example values
    
    return {
        "tests_generated": st.session_state.get("total_tests_generated", 0),
        "tests_delta": st.session_state.get("tests_delta", 0),
        "tests_executed": st.session_state.get("total_tests_executed", 0),
        "execution_delta": st.session_state.get("execution_delta", 0),
        "success_rate": st.session_state.get("success_rate", 0),
        "success_delta": st.session_state.get("success_delta", 0),
        "avg_duration": st.session_state.get("avg_duration", 0),
        "duration_delta": st.session_state.get("duration_delta", 0)
    }

def _check_api_key_status() -> Dict:
    """Check if API keys are configured"""
    openai_key = os.getenv("OPENAI_API_KEY")
    
    return {
        "configured": bool(openai_key),
        "openai": bool(openai_key)
    }

def _get_notifications() -> List[Dict]:
    """Get current notifications"""
    # Return notifications from session state or default empty list
    return st.session_state.get("notifications", [])

def add_notification(message: str, level: str = "info", persist: bool = False):
    """Add a notification to the system"""
    from datetime import datetime
    
    if "notifications" not in st.session_state:
        st.session_state.notifications = []
    
    notification = {
        "message": message,
        "level": level,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "persist": persist
    }
    
    st.session_state.notifications.append(notification)
    
    # Keep only last 20 notifications
    if len(st.session_state.notifications) > 20:
        st.session_state.notifications = st.session_state.notifications[-20:]

def clear_notifications():
    """Clear all non-persistent notifications"""
    if "notifications" in st.session_state:
        st.session_state.notifications = [
            n for n in st.session_state.notifications if n.get("persist", False)
        ]

def update_stats(key: str, value: int, delta: int = 0):
    """Update statistics in session state"""
    st.session_state[key] = value
    if delta != 0:
        st.session_state[f"{key.replace('total_', '')}_delta"] = delta
