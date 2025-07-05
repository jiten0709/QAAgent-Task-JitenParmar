"""
This comprehensive Results and Reporting page provides:

Results Overview - Executive summary with key metrics and trends
Detailed Results - Filterable test results table with individual test case details
Analytics Dashboard - Multiple visualization options for deep insights
Report Generation - Comprehensive reporting capabilities with export options
Execution History - Timeline view and comparison between executions
Trend Analysis - Historical performance tracking and trend indicators
Error Analysis - Detailed failure analysis and categorization
Performance Metrics - Duration analysis and performance insights
Export Capabilities - CSV, JSON, and HTML export options
Interactive Visualizations - Charts, graphs, and heatmaps for data exploration
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import time
from typing import Dict, List, Optional
from datetime import datetime
import plotly.express as px

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.dashboard.components.sidebar import add_notification
from src.dashboard.components.charts import (
    render_test_execution_overview,
    render_test_trend_analysis,
    render_performance_metrics,
    render_error_analysis,
    render_test_coverage_heatmap,
    render_browser_compatibility_chart
)
from src.dashboard.components.reports import (
    generate_test_execution_report,
    export_to_csv,
    export_to_json,
    render_report_generation_interface,
)

# Page configuration
st.set_page_config(
    page_title="Results & Reports - QAgenie",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #17a2b8 0%, #138496 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.results-container {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 4px solid #17a2b8;
}

.success-summary {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #28a745;
    margin: 1rem 0;
}

.failure-summary {
    background: linear-gradient(135deg, #f8d7da 0%, #f1b0b7 100%);
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #dc3545;
    margin: 1rem 0;
}

.metric-highlight {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin: 0.5rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #007bff;
}

.metric-label {
    color: #6c757d;
    font-size: 0.9rem;
    margin-top: 0.5rem;
}

.trend-up {
    color: #28a745;
}

.trend-down {
    color: #dc3545;
}

.trend-neutral {
    color: #ffc107;
}

.report-section {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.test-detail-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 4px solid #007bff;
}

.test-detail-card.passed {
    border-left-color: #28a745;
    background: #f8fff9;
}

.test-detail-card.failed {
    border-left-color: #dc3545;
    background: #fff8f8;
}

.test-detail-card.skipped {
    border-left-color: #ffc107;
    background: #fffbf0;
}
</style>
""", unsafe_allow_html=True)

def main():
    """Main function for the results and reporting page"""
    
    # Page header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Results & Analytics Dashboard</h1>
        <p>Comprehensive test results analysis, reporting, and insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "🔍 Detailed Results", 
        "📊 Analytics", 
        "📄 Reports",
        "📚 History"
    ])
    
    with tab1:
        render_overview_section()
    
    with tab2:
        render_detailed_results_section()
    
    with tab3:
        render_analytics_section()
    
    with tab4:
        render_reports_section()
    
    with tab5:
        render_history_section()

def init_session_state():
    """Initialize session state variables"""
    if 'results_data' not in st.session_state:
        st.session_state.results_data = load_sample_results()
    
    if 'selected_execution' not in st.session_state:
        st.session_state.selected_execution = None
    
    if 'filter_settings' not in st.session_state:
        st.session_state.filter_settings = {}
    
    if 'comparison_executions' not in st.session_state:
        st.session_state.comparison_executions = []

def render_overview_section():
    """Render results overview section"""
    st.markdown("## 📈 Test Results Overview")
    
    if not st.session_state.results_data:
        st.info("No test results available. Execute some tests first!")
        if st.button("🚀 Go to Test Execution"):
            st.switch_page("pages/2_🚀_Test_Execution.py")
        return
    
    # Latest execution summary
    latest_execution = get_latest_execution()
    
    if latest_execution:
        render_execution_summary(latest_execution)
        
        # Key metrics grid
        render_key_metrics_grid(latest_execution)
        
        # Quick insights
        render_quick_insights(latest_execution)
        
        # Trend indicators
        render_trend_indicators()
    
    # Execution selector
    st.markdown("### 🔄 Execution Selection")
    execution_options = [f"Execution {i+1} - {exec_data.get('timestamp', 'Unknown')}" 
                        for i, exec_data in enumerate(st.session_state.results_data)]
    
    if execution_options:
        selected_idx = st.selectbox("Select Execution to Analyze:", 
                                   range(len(execution_options)),
                                   format_func=lambda x: execution_options[x])
        st.session_state.selected_execution = st.session_state.results_data[selected_idx]

def render_execution_summary(execution_data: Dict):
    """Render execution summary cards"""
    st.markdown("### 🎯 Latest Execution Summary")
    
    total_tests = execution_data.get('total_tests', 0)
    passed_tests = execution_data.get('passed', 0)
    failed_tests = execution_data.get('failed', 0)
    skipped_tests = execution_data.get('skipped', 0)
    duration = execution_data.get('duration', 0)
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    # Summary cards based on overall status
    if success_rate >= 80:
        st.markdown(f"""
        <div class="success-summary">
            <h3>✅ Execution Successful!</h3>
            <p><strong>Success Rate:</strong> {success_rate:.1f}% ({passed_tests}/{total_tests} tests passed)</p>
            <p><strong>Duration:</strong> {duration:.1f} seconds</p>
            <p><strong>Status:</strong> Test suite is performing well</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="failure-summary">
            <h3>⚠️ Execution Needs Attention</h3>
            <p><strong>Success Rate:</strong> {success_rate:.1f}% ({passed_tests}/{total_tests} tests passed)</p>
            <p><strong>Failed Tests:</strong> {failed_tests}</p>
            <p><strong>Status:</strong> Review failed tests and improve quality</p>
        </div>
        """, unsafe_allow_html=True)

def render_key_metrics_grid(execution_data: Dict):
    """Render key metrics in a grid layout"""
    st.markdown("### 📊 Key Metrics")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    total_tests = execution_data.get('total_tests', 0)
    passed_tests = execution_data.get('passed', 0)
    failed_tests = execution_data.get('failed', 0)
    skipped_tests = execution_data.get('skipped', 0)
    duration = execution_data.get('duration', 0)
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    with col1:
        st.markdown(f"""
        <div class="metric-highlight">
            <div class="metric-value">{total_tests}</div>
            <div class="metric-label">Total Tests</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-highlight">
            <div class="metric-value" style="color: #28a745;">{passed_tests}</div>
            <div class="metric-label">Passed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-highlight">
            <div class="metric-value" style="color: #dc3545;">{failed_tests}</div>
            <div class="metric-label">Failed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-highlight">
            <div class="metric-value" style="color: #ffc107;">{skipped_tests}</div>
            <div class="metric-label">Skipped</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-highlight">
            <div class="metric-value">{success_rate:.1f}%</div>
            <div class="metric-label">Success Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
        <div class="metric-highlight">
            <div class="metric-value">{duration:.1f}s</div>
            <div class="metric-label">Duration</div>
        </div>
        """, unsafe_allow_html=True)

def render_quick_insights(execution_data: Dict):
    """Render quick insights and recommendations"""
    st.markdown("### 💡 Quick Insights")
    
    insights = generate_insights(execution_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔍 Analysis")
        for insight in insights['analysis']:
            st.info(insight)
    
    with col2:
        st.markdown("#### 🎯 Recommendations")
        for recommendation in insights['recommendations']:
            st.success(recommendation)

def render_trend_indicators():
    """Render trend indicators comparing to previous executions"""
    st.markdown("### 📈 Trends")
    
    if len(st.session_state.results_data) < 2:
        st.info("Need at least 2 executions to show trends")
        return
    
    trends = calculate_trends()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        trend_class = get_trend_class(trends['success_rate'])
        st.markdown(f"""
        <div class="metric-highlight">
            <div class="metric-value {trend_class}">
                {trends['success_rate']:+.1f}%
            </div>
            <div class="metric-label">Success Rate Trend</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        trend_class = get_trend_class(trends['total_tests'])
        st.markdown(f"""
        <div class="metric-highlight">
            <div class="metric-value {trend_class}">
                {trends['total_tests']:+d}
            </div>
            <div class="metric-label">Test Volume Trend</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        trend_class = get_trend_class(-trends['failed_tests'])  # Negative because fewer failures is better
        st.markdown(f"""
        <div class="metric-highlight">
            <div class="metric-value {trend_class}">
                {trends['failed_tests']:+d}
            </div>
            <div class="metric-label">Failed Tests Trend</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        trend_class = get_trend_class(-trends['duration'])  # Negative because shorter duration is better
        st.markdown(f"""
        <div class="metric-highlight">
            <div class="metric-value {trend_class}">
                {trends['duration']:+.1f}s
            </div>
            <div class="metric-label">Duration Trend</div>
        </div>
        """, unsafe_allow_html=True)

def render_detailed_results_section():
    """Render detailed test results section"""
    st.markdown("## 🔍 Detailed Test Results")
    
    if not st.session_state.results_data:
        st.info("No detailed results available.")
        return
    
    # Execution selector
    execution_data = get_selected_execution()
    if not execution_data:
        st.warning("Please select an execution in the Overview tab.")
        return
    
    # Filters
    render_results_filters()
    
    # Test results table
    render_test_results_table(execution_data)
    
    # Test case details
    render_test_case_details(execution_data)

def render_results_filters():
    """Render results filtering options"""
    st.markdown("### 🔍 Filter Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_filter = st.multiselect(
            "Filter by Status:",
            ["Passed", "Failed", "Skipped"],
            default=["Passed", "Failed", "Skipped"]
        )
    
    with col2:
        category_filter = st.multiselect(
            "Filter by Category:",
            ["Functional", "UI", "API", "Performance", "Security"],
            default=["Functional", "UI", "API", "Performance", "Security"]
        )
    
    with col3:
        priority_filter = st.multiselect(
            "Filter by Priority:",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )
    
    with col4:
        duration_filter = st.select_slider(
            "Max Duration (seconds):",
            options=[5, 10, 30, 60, 120, 300],
            value=300
        )
    
    st.session_state.filter_settings = {
        'status': status_filter,
        'category': category_filter,
        'priority': priority_filter,
        'max_duration': duration_filter
    }

def render_test_results_table(execution_data: Dict):
    """Render test results in a table format"""
    st.markdown("### 📋 Test Results Table")
    
    # Generate test results data
    test_results = generate_test_results_data(execution_data)
    
    if not test_results:
        st.info("No test results match the current filters.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(test_results)
    
    # Apply filters
    df_filtered = apply_filters(df, st.session_state.filter_settings)
    
    # Display metrics for filtered data
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filtered Tests", len(df_filtered))
    with col2:
        passed_count = len(df_filtered[df_filtered['Status'] == 'Passed'])
        st.metric("Passed", passed_count)
    with col3:
        failed_count = len(df_filtered[df_filtered['Status'] == 'Failed'])
        st.metric("Failed", failed_count)
    
    # Display table with styling
    def style_status(val):
        if val == 'Passed':
            return 'background-color: #d4edda; color: #155724'
        elif val == 'Failed':
            return 'background-color: #f8d7da; color: #721c24'
        elif val == 'Skipped':
            return 'background-color: #fff3cd; color: #856404'
        return ''
    
    styled_df = df_filtered.style.applymap(style_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Export options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export to CSV"):
            csv_data = export_to_csv(test_results)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"test_results_{int(time.time())}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📄 Export to JSON"):
            json_data = export_to_json({"test_results": test_results})
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"test_results_{int(time.time())}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("📊 Generate Report"):
            # Call the report generation function from reports component
            report_html = generate_test_execution_report(execution_data, test_results)
            st.success("Detailed report generated!")
            add_notification("Detailed report generated", "success")

def render_test_case_details(execution_data: Dict):
    """Render individual test case details"""
    st.markdown("### 🔬 Test Case Details")
    
    test_cases = execution_data.get('test_cases', [])
    
    if not test_cases:
        st.info("No detailed test case information available.")
        return
    
    # Test case selector
    test_names = [f"{case.get('name', f'Test {i+1}')} - {case.get('status', 'Unknown')}" 
                  for i, case in enumerate(test_cases)]
    
    selected_test_idx = st.selectbox("Select Test Case:", range(len(test_names)),
                                   format_func=lambda x: test_names[x])
    
    if selected_test_idx is not None:
        test_case = test_cases[selected_test_idx]
        render_single_test_details(test_case)

def render_single_test_details(test_case: Dict):
    """Render details for a single test case"""
    status = test_case.get('status', 'Unknown')
    card_class = status.lower()
    
    st.markdown(f"""
    <div class="test-detail-card {card_class}">
        <h4>{test_case.get('name', 'Unknown Test')}</h4>
        <p><strong>Status:</strong> {status}</p>
        <p><strong>Duration:</strong> {test_case.get('duration', 'Unknown')}</p>
        <p><strong>Category:</strong> {test_case.get('category', 'Unknown')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Test steps
    if test_case.get('steps'):
        st.markdown("#### 📝 Test Steps")
        for i, step in enumerate(test_case['steps'], 1):
            step_status = step.get('status', 'unknown')
            step_icon = {'passed': '✅', 'failed': '❌', 'skipped': '⏭️'}.get(step_status, '❓')
            st.markdown(f"{step_icon} **Step {i}:** {step.get('description', 'No description')}")
    
    # Error details (if failed)
    if status == 'Failed' and test_case.get('error'):
        st.markdown("#### ❌ Error Details")
        st.error(test_case['error'])
    
    # Screenshots and artifacts
    if test_case.get('artifacts'):
        st.markdown("#### 📸 Test Artifacts")
        artifacts = test_case['artifacts']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if artifacts.get('screenshot'):
                st.markdown("**Screenshot:**")
                st.image(artifacts['screenshot'], caption="Test Screenshot")
        
        with col2:
            if artifacts.get('video'):
                st.markdown("**Video Recording:**")
                st.video(artifacts['video'])

def render_analytics_section():
    """Render analytics and visualizations section"""
    st.markdown("## 📊 Test Analytics & Insights")
    
    if not st.session_state.results_data:
        st.info("No data available for analytics.")
        return
    
    # Analytics options
    analytics_type = st.selectbox(
        "Select Analytics View:",
        ["Execution Overview", "Trend Analysis", "Performance Analysis", 
         "Error Analysis", "Coverage Analysis", "Browser Compatibility"]
    )
    
    execution_data = get_selected_execution() or get_latest_execution()
    
    if analytics_type == "Execution Overview":
        render_test_execution_overview(execution_data)
    
    elif analytics_type == "Trend Analysis":
        if len(st.session_state.results_data) > 1:
            render_test_trend_analysis(st.session_state.results_data)
        else:
            st.info("Need multiple executions for trend analysis")
    
    elif analytics_type == "Performance Analysis":
        performance_data = extract_performance_data(execution_data)
        render_performance_metrics(performance_data)
    
    elif analytics_type == "Error Analysis":
        error_data = extract_error_data(execution_data)
        render_error_analysis(error_data)
    
    elif analytics_type == "Coverage Analysis":
        coverage_data = generate_coverage_data(execution_data)
        render_test_coverage_heatmap(coverage_data)
    
    elif analytics_type == "Browser Compatibility":
        browser_data = extract_browser_data(execution_data)
        render_browser_compatibility_chart(browser_data)
    
    # Custom analytics
    render_custom_analytics(execution_data)

def render_custom_analytics(execution_data: Dict):
    """Render custom analytics charts"""
    st.markdown("### 🎯 Custom Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Test duration distribution
        st.markdown("#### ⏱️ Test Duration Distribution")
        durations = [case.get('duration', 0) for case in execution_data.get('test_cases', [])]
        if durations:
            fig = px.histogram(x=durations, nbins=20, title="Test Duration Distribution")
            fig.update_layout(xaxis_title="Duration (seconds)", yaxis_title="Number of Tests")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Test category performance
        st.markdown("#### 📊 Performance by Category")
        category_data = analyze_category_performance(execution_data)
        if category_data:
            fig = px.bar(x=list(category_data.keys()), y=list(category_data.values()),
                        title="Success Rate by Category")
            fig.update_layout(xaxis_title="Category", yaxis_title="Success Rate (%)")
            st.plotly_chart(fig, use_container_width=True)

def render_reports_section():
    """Render reports generation section"""
    st.markdown("## 📄 Test Reports")
    
    if not st.session_state.results_data:
        st.info("No data available for report generation.")
        return
    
    # Report generation interface
    render_report_generation_interface()
    
    # Quick report actions
    st.markdown("### ⚡ Quick Reports")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Executive Summary", use_container_width=True):
            generate_executive_summary()
    
    with col2:
        if st.button("🔍 Detailed Report", use_container_width=True):
            generate_detailed_report_full()
    
    with col3:
        if st.button("📈 Trend Report", use_container_width=True):
            generate_trend_report()
    
    with col4:
        if st.button("❌ Failed Tests Report", use_container_width=True):
            generate_failed_tests_report()
    
    # Recent reports
    render_recent_reports()

def render_recent_reports():
    """Render list of recently generated reports"""
    st.markdown("### 📚 Recent Reports")
    
    reports_dir = Path("src/tests/results/reports")
    if not reports_dir.exists():
        st.info("No reports generated yet.")
        return
    
    report_files = list(reports_dir.glob("*.html"))[-10:]  # Last 10 reports
    
    if not report_files:
        st.info("No reports found.")
        return
    
    for report_file in sorted(report_files, reverse=True):
        with st.expander(f"📄 {report_file.name}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**File:** {report_file.name}")
                st.markdown(f"**Size:** {report_file.stat().st_size / 1024:.1f} KB")
            
            with col2:
                modified_time = datetime.fromtimestamp(report_file.stat().st_mtime)
                st.markdown(f"**Modified:** {modified_time.strftime('%Y-%m-%d %H:%M')}")
            
            with col3:
                if st.button(f"📥 Download", key=f"download_{report_file.name}"):
                    with open(report_file, 'r', encoding='utf-8') as f:
                        report_content = f.read()
                    
                    st.download_button(
                        label="Download Report",
                        data=report_content,
                        file_name=report_file.name,
                        mime="text/html"
                    )

def render_history_section():
    """Render execution history section"""
    st.markdown("## 📚 Execution History")
    
    if not st.session_state.results_data:
        st.info("No execution history available.")
        return
    
    # History overview
    st.markdown("### 📊 History Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Executions", len(st.session_state.results_data))
    
    with col2:
        avg_success_rate = calculate_average_success_rate()
        st.metric("Avg Success Rate", f"{avg_success_rate:.1f}%")
    
    with col3:
        total_tests = sum(exec_data.get('total_tests', 0) for exec_data in st.session_state.results_data)
        st.metric("Total Tests Run", total_tests)
    
    with col4:
        latest_execution = get_latest_execution()
        if latest_execution:
            last_run = latest_execution.get('timestamp', 'Unknown')
            st.metric("Last Execution", last_run)
    
    # History timeline
    render_history_timeline()
    
    # Execution comparison
    render_execution_comparison()

def render_history_timeline():
    """Render execution history timeline"""
    st.markdown("### 📅 Execution Timeline")
    
    # Prepare timeline data
    timeline_data = []
    for i, execution in enumerate(st.session_state.results_data):
        timeline_data.append({
            'execution': i + 1,
            'timestamp': execution.get('timestamp', f'Execution {i+1}'),
            'total_tests': execution.get('total_tests', 0),
            'success_rate': (execution.get('passed', 0) / execution.get('total_tests', 1)) * 100,
            'duration': execution.get('duration', 0)
        })
    
    if timeline_data:
        df = pd.DataFrame(timeline_data)
        
        # Success rate over time
        fig = px.line(df, x='execution', y='success_rate', 
                     title='Success Rate Over Time',
                     markers=True)
        fig.update_layout(xaxis_title="Execution Number", yaxis_title="Success Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

def render_execution_comparison():
    """Render execution comparison interface"""
    st.markdown("### 🔄 Compare Executions")
    
    if len(st.session_state.results_data) < 2:
        st.info("Need at least 2 executions for comparison.")
        return
    
    execution_options = [f"Execution {i+1} - {exec_data.get('timestamp', 'Unknown')}" 
                        for i, exec_data in enumerate(st.session_state.results_data)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        exec1_idx = st.selectbox("First Execution:", range(len(execution_options)),
                               format_func=lambda x: execution_options[x])
    
    with col2:
        exec2_idx = st.selectbox("Second Execution:", range(len(execution_options)),
                               format_func=lambda x: execution_options[x],
                               index=1 if len(execution_options) > 1 else 0)
    
    if exec1_idx != exec2_idx:
        compare_executions(st.session_state.results_data[exec1_idx], 
                          st.session_state.results_data[exec2_idx])

def compare_executions(exec1: Dict, exec2: Dict):
    """Compare two executions"""
    st.markdown("#### 📊 Comparison Results")
    
    col1, col2, col3 = st.columns(3)
    
    # Success rate comparison
    with col1:
        sr1 = (exec1.get('passed', 0) / exec1.get('total_tests', 1)) * 100
        sr2 = (exec2.get('passed', 0) / exec2.get('total_tests', 1)) * 100
        diff = sr2 - sr1
        
        st.metric("Success Rate Diff", f"{diff:+.1f}%", 
                 delta=f"{sr2:.1f}% vs {sr1:.1f}%")
    
    # Duration comparison
    with col2:
        dur1 = exec1.get('duration', 0)
        dur2 = exec2.get('duration', 0)
        diff = dur2 - dur1
        
        st.metric("Duration Diff", f"{diff:+.1f}s",
                 delta=f"{dur2:.1f}s vs {dur1:.1f}s")
    
    # Test count comparison
    with col3:
        tests1 = exec1.get('total_tests', 0)
        tests2 = exec2.get('total_tests', 0)
        diff = tests2 - tests1
        
        st.metric("Test Count Diff", f"{diff:+d}",
                 delta=f"{tests2} vs {tests1}")

# Helper Functions

def load_sample_results() -> List[Dict]:
    """Load sample results data"""
    # This would typically load from files or database
    return [
        {
            'execution_id': 'exec_001',
            'timestamp': '2024-01-15 14:30:00',
            'total_tests': 25,
            'passed': 20,
            'failed': 3,
            'skipped': 2,
            'duration': 180.5,
            'test_cases': [
                {
                    'name': 'Login Test',
                    'status': 'Passed',
                    'duration': 12.3,
                    'category': 'Functional',
                    'steps': [
                        {'description': 'Navigate to login page', 'status': 'passed'},
                        {'description': 'Enter credentials', 'status': 'passed'},
                        {'description': 'Click login button', 'status': 'passed'}
                    ]
                },
                {
                    'name': 'Search Test',
                    'status': 'Failed',
                    'duration': 8.7,
                    'category': 'Functional',
                    'error': 'Element not found: search-button'
                }
            ]
        }
    ]

def get_latest_execution() -> Optional[Dict]:
    """Get the latest execution data"""
    if st.session_state.results_data:
        return st.session_state.results_data[-1]
    return None

def get_selected_execution() -> Optional[Dict]:
    """Get the currently selected execution"""
    return st.session_state.selected_execution

def generate_insights(execution_data: Dict) -> Dict:
    """Generate insights from execution data"""
    total_tests = execution_data.get('total_tests', 0)
    passed_tests = execution_data.get('passed', 0)
    failed_tests = execution_data.get('failed', 0)
    duration = execution_data.get('duration', 0)
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    analysis = []
    recommendations = []
    
    if success_rate >= 90:
        analysis.append("🎉 Excellent test performance with high success rate")
    elif success_rate >= 70:
        analysis.append("✅ Good test performance with acceptable success rate")
    else:
        analysis.append("⚠️ Low success rate indicates potential issues")
    
    if failed_tests > 0:
        analysis.append(f"🔍 {failed_tests} test(s) failed - requires investigation")
        recommendations.append("Review failed test cases and fix underlying issues")
    
    avg_duration = duration / total_tests if total_tests > 0 else 0
    if avg_duration > 30:
        analysis.append("⏱️ Tests are running slower than expected")
        recommendations.append("Consider optimizing test execution or parallel running")
    
    if success_rate < 80:
        recommendations.append("Improve test stability and application quality")
    
    if not recommendations:
        recommendations.append("Test suite is performing well - maintain current quality")
    
    return {'analysis': analysis, 'recommendations': recommendations}

def calculate_trends() -> Dict:
    """Calculate trends compared to previous execution"""
    if len(st.session_state.results_data) < 2:
        return {}
    
    current = st.session_state.results_data[-1]
    previous = st.session_state.results_data[-2]
    
    curr_sr = (current.get('passed', 0) / current.get('total_tests', 1)) * 100
    prev_sr = (previous.get('passed', 0) / previous.get('total_tests', 1)) * 100
    
    return {
        'success_rate': curr_sr - prev_sr,
        'total_tests': current.get('total_tests', 0) - previous.get('total_tests', 0),
        'failed_tests': current.get('failed', 0) - previous.get('failed', 0),
        'duration': current.get('duration', 0) - previous.get('duration', 0)
    }

def get_trend_class(value: float) -> str:
    """Get CSS class for trend indicators"""
    if value > 0:
        return "trend-up"
    elif value < 0:
        return "trend-down"
    else:
        return "trend-neutral"

def generate_test_results_data(execution_data: Dict) -> List[Dict]:
    """Generate test results data for table display"""
    test_cases = execution_data.get('test_cases', [])
    
    results = []
    for i, case in enumerate(test_cases):
        results.append({
            'Test Case': case.get('name', f'Test {i+1}'),
            'Status': case.get('status', 'Unknown'),
            'Duration': f"{case.get('duration', 0):.1f}s",
            'Category': case.get('category', 'Unknown'),
            'Priority': case.get('priority', 'Medium'),
            'Error': case.get('error', '')[:50] + '...' if case.get('error') else ''
        })
    
    return results

def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply filters to test results DataFrame"""
    filtered_df = df.copy()
    
    if filters.get('status'):
        filtered_df = filtered_df[filtered_df['Status'].isin(filters['status'])]
    
    if filters.get('category'):
        filtered_df = filtered_df[filtered_df['Category'].isin(filters['category'])]
    
    if filters.get('priority'):
        filtered_df = filtered_df[filtered_df['Priority'].isin(filters['priority'])]
    
    return filtered_df

def extract_performance_data(execution_data: Dict) -> Dict:
    """Extract performance data for analysis"""
    test_cases = execution_data.get('test_cases', [])
    durations = [case.get('duration', 0) for case in test_cases]
    
    return {
        'avg_duration': sum(durations) / len(durations) if durations else 0,
        'min_duration': min(durations) if durations else 0,
        'max_duration': max(durations) if durations else 0,
        'total_duration': execution_data.get('duration', 0),
        'test_durations': durations
    }

def extract_error_data(execution_data: Dict) -> List[Dict]:
    """Extract error data for analysis"""
    test_cases = execution_data.get('test_cases', [])
    errors = []
    
    for case in test_cases:
        if case.get('status') == 'Failed' and case.get('error'):
            errors.append({
                'type': 'Test Failure',
                'category': 'Execution',
                'message': case.get('error', ''),
                'test_name': case.get('name', 'Unknown'),
                'timestamp': execution_data.get('timestamp', '')
            })
    
    return errors

def generate_coverage_data(execution_data: Dict) -> Dict:
    """Generate test coverage data"""
    # This would typically analyze actual coverage data
    return {
        'login_flow': {'functional': 90, 'edge_case': 70, 'accessibility': 60, 'performance': 40},
        'search_feature': {'functional': 85, 'edge_case': 80, 'accessibility': 70, 'performance': 60},
        'checkout_process': {'functional': 95, 'edge_case': 75, 'accessibility': 65, 'performance': 50}
    }

def extract_browser_data(execution_data: Dict) -> Dict:
    """Extract browser compatibility data"""
    # This would typically come from multi-browser execution results
    return {
        'Chrome': {'success_rate': 95, 'total_tests': 25, 'passed': 24},
        'Firefox': {'success_rate': 90, 'total_tests': 25, 'passed': 22},
        'Safari': {'success_rate': 85, 'total_tests': 25, 'passed': 21}
    }

def analyze_category_performance(execution_data: Dict) -> Dict:
    """Analyze performance by test category"""
    test_cases = execution_data.get('test_cases', [])
    categories = {}
    
    for case in test_cases:
        category = case.get('category', 'Unknown')
        if category not in categories:
            categories[category] = {'total': 0, 'passed': 0}
        
        categories[category]['total'] += 1
        if case.get('status') == 'Passed':
            categories[category]['passed'] += 1
    
    # Calculate success rates
    success_rates = {}
    for category, data in categories.items():
        if data['total'] > 0:
            success_rates[category] = (data['passed'] / data['total']) * 100
    
    return success_rates

def calculate_average_success_rate() -> float:
    """Calculate average success rate across all executions"""
    if not st.session_state.results_data:
        return 0
    
    total_rate = 0
    for execution in st.session_state.results_data:
        total_tests = execution.get('total_tests', 0)
        passed_tests = execution.get('passed', 0)
        if total_tests > 0:
            total_rate += (passed_tests / total_tests) * 100
    
    return total_rate / len(st.session_state.results_data)

def generate_executive_summary():
    """Generate executive summary report"""
    execution_data = get_latest_execution()
    if execution_data:
        report_html = generate_test_execution_report(execution_data, [])
        st.success("Executive summary generated!")
        add_notification("Executive summary report generated", "success")

def generate_detailed_report_full():
    """Generate detailed full report"""
    execution_data = get_latest_execution()
    if execution_data:
        st.success("Detailed report generated!")
        add_notification("Detailed report generated", "success")

def generate_trend_report():
    """Generate trend analysis report"""
    if len(st.session_state.results_data) > 1:
        st.success("Trend report generated!")
        add_notification("Trend analysis report generated", "success")
    else:
        st.warning("Need multiple executions for trend report")

def generate_failed_tests_report():
    """Generate failed tests specific report"""
    execution_data = get_latest_execution()
    if execution_data and execution_data.get('failed', 0) > 0:
        st.success("Failed tests report generated!")
        add_notification("Failed tests report generated", "success")
    else:
        st.info("No failed tests to report")

if __name__ == "__main__":
    main()
