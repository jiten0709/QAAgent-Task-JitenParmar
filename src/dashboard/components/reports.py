"""
This comprehensive reports component provides:

HTML Report Generation - Beautiful, styled HTML reports with metrics and charts
Executive Summaries - High-level overview with key metrics
Detailed Test Results - Complete test case results with error analysis
Performance Analysis - Duration and performance metrics
Error Analysis - Comprehensive error tracking and categorization
Recommendations - AI-generated insights and suggestions
Export Functionality - CSV, JSON, and HTML export capabilities
Download Links - Easy file download generation
Report Customization - Configurable sections and styling
Dashboard Integration - Summary metrics for dashboards
"""

import streamlit as st
import pandas as pd
import json
import base64
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from jinja2 import Template

# Report templates and styling
HTML_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        {{ css_styles }}
    </style>
</head>
<body>
    <div class="container">
        {{ content }}
    </div>
</body>
</html>
"""

CSS_STYLES = """
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    margin: 0;
    padding: 20px;
    background-color: #f8f9fa;
    color: #333;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    background: white;
    border-radius: 10px;
    box-shadow: 0 0 20px rgba(0,0,0,0.1);
    overflow: hidden;
}

.header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px;
    text-align: center;
}

.header h1 {
    margin: 0;
    font-size: 2.5em;
    font-weight: 300;
}

.header .subtitle {
    margin: 10px 0 0 0;
    opacity: 0.9;
    font-size: 1.1em;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    padding: 30px;
    background: #f8f9fa;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 8px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    transition: transform 0.2s;
}

.metric-card:hover {
    transform: translateY(-2px);
}

.metric-value {
    font-size: 2.5em;
    font-weight: bold;
    color: #007bff;
    margin: 10px 0;
}

.metric-label {
    color: #666;
    font-size: 0.9em;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.section {
    padding: 30px;
    border-bottom: 1px solid #eee;
}

.section:last-child {
    border-bottom: none;
}

.section-title {
    font-size: 1.8em;
    margin-bottom: 20px;
    color: #333;
    border-left: 4px solid #007bff;
    padding-left: 15px;
}

.test-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-top: 20px;
}

.test-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 20px;
    background: #fafafa;
}

.test-card.passed {
    border-left: 4px solid #28a745;
}

.test-card.failed {
    border-left: 4px solid #dc3545;
}

.test-card.skipped {
    border-left: 4px solid #ffc107;
}

.test-title {
    font-weight: bold;
    margin-bottom: 10px;
}

.test-status {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.8em;
    text-transform: uppercase;
    font-weight: bold;
}

.status-passed {
    background: #d4edda;
    color: #155724;
}

.status-failed {
    background: #f8d7da;
    color: #721c24;
}

.status-skipped {
    background: #fff3cd;
    color: #856404;
}

.chart-container {
    margin: 20px 0;
    text-align: center;
}

.footer {
    background: #343a40;
    color: white;
    text-align: center;
    padding: 20px;
    font-size: 0.9em;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
}

th, td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}

th {
    background: #f8f9fa;
    font-weight: bold;
}

.error-details {
    background: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 4px;
    padding: 15px;
    margin: 10px 0;
}

.recommendation {
    background: #d1ecf1;
    border: 1px solid #bee5eb;
    border-radius: 4px;
    padding: 15px;
    margin: 10px 0;
}
"""

def generate_test_execution_report(execution_data: Dict, 
                                 test_results: List[Dict],
                                 save_path: Optional[str] = None) -> str:
    """Generate comprehensive test execution report"""
    
    # Prepare report data
    report_data = {
        "title": f"Test Execution Report - {execution_data.get('execution_id', 'Unknown')}",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "execution_data": execution_data,
        "test_results": test_results
    }
    
    # Generate report sections
    content_sections = []
    
    # Header section
    content_sections.append(_generate_header_section(report_data))
    
    # Executive summary
    content_sections.append(_generate_summary_section(execution_data))
    
    # Test results overview
    content_sections.append(_generate_results_overview(test_results))
    
    # Detailed test results
    content_sections.append(_generate_detailed_results(test_results))
    
    # Error analysis
    content_sections.append(_generate_error_analysis(test_results))
    
    # Performance analysis
    content_sections.append(_generate_performance_analysis(execution_data))
    
    # Recommendations
    content_sections.append(_generate_recommendations(execution_data, test_results))
    
    # Footer
    content_sections.append(_generate_footer_section())
    
    # Combine all sections
    full_content = "\n".join(content_sections)
    
    # Generate final HTML
    template = Template(HTML_REPORT_TEMPLATE)
    html_report = template.render(
        title=report_data["title"],
        css_styles=CSS_STYLES,
        content=full_content
    )
    
    # Save report if path provided
    if save_path:
        save_report_to_file(html_report, save_path)
    
    return html_report

def _generate_header_section(report_data: Dict) -> str:
    """Generate report header section"""
    return f"""
    <div class="header">
        <h1>� TestRAGic Test Report</h1>
        <div class="subtitle">
            Generated on {report_data['generated_at']}<br>
            Execution ID: {report_data['execution_data'].get('execution_id', 'N/A')}
        </div>
    </div>
    """

def _generate_summary_section(execution_data: Dict) -> str:
    """Generate executive summary section"""
    total_tests = execution_data.get('total_tests', 0)
    passed = execution_data.get('passed', 0)
    failed = execution_data.get('failed', 0)
    skipped = execution_data.get('skipped', 0)
    duration = execution_data.get('duration', 0)
    success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
    
    return f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value">{total_tests}</div>
            <div class="metric-label">Total Tests</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color: #28a745;">{passed}</div>
            <div class="metric-label">Passed</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color: #dc3545;">{failed}</div>
            <div class="metric-label">Failed</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color: #ffc107;">{skipped}</div>
            <div class="metric-label">Skipped</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{success_rate:.1f}%</div>
            <div class="metric-label">Success Rate</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{duration:.1f}s</div>
            <div class="metric-label">Total Duration</div>
        </div>
    </div>
    """

def _generate_results_overview(test_results: List[Dict]) -> str:
    """Generate test results overview section"""
    if not test_results:
        return """
        <div class="section">
            <h2 class="section-title">📊 Test Results Overview</h2>
            <p>No test results available.</p>
        </div>
        """
    
    # Create summary table
    table_rows = []
    for result in test_results:
        test_name = result.get('test_name', 'Unknown')
        total = result.get('total_tests', 0)
        passed = result.get('passed', 0)
        failed = result.get('failed', 0)
        success_rate = (passed / total * 100) if total > 0 else 0
        
        table_rows.append(f"""
        <tr>
            <td>{test_name}</td>
            <td>{total}</td>
            <td style="color: #28a745;">{passed}</td>
            <td style="color: #dc3545;">{failed}</td>
            <td>{success_rate:.1f}%</td>
        </tr>
        """)
    
    return f"""
    <div class="section">
        <h2 class="section-title">📊 Test Results Overview</h2>
        <table>
            <thead>
                <tr>
                    <th>Test Suite</th>
                    <th>Total</th>
                    <th>Passed</th>
                    <th>Failed</th>
                    <th>Success Rate</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
    </div>
    """

def _generate_detailed_results(test_results: List[Dict]) -> str:
    """Generate detailed test results section"""
    if not test_results:
        return ""
    
    test_cards = []
    for result in test_results:
        test_cases = result.get('test_cases', [])
        
        for case in test_cases:
            status = case.get('status', 'unknown')
            status_class = f"status-{status}"
            card_class = status
            
            error_html = ""
            if case.get('error'):
                error_html = f"""
                <div class="error-details">
                    <strong>Error:</strong> {case['error']}
                </div>
                """
            
            test_cards.append(f"""
            <div class="test-card {card_class}">
                <div class="test-title">{case.get('title', 'Untitled Test')}</div>
                <div class="test-status {status_class}">{status}</div>
                <p><strong>Duration:</strong> {case.get('duration', 'N/A')}</p>
                {error_html}
            </div>
            """)
    
    return f"""
    <div class="section">
        <h2 class="section-title">🔍 Detailed Test Results</h2>
        <div class="test-grid">
            {''.join(test_cards)}
        </div>
    </div>
    """

def _generate_error_analysis(test_results: List[Dict]) -> str:
    """Generate error analysis section"""
    errors = []
    
    for result in test_results:
        test_cases = result.get('test_cases', [])
        for case in test_cases:
            if case.get('status') == 'failed' and case.get('error'):
                errors.append({
                    'test': case.get('title', 'Unknown'),
                    'error': case.get('error', 'Unknown error'),
                    'timestamp': case.get('end_time', 'Unknown')
                })
    
    if not errors:
        return """
        <div class="section">
            <h2 class="section-title">🚨 Error Analysis</h2>
            <div class="recommendation">
                <strong>Excellent!</strong> No errors were detected during test execution.
            </div>
        </div>
        """
    
    error_rows = []
    for error in errors:
        error_rows.append(f"""
        <tr>
            <td>{error['test']}</td>
            <td>{error['error']}</td>
            <td>{error['timestamp']}</td>
        </tr>
        """)
    
    return f"""
    <div class="section">
        <h2 class="section-title">🚨 Error Analysis</h2>
        <p>Found {len(errors)} error(s) during test execution:</p>
        <table>
            <thead>
                <tr>
                    <th>Test Case</th>
                    <th>Error Message</th>
                    <th>Timestamp</th>
                </tr>
            </thead>
            <tbody>
                {''.join(error_rows)}
            </tbody>
        </table>
    </div>
    """

def _generate_performance_analysis(execution_data: Dict) -> str:
    """Generate performance analysis section"""
    duration = execution_data.get('duration', 0)
    total_tests = execution_data.get('total_tests', 0)
    avg_per_test = (duration / total_tests) if total_tests > 0 else 0
    
    # Performance assessment
    performance_status = "Good"
    performance_color = "#28a745"
    
    if avg_per_test > 30:
        performance_status = "Needs Improvement"
        performance_color = "#dc3545"
    elif avg_per_test > 15:
        performance_status = "Average"
        performance_color = "#ffc107"
    
    return f"""
    <div class="section">
        <h2 class="section-title">⚡ Performance Analysis</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{duration:.1f}s</div>
                <div class="metric-label">Total Execution Time</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{avg_per_test:.1f}s</div>
                <div class="metric-label">Average per Test</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: {performance_color};">{performance_status}</div>
                <div class="metric-label">Performance Rating</div>
            </div>
        </div>
    </div>
    """

def _generate_recommendations(execution_data: Dict, test_results: List[Dict]) -> str:
    """Generate recommendations section"""
    recommendations = []
    
    # Analyze results and generate recommendations
    total_tests = execution_data.get('total_tests', 0)
    failed_tests = execution_data.get('failed', 0)
    success_rate = (execution_data.get('passed', 0) / total_tests * 100) if total_tests > 0 else 0
    
    if failed_tests > 0:
        recommendations.append("🔍 <strong>Review Failed Tests:</strong> Investigate and fix failing test cases to improve overall quality.")
    
    if success_rate < 80:
        recommendations.append("📈 <strong>Improve Test Quality:</strong> Success rate is below 80%. Consider reviewing test data and application stability.")
    
    if execution_data.get('duration', 0) > total_tests * 20:
        recommendations.append("⚡ <strong>Optimize Performance:</strong> Tests are running slower than expected. Consider parallel execution or test optimization.")
    
    if total_tests < 10:
        recommendations.append("📝 <strong>Expand Test Coverage:</strong> Consider adding more test cases to improve coverage.")
    
    if not recommendations:
        recommendations.append("✅ <strong>Great Job!</strong> Your test suite is performing well. Continue monitoring and maintaining quality.")
    
    recommendation_html = "\n".join([f'<div class="recommendation">{rec}</div>' for rec in recommendations])
    
    return f"""
    <div class="section">
        <h2 class="section-title">💡 Recommendations</h2>
        {recommendation_html}
    </div>
    """

def _generate_footer_section() -> str:
    """Generate report footer"""
    return f"""
    <div class="footer">
        <p>Generated by TestRAGic - AI-Powered QA Automation Platform</p>
        <p>Report created on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
    </div>
    """

def export_to_csv(data: List[Dict], filename: str = "test_results.csv") -> str:
    """Export test results to CSV format"""
    if not data:
        return ""
    
    # Flatten the data for CSV export
    flattened_data = []
    
    for result in data:
        test_cases = result.get('test_cases', [])
        test_name = result.get('test_name', 'Unknown')
        
        for case in test_cases:
            flattened_data.append({
                'test_suite': test_name,
                'test_case': case.get('title', 'Unknown'),
                'status': case.get('status', 'Unknown'),
                'start_time': case.get('start_time', ''),
                'end_time': case.get('end_time', ''),
                'error': case.get('error', ''),
                'case_id': case.get('case_id', '')
            })
    
    # Convert to DataFrame and CSV
    df = pd.DataFrame(flattened_data)
    csv_string = df.to_csv(index=False)
    
    return csv_string

def export_to_json(data: Dict, filename: str = "test_results.json") -> str:
    """Export test results to JSON format"""
    return json.dumps(data, indent=2, default=str)

def save_report_to_file(content: str, file_path: str) -> bool:
    """Save report content to file"""
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    except Exception as e:
        st.error(f"Error saving report: {str(e)}")
        return False

def create_download_link(content: str, filename: str, mime_type: str = "text/html") -> str:
    """Create a download link for report content"""
    b64_content = base64.b64encode(content.encode()).decode()
    return f'<a href="data:{mime_type};base64,{b64_content}" download="{filename}">Download {filename}</a>'

def render_report_preview(report_html: str) -> None:
    """Render HTML report preview in Streamlit"""
    st.components.v1.html(report_html, height=800, scrolling=True)

def generate_summary_dashboard(historical_data: List[Dict]) -> Dict:
    """Generate summary dashboard data"""
    if not historical_data:
        return {}
    
    # Calculate trends and metrics
    total_executions = len(historical_data)
    
    # Success rate trend
    success_rates = [data.get('success_rate', 0) for data in historical_data]
    avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
    
    # Test volume trend
    test_volumes = [data.get('total_tests', 0) for data in historical_data]
    total_tests_run = sum(test_volumes)
    
    # Duration analysis
    durations = [data.get('duration', 0) for data in historical_data]
    avg_duration = sum(durations) / len(durations) if durations else 0
    
    return {
        'total_executions': total_executions,
        'avg_success_rate': avg_success_rate,
        'total_tests_run': total_tests_run,
        'avg_duration': avg_duration,
        'success_rates': success_rates,
        'test_volumes': test_volumes,
        'durations': durations
    }

def render_report_generation_interface() -> None:
    """Render report generation interface in Streamlit"""
    st.subheader("📄 Generate Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate HTML Report", use_container_width=True):
            st.info("HTML report generation functionality")
    
    with col2:
        if st.button("📈 Export CSV Data", use_container_width=True):
            st.info("CSV export functionality")
    
    with col3:
        if st.button("📋 Export JSON Data", use_container_width=True):
            st.info("JSON export functionality")
    
    # Report customization options
    with st.expander("🎨 Report Customization"):
        st.selectbox("Report Template", ["Standard", "Executive", "Detailed", "Custom"])
        st.multiselect("Include Sections", 
                      ["Summary", "Test Results", "Error Analysis", "Performance", "Recommendations"],
                      default=["Summary", "Test Results"])
        st.checkbox("Include Charts")
        st.checkbox("Include Screenshots")
        st.text_input("Custom Report Title", placeholder="Enter custom title...")

def create_pdf_report(html_content: str) -> bytes:
    """Convert HTML report to PDF (placeholder - requires additional libraries)"""
    # This would require libraries like weasyprint or pdfkit
    # For now, return empty bytes
    st.warning("PDF generation requires additional libraries (weasyprint/pdfkit)")
    return b""

def schedule_report_generation(schedule_config: Dict) -> bool:
    """Schedule automatic report generation (placeholder)"""
    # This would integrate with task schedulers
    st.info("Report scheduling functionality would be implemented here")
    return True
