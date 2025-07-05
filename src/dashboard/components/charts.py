"""
This comprehensive charts component provides:

Test Execution Overview - Key metrics and donut charts
Trend Analysis - Historical data visualization with line/bar charts
Category Breakdown - Test distribution across categories
Performance Metrics - Duration analysis with histograms and box plots
Error Analysis - Error patterns, frequency, and timeline
Coverage Heatmap - Test coverage visualization
Browser Compatibility - Multi-browser test results
Real-time Dashboard - Live execution monitoring
Reusable Chart Functions - Modular chart creation helpers
Custom Styling - Dark theme compatible with consistent colors
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta

# Chart color schemes
COLOR_SCHEMES = {
    "success": "#28a745",
    "warning": "#ffc107", 
    "error": "#dc3545",
    "info": "#17a2b8",
    "primary": "#007bff",
    "secondary": "#6c757d"
}

# Default chart styling
DEFAULT_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#fafafa", "size": 12},
    "title": {"font": {"size": 16, "color": "#fafafa"}},
    "xaxis": {"gridcolor": "#444", "color": "#fafafa"},
    "yaxis": {"gridcolor": "#444", "color": "#fafafa"},
    "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": "#fafafa"}}
}

def render_test_execution_overview(execution_data: Dict) -> None:
    """Render test execution overview with key metrics"""
    st.subheader("📊 Test Execution Overview")
    
    # Extract metrics
    total_tests = execution_data.get("total_tests", 0)
    passed_tests = execution_data.get("passed", 0)
    failed_tests = execution_data.get("failed", 0)
    skipped_tests = execution_data.get("skipped", 0)
    
    # Calculate percentages
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tests", total_tests)
    with col2:
        st.metric("Passed", passed_tests, delta=f"{success_rate:.1f}%")
    with col3:
        st.metric("Failed", failed_tests)
    with col4:
        st.metric("Skipped", skipped_tests)
    
    # Create donut chart for test results
    if total_tests > 0:
        fig = create_donut_chart(
            values=[passed_tests, failed_tests, skipped_tests],
            labels=["Passed", "Failed", "Skipped"],
            colors=[COLOR_SCHEMES["success"], COLOR_SCHEMES["error"], COLOR_SCHEMES["secondary"]],
            title="Test Results Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

def render_test_trend_analysis(historical_data: List[Dict]) -> None:
    """Render test execution trends over time"""
    st.subheader("📈 Test Execution Trends")
    
    if not historical_data:
        st.info("No historical data available for trend analysis")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(historical_data)
    
    # Ensure datetime column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        # Generate sample dates
        df['date'] = pd.date_range(
            start=datetime.now() - timedelta(days=len(df)),
            periods=len(df),
            freq='D'
        )
    
    # Create trend charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Success rate trend
        fig_success = create_line_chart(
            data=df,
            x_col='date',
            y_col='success_rate',
            title="Success Rate Trend",
            color=COLOR_SCHEMES["success"]
        )
        st.plotly_chart(fig_success, use_container_width=True)
    
    with col2:
        # Test volume trend
        fig_volume = create_bar_chart(
            data=df,
            x_col='date',
            y_col='total_tests',
            title="Test Volume Trend",
            color=COLOR_SCHEMES["primary"]
        )
        st.plotly_chart(fig_volume, use_container_width=True)
    
    # Combined execution details
    fig_combined = create_stacked_bar_chart(
        data=df,
        x_col='date',
        y_cols=['passed', 'failed', 'skipped'],
        colors=[COLOR_SCHEMES["success"], COLOR_SCHEMES["error"], COLOR_SCHEMES["secondary"]],
        title="Test Execution Details Over Time"
    )
    st.plotly_chart(fig_combined, use_container_width=True)

def render_test_category_breakdown(category_data: Dict) -> None:
    """Render breakdown of tests by category"""
    st.subheader("🏷️ Test Category Breakdown")
    
    if not category_data:
        st.info("No category data available")
        return
    
    # Prepare data
    categories = list(category_data.keys())
    counts = [len(category_data[cat]) for cat in categories]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Horizontal bar chart
        fig_bar = create_horizontal_bar_chart(
            labels=categories,
            values=counts,
            title="Tests by Category",
            color=COLOR_SCHEMES["info"]
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Pie chart
        fig_pie = create_pie_chart(
            labels=categories,
            values=counts,
            title="Category Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

def render_performance_metrics(performance_data: Dict) -> None:
    """Render performance and timing metrics"""
    st.subheader("⚡ Performance Metrics")
    
    # Extract performance data
    avg_duration = performance_data.get("avg_duration", 0)
    min_duration = performance_data.get("min_duration", 0)
    max_duration = performance_data.get("max_duration", 0)
    total_duration = performance_data.get("total_duration", 0)
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Duration", f"{avg_duration:.2f}s")
    with col2:
        st.metric("Min Duration", f"{min_duration:.2f}s")
    with col3:
        st.metric("Max Duration", f"{max_duration:.2f}s")
    with col4:
        st.metric("Total Duration", f"{total_duration:.2f}s")
    
    # Performance distribution
    if "test_durations" in performance_data:
        durations = performance_data["test_durations"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram of test durations
            fig_hist = create_histogram(
                data=durations,
                title="Test Duration Distribution",
                x_label="Duration (seconds)",
                color=COLOR_SCHEMES["info"]
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot for duration analysis
            fig_box = create_box_plot(
                data=durations,
                title="Duration Analysis",
                y_label="Duration (seconds)",
                color=COLOR_SCHEMES["primary"]
            )
            st.plotly_chart(fig_box, use_container_width=True)

def render_error_analysis(error_data: List[Dict]) -> None:
    """Render error analysis and failure patterns"""
    st.subheader("🚨 Error Analysis")
    
    if not error_data:
        st.success("No errors detected! 🎉")
        return
    
    # Error frequency analysis
    error_counts = {}
    error_categories = {}
    
    for error in error_data:
        error_type = error.get("type", "Unknown")
        error_category = error.get("category", "General")
        
        error_counts[error_type] = error_counts.get(error_type, 0) + 1
        error_categories[error_category] = error_categories.get(error_category, 0) + 1
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Error types chart
        if error_counts:
            fig_errors = create_horizontal_bar_chart(
                labels=list(error_counts.keys()),
                values=list(error_counts.values()),
                title="Most Common Errors",
                color=COLOR_SCHEMES["error"]
            )
            st.plotly_chart(fig_errors, use_container_width=True)
    
    with col2:
        # Error categories
        if error_categories:
            fig_categories = create_donut_chart(
                labels=list(error_categories.keys()),
                values=list(error_categories.values()),
                title="Error Categories"
            )
            st.plotly_chart(fig_categories, use_container_width=True)
    
    # Error timeline
    if any("timestamp" in error for error in error_data):
        render_error_timeline(error_data)

def render_test_coverage_heatmap(coverage_data: Dict) -> None:
    """Render test coverage heatmap"""
    st.subheader("🎯 Test Coverage Heatmap")
    
    if not coverage_data:
        st.info("No coverage data available")
        return
    
    # Convert coverage data to matrix format
    features = list(coverage_data.keys())
    test_types = ["functional", "edge_case", "accessibility", "performance"]
    
    # Create matrix
    matrix = []
    for feature in features:
        row = []
        for test_type in test_types:
            coverage = coverage_data[feature].get(test_type, 0)
            row.append(coverage)
        matrix.append(row)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=test_types,
        y=features,
        colorscale='RdYlGn',
        text=matrix,
        texttemplate="%{text}%",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Test Coverage by Feature and Type",
        **DEFAULT_LAYOUT
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_browser_compatibility_chart(browser_data: Dict) -> None:
    """Render browser compatibility test results"""
    st.subheader("🌐 Browser Compatibility")
    
    if not browser_data:
        st.info("No browser compatibility data available")
        return
    
    browsers = list(browser_data.keys())
    success_rates = [browser_data[browser].get("success_rate", 0) for browser in browsers]
    
    # Create radar chart for browser compatibility
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=success_rates,
        theta=browsers,
        fill='toself',
        name='Success Rate (%)',
        line_color=COLOR_SCHEMES["success"]
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        title="Browser Compatibility Success Rates",
        **DEFAULT_LAYOUT
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_real_time_dashboard(metrics: Dict) -> None:
    """Render real-time execution dashboard"""
    st.subheader("🔴 Live Test Execution")
    
    # Real-time metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_test = metrics.get("current_test", "None")
        st.metric("Current Test", current_test)
    
    with col2:
        tests_remaining = metrics.get("tests_remaining", 0)
        st.metric("Tests Remaining", tests_remaining)
    
    with col3:
        estimated_time = metrics.get("estimated_completion", "Unknown")
        st.metric("Est. Completion", estimated_time)
    
    # Progress visualization
    progress = metrics.get("progress", 0)
    st.progress(progress / 100)
    
    # Live chart (would be updated in real-time)
    if "live_data" in metrics:
        live_data = metrics["live_data"]
        fig = create_live_chart(live_data)
        chart_placeholder = st.empty()
        chart_placeholder.plotly_chart(fig, use_container_width=True)

# Chart creation helper functions

def create_donut_chart(values: List[float], labels: List[str], 
                      colors: List[str] = None, title: str = "") -> go.Figure:
    """Create a donut chart"""
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors or px.colors.qualitative.Set3
    )])
    
    fig.update_layout(
        title=title,
        **DEFAULT_LAYOUT
    )
    
    return fig

def create_line_chart(data: pd.DataFrame, x_col: str, y_col: str, 
                     title: str = "", color: str = None) -> go.Figure:
    """Create a line chart"""
    fig = px.line(
        data, 
        x=x_col, 
        y=y_col,
        title=title,
        color_discrete_sequence=[color] if color else None
    )
    
    fig.update_layout(**DEFAULT_LAYOUT)
    return fig

def create_bar_chart(data: pd.DataFrame, x_col: str, y_col: str, 
                    title: str = "", color: str = None) -> go.Figure:
    """Create a bar chart"""
    fig = px.bar(
        data, 
        x=x_col, 
        y=y_col,
        title=title,
        color_discrete_sequence=[color] if color else None
    )
    
    fig.update_layout(**DEFAULT_LAYOUT)
    return fig

def create_stacked_bar_chart(data: pd.DataFrame, x_col: str, y_cols: List[str],
                           colors: List[str] = None, title: str = "") -> go.Figure:
    """Create a stacked bar chart"""
    fig = go.Figure()
    
    for i, col in enumerate(y_cols):
        color = colors[i] if colors and i < len(colors) else None
        fig.add_trace(go.Bar(
            name=col.title(),
            x=data[x_col],
            y=data[col],
            marker_color=color
        ))
    
    fig.update_layout(
        barmode='stack',
        title=title,
        **DEFAULT_LAYOUT
    )
    
    return fig

def create_horizontal_bar_chart(labels: List[str], values: List[float],
                               title: str = "", color: str = None) -> go.Figure:
    """Create a horizontal bar chart"""
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker_color=color or COLOR_SCHEMES["primary"]
    ))
    
    fig.update_layout(
        title=title,
        **DEFAULT_LAYOUT
    )
    
    return fig

def create_pie_chart(labels: List[str], values: List[float], 
                    title: str = "") -> go.Figure:
    """Create a pie chart"""
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        textinfo='label+percent'
    )])
    
    fig.update_layout(
        title=title,
        **DEFAULT_LAYOUT
    )
    
    return fig

def create_histogram(data: List[float], title: str = "", 
                    x_label: str = "", color: str = None) -> go.Figure:
    """Create a histogram"""
    fig = go.Figure(data=[go.Histogram(
        x=data,
        marker_color=color or COLOR_SCHEMES["info"]
    )])
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Frequency",
        **DEFAULT_LAYOUT
    )
    
    return fig

def create_box_plot(data: List[float], title: str = "", 
                   y_label: str = "", color: str = None) -> go.Figure:
    """Create a box plot"""
    fig = go.Figure(data=[go.Box(
        y=data,
        marker_color=color or COLOR_SCHEMES["primary"]
    )])
    
    fig.update_layout(
        title=title,
        yaxis_title=y_label,
        **DEFAULT_LAYOUT
    )
    
    return fig

def create_live_chart(data: List[Dict]) -> go.Figure:
    """Create a live updating chart"""
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.get('timestamp', range(len(df))),
        y=df.get('value', [0] * len(df)),
        mode='lines+markers',
        line=dict(color=COLOR_SCHEMES["success"]),
        name='Live Metrics'
    ))
    
    fig.update_layout(
        title="Live Test Execution Metrics",
        **DEFAULT_LAYOUT
    )
    
    return fig

def render_error_timeline(error_data: List[Dict]) -> None:
    """Render error occurrence timeline"""
    st.subheader("⏰ Error Timeline")
    
    # Prepare timeline data
    timeline_data = []
    for error in error_data:
        if "timestamp" in error:
            timeline_data.append({
                "timestamp": pd.to_datetime(error["timestamp"]),
                "error_type": error.get("type", "Unknown"),
                "severity": error.get("severity", "Medium")
            })
    
    if not timeline_data:
        return
    
    df = pd.DataFrame(timeline_data)
    
    # Create timeline chart
    fig = px.scatter(
        df,
        x="timestamp",
        y="error_type",
        color="severity",
        title="Error Occurrence Timeline",
        color_discrete_map={
            "High": COLOR_SCHEMES["error"],
            "Medium": COLOR_SCHEMES["warning"],
            "Low": COLOR_SCHEMES["info"]
        }
    )
    
    fig.update_layout(**DEFAULT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

def render_custom_metric_card(title: str, value: Any, delta: Any = None, 
                             delta_color: str = "normal") -> None:
    """Render a custom metric card with styling"""
    
    delta_html = ""
    if delta is not None:
        color = COLOR_SCHEMES.get(delta_color, "#fafafa")
        delta_html = f'<div style="color: {color}; font-size: 14px;">Δ {delta}</div>'
    
    card_html = f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    ">
        <h3 style="color: white; margin: 0; font-size: 18px;">{title}</h3>
        <div style="color: white; font-size: 24px; font-weight: bold; margin: 10px 0;">{value}</div>
        {delta_html}
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
