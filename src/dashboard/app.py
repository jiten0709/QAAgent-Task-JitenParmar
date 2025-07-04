import datetime
import json
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
from src.agents.data_ingestion import DataIngestionAgent
from src.agents.test_generator import TestGeneratorAgent
from src.agents.test_executor import TestExecutorAgent

def main():
    st.set_page_config(
        page_title="QAgenie - AI QA Agent",
        page_icon="🧩",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🧩 QAgenie - AI QA Agent</div>', unsafe_allow_html=True)
    st.markdown("*Automated end-to-end frontend test case generation, execution, and reporting*")
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/1f77b4/white?text=QAgenie", width=200)
        
        selected = option_menu(
            menu_title="Navigation",
            options=["🎯 Test Generation", "🚀 Test Execution", "📊 Results & Reports", "⚙️ Settings"],
            icons=["target", "rocket", "bar-chart", "gear"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#1f77b4", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#1f77b4"},
            }
        )
    
    # Route to appropriate page
    if selected == "🎯 Test Generation":
        render_test_generation_page()
    elif selected == "🚀 Test Execution":
        render_test_execution_page()
    elif selected == "📊 Results & Reports":
        render_results_page()
    else:
        render_settings_page()

def render_test_generation_page():
    st.header("🎯 Test Case Generation")
    
    # Two column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Video Input")
        
        # Input methods
        input_method = st.radio("Choose input method:", ["YouTube URL", "Upload Video File"])
        
        if input_method == "YouTube URL":
            video_url = st.text_input(
                "Enter YouTube URL:",
                placeholder="https://youtube.com/watch?v=example",
                help="Paste the URL of the Recruter.ai how-to video"
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload video file:",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a video file (max 200MB)"
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            test_categories = st.multiselect(
                "Test Categories to Generate:",
                ["Core User Flows", "Edge Cases", "Cross-browser", "Mobile", "Accessibility", "Performance"],
                default=["Core User Flows", "Edge Cases"]
            )
            
            priority_levels = st.multiselect(
                "Priority Levels:",
                ["Critical", "High", "Medium", "Low"],
                default=["Critical", "High"]
            )
            
            llm_model = st.selectbox(
                "LLM Model:",
                ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                index=0
            )
        
        # Generation button
        if st.button("🚀 Generate Test Cases", type="primary", use_container_width=True):
            if input_method == "YouTube URL" and video_url:
                generate_test_cases_from_url(video_url, test_categories, priority_levels, llm_model)
            elif input_method == "Upload Video File" and uploaded_file:
                generate_test_cases_from_file(uploaded_file, test_categories, priority_levels, llm_model)
            else:
                st.error("Please provide a video URL or upload a video file.")
    
    with col2:
        st.subheader("📊 Generation Statistics")
        
        # Metrics placeholder
        if 'generation_stats' in st.session_state:
            stats = st.session_state.generation_stats
            st.metric("Total Test Cases", stats.get('total_cases', 0))
            st.metric("Core Flows", stats.get('core_flows', 0))
            st.metric("Edge Cases", stats.get('edge_cases', 0))
            st.metric("Processing Time", f"{stats.get('processing_time', 0):.1f}s")
        else:
            st.info("Generate test cases to see statistics")
        
        # Recent generations
        st.subheader("📝 Recent Generations")
        if 'recent_generations' in st.session_state:
            for gen in st.session_state.recent_generations[-5:]:
                st.write(f"• {gen['name']} - {gen['timestamp']}")
        else:
            st.info("No recent generations")

def generate_test_cases_from_url(url, categories, priorities, model):
    """Generate test cases from YouTube URL"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize agents
        data_agent = DataIngestionAgent()
        test_agent = TestGeneratorAgent(model=model)
        
        # Step 1: Process video
        status_text.text("🎬 Processing video...")
        progress_bar.progress(25)
        video_content = data_agent.process_video_url(url)
        
        # Step 2: Generate test cases
        status_text.text("🤖 Generating test cases...")
        progress_bar.progress(50)
        test_cases = test_agent.generate_comprehensive_tests(
            video_content, categories, priorities
        )
        
        # Step 3: Format and save
        status_text.text("💾 Formatting and saving...")
        progress_bar.progress(75)
        formatted_cases = test_agent.format_test_cases(test_cases)
        
        # Step 4: Display results
        status_text.text("✅ Complete!")
        progress_bar.progress(100)
        
        # Show results
        display_generated_test_cases(formatted_cases)
        
        # Update session state
        st.session_state.generation_stats = {
            'total_cases': len(formatted_cases.get('test_cases', [])),
            'core_flows': len([tc for tc in formatted_cases.get('test_cases', []) if tc.get('category') == 'core_flow']),
            'edge_cases': len([tc for tc in formatted_cases.get('test_cases', []) if tc.get('category') == 'edge_case']),
            'processing_time': 45.2  # Replace with actual time
        }
        
    except Exception as e:
        st.error(f"Error generating test cases: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def display_generated_test_cases(test_cases):
    """Display generated test cases in a user-friendly format"""
    st.success("✅ Test cases generated successfully!")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cases", len(test_cases.get('test_cases', [])))
    with col2:
        critical_cases = len([tc for tc in test_cases.get('test_cases', []) if tc.get('priority') == 'critical'])
        st.metric("Critical", critical_cases)
    with col3:
        high_cases = len([tc for tc in test_cases.get('test_cases', []) if tc.get('priority') == 'high'])
        st.metric("High Priority", high_cases)
    with col4:
        st.metric("Test Suites", len(set(tc.get('suite', '') for tc in test_cases.get('test_cases', []))))
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Test Cases", "📄 JSON Format", "📝 Markdown"])
    
    with tab1:
        # Display test cases in a nice format
        for i, tc in enumerate(test_cases.get('test_cases', [])):
            with st.expander(f"TC{i+1:03d}: {tc.get('title', 'Untitled')}", expanded=i < 3):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Description:** {tc.get('description', 'No description')}")
                    st.write(f"**Category:** {tc.get('category', 'Unknown')}")
                    
                    if tc.get('steps'):
                        st.write("**Steps:**")
                        for step in tc['steps']:
                            st.write(f"  {step.get('step', 0)}. {step.get('action', '')} - {step.get('expected', '')}")
                    
                    if tc.get('assertions'):
                        st.write("**Assertions:**")
                        for assertion in tc['assertions']:
                            st.write(f"  • {assertion}")
                
                with col2:
                    priority_color = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
                    st.write(f"**Priority:** {priority_color.get(tc.get('priority', 'low'), '⚪')} {tc.get('priority', 'Low').title()}")
                    st.write(f"**ID:** {tc.get('id', 'Unknown')}")
    
    with tab2:
        st.json(test_cases)
        
        # Download button
        st.download_button(
            label="📥 Download JSON",
            data=json.dumps(test_cases, indent=2),
            file_name=f"test_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with tab3:
        markdown_content = convert_to_markdown(test_cases)
        st.markdown(markdown_content)
        
        # Download button
        st.download_button(
            label="📥 Download Markdown",
            data=markdown_content,
            file_name=f"test_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

def render_test_execution_page():
    st.header("🚀 Test Execution")
    
    # Load available test cases
    test_files = load_available_test_files()
    
    if not test_files:
        st.warning("No test cases found. Please generate test cases first.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📂 Select Test Cases")
        
        selected_files = st.multiselect(
            "Choose test files to execute:",
            test_files,
            default=test_files[:1] if test_files else []
        )
        
        # Execution options
        with st.expander("🔧 Execution Options"):
            browsers = st.multiselect(
                "Target Browsers:",
                ["Chromium", "Firefox", "Safari", "Mobile Chrome", "Mobile Safari"],
                default=["Chromium"]
            )
            
            headless = st.checkbox("Run in headless mode", value=False)
            
            parallel_execution = st.checkbox("Parallel execution", value=True)
            
            capture_options = st.multiselect(
                "Capture on failure:",
                ["Screenshots", "Videos", "Traces", "Logs"],
                default=["Screenshots", "Logs"]
            )
        
        # Execute button
        if st.button("🎬 Execute Tests", type="primary", use_container_width=True):
            if selected_files:
                execute_playwright_tests(selected_files, browsers, headless, parallel_execution, capture_options)
            else:
                st.error("Please select at least one test file.")
    
    with col2:
        st.subheader("📊 Execution Status")
        
        if 'execution_status' in st.session_state:
            status = st.session_state.execution_status
            st.metric("Tests Executed", status.get('executed', 0))
            st.metric("Passed", status.get('passed', 0))
            st.metric("Failed", status.get('failed', 0))
            st.metric("Execution Time", f"{status.get('duration', 0):.1f}s")
        else:
            st.info("No execution status available")
        
        # Real-time execution log
        st.subheader("📝 Execution Log")
        if 'execution_log' in st.session_state:
            log_container = st.container()
            with log_container:
                for log_entry in st.session_state.execution_log[-10:]:
                    st.text(log_entry)
        else:
            st.info("No execution logs available")

def render_results_page():
    st.header("📊 Test Results & Reports")
    
    # Load test results
    results = load_test_results()
    
    if not results:
        st.warning("No test results found. Please execute tests first.")
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tests", results.get('total_tests', 0))
    with col2:
        passed = results.get('passed', 0)
        passed_delta = results.get('passed_delta', 0)
        st.metric("Passed", passed, delta=passed_delta)
    with col3:
        failed = results.get('failed', 0)
        failed_delta = results.get('failed_delta', 0)
        st.metric("Failed", failed, delta=failed_delta)
    with col4:
        success_rate = (passed / (passed + failed) * 100) if (passed + failed) > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Charts
    st.subheader("📈 Test Results Trends")
    
    # Create sample trend data
    trend_data = create_trend_chart_data(results)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Line chart for trends
        fig_line = px.line(
            trend_data, 
            x='date', 
            y=['passed', 'failed'], 
            title='Test Results Over Time',
            labels={'value': 'Number of Tests', 'date': 'Date'}
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    with col2:
        # Pie chart for current results
        fig_pie = px.pie(
            values=[passed, failed],
            names=['Passed', 'Failed'],
            title='Current Test Status',
            color_discrete_map={'Passed': '#00CC96', 'Failed': '#FF6B6B'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Detailed results
    st.subheader("📋 Detailed Test Results")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox("Filter by Status:", ["All", "Passed", "Failed", "Skipped"])
    
    with col2:
        browser_filter = st.selectbox("Filter by Browser:", ["All", "Chromium", "Firefox", "Safari"])
    
    with col3:
        priority_filter = st.selectbox("Filter by Priority:", ["All", "Critical", "High", "Medium", "Low"])
    
    # Display filtered results
    filtered_results = apply_filters(results.get('detailed_results', []), status_filter, browser_filter, priority_filter)
    
    if filtered_results:
        df = pd.DataFrame(filtered_results)
        st.dataframe(df, use_container_width=True)
        
        # Download reports
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            # Generate PDF report (placeholder)
            st.button("📄 Generate PDF Report", help="PDF report generation coming soon!")
    else:
        st.info("No results match the selected filters.")

def render_settings_page():
    st.header("⚙️ Settings")
    
    # API Configuration
    st.subheader("🔑 API Configuration")
    
    with st.expander("OpenAI Settings"):
        openai_key = st.text_input(
            "OpenAI API Key:", 
            type="password",
            help="Enter your OpenAI API key for LLM functionality"
        )
        
        if st.button("🔍 Test API Key"):
            if openai_key:
                test_openai_connection(openai_key)
            else:
                st.error("Please enter an API key")
    
    # Playwright Configuration
    st.subheader("🎭 Playwright Configuration")
    
    with st.expander("Browser Settings"):
        default_browser = st.selectbox("Default Browser:", ["Chromium", "Firefox", "Safari"])
        
        default_viewport = st.selectbox# QA Agent - Complete Implementation Walkthrough

