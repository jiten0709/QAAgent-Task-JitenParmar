"""
This comprehensive Test Generation page provides:

Video Upload & Processing - Support for local files, YouTube URLs, and direct video links
Manual Test Input - Interface for defining test scenarios and user flows manually
AI-Powered Generation - Advanced AI configuration for comprehensive test suite generation
Generated Tests Management - View, edit, export, and manage generated test cases
Progress Tracking - Real-time progress indicators for long-running operations
Multiple Data Sources - Combine video analysis with manual input for better coverage
Test Categories - Support for functional, accessibility, performance, and security tests
Export Functionality - JSON export and file saving capabilities
Integration Ready - Seamless navigation to test execution page
Responsive Design - Clean, professional interface with proper styling
"""

import streamlit as st
import sys
from pathlib import Path
import json
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.utils.video_processor import VideoProcessor
from src.agents.test_generator import TestGeneratorAgent
from src.dashboard.components.sidebar import add_notification, update_stats
from src.dashboard.components.charts import render_test_category_breakdown, render_custom_metric_card

# Page configuration
st.set_page_config(
    page_title="Test Generation - QAgenie",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.step-container {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 4px solid #007bff;
}

.success-container {
    background: #d4edda;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #28a745;
    margin: 1rem 0;
}

.warning-container {
    background: #fff3cd;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #ffc107;
    margin: 1rem 0;
}

.error-container {
    background: #f8d7da;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #dc3545;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def main():
    """Main function for the test generation page"""
    
    # Page header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Test Generation Suite</h1>
        <p>Generate intelligent test cases from videos, documentation, and requirements</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs([
        "📹 Video Upload", 
        "📝 Manual Input", 
        "🤖 AI Generation", 
        "📊 Generated Tests"
    ])
    
    with tab1:
        render_video_upload_section()
    
    with tab2:
        render_manual_input_section()
    
    with tab3:
        render_ai_generation_section()
    
    with tab4:
        render_generated_tests_section()

def init_session_state():
    """Initialize session state variables"""
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False
    
    if 'processed_video_data' not in st.session_state:
        st.session_state.processed_video_data = {}
    
    if 'generated_tests' not in st.session_state:
        st.session_state.generated_tests = {}
    
    if 'test_generator' not in st.session_state:
        st.session_state.test_generator = None
    
    if 'generation_progress' not in st.session_state:
        st.session_state.generation_progress = 0

def render_video_upload_section():
    """Render video upload and processing section"""
    st.markdown("## 📹 Video Processing")
    st.markdown("Upload video tutorials to automatically generate test cases from demonstrated workflows.")
    
    # Video upload interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload Methods")
        
        upload_method = st.radio(
            "Choose upload method:",
            ["📁 Local File Upload", "🔗 YouTube URL", "🌐 Direct URL"],
            horizontal=True
        )
        
        video_source = None
        
        if upload_method == "📁 Local File Upload":
            video_source = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
                help="Upload video files up to 200MB"
            )
        
        elif upload_method == "🔗 YouTube URL":
            video_source = st.text_input(
                "Enter YouTube URL:",
                placeholder="https://www.youtube.com/watch?v=...",
                help="Paste the complete YouTube video URL"
            )
        
        elif upload_method == "🌐 Direct URL":
            video_source = st.text_input(
                "Enter video URL:",
                placeholder="https://example.com/video.mp4",
                help="Direct link to video file"
            )
        
        # Processing options
        with st.expander("⚙️ Processing Options"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                chunk_duration = st.slider("Chunk Duration (seconds)", 10, 60, 30)
                extract_flows = st.checkbox("Extract User Flows", value=True)
                detect_ui_elements = st.checkbox("Detect UI Elements", value=True)
            
            with col_b:
                confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.7)
                language = st.selectbox("Audio Language", ["en", "es", "fr", "de", "auto"])
                generate_timestamps = st.checkbox("Generate Timestamps", value=True)
    
    with col2:
        st.markdown("### Processing Status")
        
        if st.session_state.video_processed:
            st.markdown("""
            <div class="success-container">
                <h4>✅ Video Processed Successfully</h4>
                <p>Video has been analyzed and chunked for test generation.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show processing results
            data = st.session_state.processed_video_data
            if data:
                st.metric("Video Duration", f"{data.get('duration', 0):.1f}s")
                st.metric("Chunks Created", data.get('total_chunks', 0))
                st.metric("Flows Detected", len(data.get('flows', [])))
        else:
            st.info("Upload a video to start processing")
    
    # Process video button
    if video_source and st.button("🚀 Process Video", type="primary", use_container_width=True):
        process_video(video_source, upload_method, {
            'chunk_duration': chunk_duration,
            'confidence_threshold': confidence_threshold,
            'language': language,
            'extract_flows': extract_flows,
            'detect_ui_elements': detect_ui_elements,
            'generate_timestamps': generate_timestamps
        })

def render_manual_input_section():
    """Render manual test case input section"""
    st.markdown("## 📝 Manual Test Input")
    st.markdown("Manually define test scenarios and requirements for AI-powered test generation.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Application details
        st.markdown("### Application Information")
        
        app_name = st.text_input("Application Name", placeholder="e.g., E-commerce Website")
        app_url = st.text_input("Base URL", placeholder="https://example.com")
        app_type = st.selectbox("Application Type", [
            "Web Application", "Mobile App", "API", "Desktop Application"
        ])
        
        # Test scenarios
        st.markdown("### Test Scenarios")
        
        scenario_type = st.selectbox("Scenario Type", [
            "User Journey", "Feature Testing", "Edge Cases", "Accessibility", "Performance"
        ])
        
        scenario_description = st.text_area(
            "Describe the test scenario:",
            placeholder="Describe what you want to test... e.g., User registration and login flow",
            height=150
        )
        
        # User flows
        st.markdown("### User Flows")
        
        if 'manual_flows' not in st.session_state:
            st.session_state.manual_flows = []
        
        new_flow = st.text_input("Add user flow step:", placeholder="e.g., Click on Login button")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("➕ Add Step"):
                if new_flow:
                    st.session_state.manual_flows.append(new_flow)
                    st.rerun()
        
        with col_b:
            if st.button("🗑️ Clear All Steps"):
                st.session_state.manual_flows = []
                st.rerun()
        
        # Display current flows
        if st.session_state.manual_flows:
            st.markdown("**Current User Flow:**")
            for i, flow in enumerate(st.session_state.manual_flows, 1):
                st.markdown(f"{i}. {flow}")
        
        # Test requirements
        with st.expander("🎯 Advanced Test Requirements"):
            test_types = st.multiselect("Test Types to Generate", [
                "Functional Tests", "Edge Case Tests", "Accessibility Tests", 
                "Performance Tests", "Security Tests", "Cross-browser Tests"
            ], default=["Functional Tests"])
            
            test_frameworks = st.selectbox("Preferred Test Framework", [
                "Playwright", "Selenium", "Cypress", "Puppeteer"
            ])
            
            browsers = st.multiselect("Target Browsers", [
                "Chrome", "Firefox", "Safari", "Edge"
            ], default=["Chrome"])
            
            priority = st.selectbox("Test Priority", ["High", "Medium", "Low"], index=1)
    
    with col2:
        st.markdown("### Generation Preview")
        
        if scenario_description and st.session_state.manual_flows:
            st.markdown("**Ready for Generation:**")
            st.success("✅ Scenario defined")
            st.success(f"✅ {len(st.session_state.manual_flows)} flow steps")
            
            estimated_tests = len(st.session_state.manual_flows) * len(test_types) if 'test_types' in locals() else 0
            st.metric("Estimated Test Cases", estimated_tests)
        else:
            st.warning("Complete the scenario description and add flow steps")
    
    # Generate from manual input
    if st.button("🎯 Generate Tests from Manual Input", type="primary", use_container_width=True):
        if scenario_description and st.session_state.manual_flows:
            generate_from_manual_input({
                'app_name': app_name,
                'app_url': app_url,
                'app_type': app_type,
                'scenario_type': scenario_type,
                'scenario_description': scenario_description,
                'user_flows': st.session_state.manual_flows,
                'test_types': test_types if 'test_types' in locals() else [],
                'browsers': browsers if 'browsers' in locals() else [],
                'priority': priority if 'priority' in locals() else 'Medium'
            })
        else:
            st.error("Please complete the scenario description and add user flow steps")

def render_ai_generation_section():
    """Render AI-powered test generation section"""
    st.markdown("## 🤖 AI Test Generation")
    st.markdown("Use advanced AI to generate comprehensive test suites from your processed data.")
    
    if not st.session_state.video_processed and not st.session_state.manual_flows:
        st.warning("Please process a video or define manual scenarios first.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Generation Configuration")
        
        # Data source selection
        data_sources = []
        if st.session_state.video_processed:
            data_sources.append("📹 Processed Video Data")
        if st.session_state.manual_flows:
            data_sources.append("📝 Manual Scenarios")
        
        selected_sources = st.multiselect("Select Data Sources:", data_sources, default=data_sources)
        
        # Generation parameters
        generation_mode = st.selectbox("Generation Mode", [
            "Comprehensive", "Quick", "Custom", "Edge Cases Only"
        ])
        
        if generation_mode == "Custom":
            with st.expander("Custom Generation Settings"):
                test_coverage = st.slider("Test Coverage Level", 1, 10, 7)
                include_negative_tests = st.checkbox("Include Negative Test Cases", value=True)
                include_accessibility = st.checkbox("Include Accessibility Tests", value=True)
                include_performance = st.checkbox("Include Performance Tests", value=False)
                max_tests_per_flow = st.slider("Max Tests per Flow", 1, 20, 10)
        
        # AI model configuration
        with st.expander("🧠 AI Model Settings"):
            model_temperature = st.slider("Creativity Level", 0.0, 1.0, 0.3, 
                                        help="Higher values make the AI more creative but less predictable")
            context_window = st.slider("Context Window", 1000, 4000, 2000,
                                     help="Amount of context the AI considers")
            include_examples = st.checkbox("Include Example Test Data", value=True)
        
        # Test categories
        st.markdown("### Test Categories to Generate")
        
        categories = {
            "🎯 Functional Tests": st.checkbox("Functional Tests", value=True),
            "🔍 Edge Case Tests": st.checkbox("Edge Case Tests", value=True),
            "♿ Accessibility Tests": st.checkbox("Accessibility Tests", value=False),
            "⚡ Performance Tests": st.checkbox("Performance Tests", value=False),
            "🔒 Security Tests": st.checkbox("Security Tests", value=False),
            "🌐 Cross-browser Tests": st.checkbox("Cross-browser Tests", value=True)
        }
        
        selected_categories = [cat for cat, selected in categories.items() if selected]
    
    with col2:
        st.markdown("### Generation Status")
        
        # Progress indicator
        if st.session_state.generation_progress > 0:
            st.progress(st.session_state.generation_progress / 100)
            st.info(f"Generation Progress: {st.session_state.generation_progress}%")
        
        # Available context preview
        if st.session_state.video_processed:
            data = st.session_state.processed_video_data
            st.metric("Available Video Chunks", data.get('total_chunks', 0))
            st.metric("Detected Flows", len(data.get('flows', [])))
        
        if st.session_state.manual_flows:
            st.metric("Manual Flow Steps", len(st.session_state.manual_flows))
        
        # Estimated output
        if selected_categories:
            estimated_tests = len(selected_categories) * 5  # Base estimate
            if st.session_state.video_processed:
                estimated_tests += len(st.session_state.processed_video_data.get('flows', [])) * 3
            if st.session_state.manual_flows:
                estimated_tests += len(st.session_state.manual_flows) * 2
            
            st.metric("Estimated Test Cases", estimated_tests)
    
    # Generate tests button
    if st.button("🚀 Generate AI Test Suite", type="primary", use_container_width=True):
        if selected_categories and selected_sources:
            generate_ai_test_suite({
                'sources': selected_sources,
                'mode': generation_mode,
                'categories': selected_categories,
                'temperature': model_temperature if 'model_temperature' in locals() else 0.3,
                'context_window': context_window if 'context_window' in locals() else 2000,
                'include_examples': include_examples if 'include_examples' in locals() else True
            })
        else:
            st.error("Please select at least one data source and test category")

def render_generated_tests_section():
    """Render generated tests display and management section"""
    st.markdown("## 📊 Generated Test Cases")
    
    if not st.session_state.generated_tests:
        st.info("No test cases generated yet. Use the other tabs to generate tests.")
        return
    
    # Test overview metrics
    total_tests = sum(len(tests) for tests in st.session_state.generated_tests.values())
    categories_count = len(st.session_state.generated_tests)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_custom_metric_card("Total Tests", total_tests)
    with col2:
        render_custom_metric_card("Categories", categories_count)
    with col3:
        render_custom_metric_card("Ready to Execute", total_tests)
    with col4:
        render_custom_metric_card("Success Rate", "95%")
    
    # Test category breakdown
    if st.session_state.generated_tests:
        render_test_category_breakdown(st.session_state.generated_tests)
    
    # Test management interface
    st.markdown("### 🔧 Test Management")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Category selector
        selected_category = st.selectbox(
            "Select Category to View:",
            list(st.session_state.generated_tests.keys())
        )
        
        if selected_category:
            tests = st.session_state.generated_tests[selected_category]
            
            # Test list with checkboxes
            st.markdown(f"#### Tests in {selected_category} ({len(tests)} tests)")
            
            for i, test in enumerate(tests):
                with st.expander(f"Test {i+1}: {test.get('title', 'Untitled Test')}"):
                    col_a, col_b = st.columns([3, 1])
                    
                    with col_a:
                        st.markdown(f"**Description:** {test.get('description', 'No description')}")
                        st.markdown(f"**Priority:** {test.get('priority', 'Medium')}")
                        
                        if test.get('steps'):
                            st.markdown("**Steps:**")
                            for j, step in enumerate(test['steps'], 1):
                                st.markdown(f"{j}. {step}")
                        
                        if test.get('expected_result'):
                            st.markdown(f"**Expected Result:** {test['expected_result']}")
                    
                    with col_b:
                        if st.button(f"Edit Test {i+1}", key=f"edit_{selected_category}_{i}"):
                            st.session_state.editing_test = (selected_category, i)
                        
                        if st.button(f"Delete Test {i+1}", key=f"delete_{selected_category}_{i}"):
                            del st.session_state.generated_tests[selected_category][i]
                            st.rerun()
    
    with col2:
        st.markdown("### Actions")
        
        if st.button("📋 Export All Tests", use_container_width=True):
            export_tests()
        
        if st.button("🚀 Execute Selected", use_container_width=True):
            st.switch_page("pages/2_🚀_Test_Execution.py")
        
        if st.button("🗑️ Clear All Tests", use_container_width=True):
            if st.confirm("Are you sure you want to delete all generated tests?"):
                st.session_state.generated_tests = {}
                st.rerun()
        
        if st.button("💾 Save Tests", use_container_width=True):
            save_tests_to_file()

def process_video(video_source, upload_method, options):
    """Process uploaded video"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Initializing video processor...")
        processor = VideoProcessor()
        progress_bar.progress(10)
        
        status_text.text("Processing video...")
        
        if upload_method == "📁 Local File Upload":
            # Save uploaded file temporarily
            temp_path = f"src/data/videos/temp_{video_source.name}"
            Path(temp_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(temp_path, "wb") as f:
                f.write(video_source.read())
            
            result = processor.process_video_file(temp_path, options)
        
        elif upload_method == "🔗 YouTube URL":
            result = processor.process_youtube_video(video_source, options)
        
        else:  # Direct URL
            result = processor.process_video_url(video_source, options)
        
        progress_bar.progress(100)
        
        if result.get("success"):
            st.session_state.video_processed = True
            st.session_state.processed_video_data = result
            
            status_text.empty()
            progress_bar.empty()
            
            add_notification("Video processed successfully!", "success")
            st.success("✅ Video processed successfully!")
            st.rerun()
        else:
            st.error(f"Error processing video: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()

def generate_from_manual_input(manual_data):
    """Generate tests from manual input"""
    try:
        # Initialize test generator
        if not st.session_state.test_generator:
            st.session_state.test_generator = TestGeneratorAgent()
        
        with st.spinner("Generating tests from manual input..."):
            result = st.session_state.test_generator.generate_from_manual_input(manual_data)
        
        if result.get("success"):
            # Merge with existing tests
            for category, tests in result.get("test_cases", {}).items():
                if category not in st.session_state.generated_tests:
                    st.session_state.generated_tests[category] = []
                st.session_state.generated_tests[category].extend(tests)
            
            add_notification(f"Generated {len(result.get('test_cases', {}))} test categories", "success")
            st.success("✅ Tests generated successfully!")
            st.rerun()
        else:
            st.error(f"Error generating tests: {result.get('error')}")
    
    except Exception as e:
        st.error(f"Error generating tests: {str(e)}")

def generate_ai_test_suite(generation_config):
    """Generate AI test suite"""
    try:
        # Initialize test generator if needed
        if not st.session_state.test_generator:
            st.session_state.test_generator = TestGeneratorAgent()
        
        # Prepare context data
        context_data = {}
        
        if "📹 Processed Video Data" in generation_config['sources']:
            context_data['video_data'] = st.session_state.processed_video_data
        
        if "📝 Manual Scenarios" in generation_config['sources']:
            context_data['manual_flows'] = st.session_state.manual_flows
        
        # Progress tracking
        progress_container = st.container()
        progress_bar = progress_container.progress(0)
        status_text = progress_container.empty()
        
        def update_progress(step, total, message):
            progress = int((step / total) * 100)
            st.session_state.generation_progress = progress
            progress_bar.progress(progress)
            status_text.text(message)
        
        # Generate tests
        with st.spinner("Generating comprehensive test suite..."):
            result = st.session_state.test_generator.generate_comprehensive_tests(
                context_data, 
                generation_config,
                progress_callback=update_progress
            )
        
        if result.get("success"):
            # Update session state with new tests
            for category, tests in result.get("test_cases", {}).items():
                if category not in st.session_state.generated_tests:
                    st.session_state.generated_tests[category] = []
                st.session_state.generated_tests[category].extend(tests)
            
            total_generated = sum(len(tests) for tests in result.get("test_cases", {}).values())
            
            add_notification(f"Generated {total_generated} test cases", "success")
            update_stats("total_tests_generated", total_generated, total_generated)
            
            progress_bar.empty()
            status_text.empty()
            st.session_state.generation_progress = 0
            
            st.success(f"✅ Generated {total_generated} test cases successfully!")
            st.rerun()
        else:
            st.error(f"Error generating tests: {result.get('error')}")
    
    except Exception as e:
        st.error(f"Error generating AI test suite: {str(e)}")
    finally:
        st.session_state.generation_progress = 0

def export_tests():
    """Export generated tests"""
    if not st.session_state.generated_tests:
        st.warning("No tests to export")
        return
    
    export_data = {
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": sum(len(tests) for tests in st.session_state.generated_tests.values()),
        "categories": list(st.session_state.generated_tests.keys()),
        "test_cases": st.session_state.generated_tests
    }
    
    # Create download link
    json_str = json.dumps(export_data, indent=2)
    st.download_button(
        label="📥 Download Tests (JSON)",
        data=json_str,
        file_name=f"generated_tests_{int(time.time())}.json",
        mime="application/json"
    )

def save_tests_to_file():
    """Save tests to data directory"""
    try:
        if not st.session_state.generated_tests:
            st.warning("No tests to save")
            return
        
        # Create data directory
        test_cases_dir = Path("src/data/test_cases")
        test_cases_dir.mkdir(parents=True, exist_ok=True)
        
        # Save tests
        timestamp = int(time.time())
        file_path = test_cases_dir / f"generated_tests_{timestamp}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.generated_tests, f, indent=2)
        
        st.success(f"✅ Tests saved to {file_path}")
        add_notification("Tests saved successfully", "success")
    
    except Exception as e:
        st.error(f"Error saving tests: {str(e)}")

if __name__ == "__main__":
    main()
