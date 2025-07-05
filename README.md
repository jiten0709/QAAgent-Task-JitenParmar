# 🤖 QAgenie - AI-Powered QA Testing Platform

## 📋 Project Overview

**QAgenie** is an intelligent, end-to-end QA automation platform that revolutionizes how software testing is done. It combines **AI-powered test generation**, **automated test execution**, and **comprehensive reporting** into a single, user-friendly platform.

## 🎯 What It Can Do

### 1. **Intelligent Test Case Generation**

- 📹 **Video-to-Tests**: Upload demo videos and automatically generate test cases.
- 🔗 **YouTube Integration**: Process YouTube how-to videos for test generation.
- 📄 **Document Processing**: Convert requirements docs into executable tests.
- 🤖 **AI-Powered**: Uses RAG (Retrieval-Augmented Generation) + LLM for smart test creation.

### 2. **Automated Test Execution**

- 🎭 **Multi-Browser Testing**: Chrome, Firefox, and Safari support.
- 📱 **Cross-Platform**: Desktop and mobile testing.
- ⚡ **Parallel Execution**: Run tests simultaneously for faster results.
- 🎬 **Rich Artifacts**: Screenshots, videos, and traces on failures.

### 3. **Comprehensive Analytics & Reporting**

- 📊 **Real-time Dashboards**: Live execution monitoring.
- 📈 **Trend Analysis**: Historical performance tracking.
- 📄 **Multiple Export Formats**: JSON, CSV, HTML, and PDF reports.
- 🔍 **Detailed Insights**: Failure analysis and recommendations.

## 🏗️ How It Works

### **System Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                      │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│ │   Upload    │ │  Test Gen   │ │   Results   │            │
│ │   Video     │ │  Dashboard  │ │  Reporting  │            │
│ └─────────────┘ └─────────────┘ └─────────────┘            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Data Ingestion  │ │ Test Generation │ │ Test Execution  │
│ Agent          │─▶│ Agent (RAG+LLM) │─▶│ Agent          │
└─────────────────┘ └─────────────────┘ └─────────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Video/Doc      │ │ Test Cases     │ │ Test Results   │
│ Processing     │ │ (JSON/Markdown) │ │ & Reports      │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### **Core Workflow**

#### **Phase 1: Data Ingestion** 🎬

1. **Video Upload**: User uploads demo videos or provides YouTube URLs.
2. **Content Processing**: AI extracts visual elements, user interactions, and workflows.
3. **Transcript Generation**: Audio-to-text conversion for context understanding.
4. **Chunk Analysis**: Video is segmented into logical test scenarios.

#### **Phase 2: Test Generation** 🤖

1. **RAG Pipeline**: A vector database stores processed content for retrieval.
2. **LLM Analysis**: GPT-4 analyzes video content and generates test scenarios.
3. **Test Case Creation**: Structured test cases with steps, assertions, and priorities are created.
4. **Multi-Format Output**: Test cases are available in JSON, YAML, and Markdown formats.

#### **Phase 3: Test Execution** 🚀

1. **Playwright Integration**: Converts test cases to executable Playwright scripts.
2. **Multi-Browser Execution**: Runs tests across different browsers and devices.
3. **Real-time Monitoring**: Live execution status and progress tracking.
4. **Artifact Collection**: Screenshots, videos, and logs are collected for debugging.

#### **Phase 4: Results & Reporting** 📊

1. **Analytics Dashboard**: Visual representation of test results.
2. **Trend Analysis**: Historical performance and quality metrics.
3. **Report Generation**: Comprehensive reports for stakeholders.
4. **Failure Analysis**: AI-powered insights into test failures.

## 🛠️ Technology Stack

### **Backend Technologies**

- **Python 3.13**: Core application language.
- **LangChain**: RAG pipeline and LLM orchestration.
- **OpenAI GPT-4**: AI-powered test generation.
- **FAISS**: Vector database for content retrieval.
- **Playwright**: Browser automation and testing.
- **FastAPI**: API endpoints (if needed).

### **Frontend & UI**

- **Streamlit**: Interactive web dashboard.
- **Plotly**: Advanced data visualizations.
- **Pandas**: Data manipulation and analysis.
- **Custom CSS**: Enhanced UI styling.

### **AI & ML Components**

- **Computer Vision**: Video frame analysis.
- **NLP**: Text processing and understanding.
- **Speech-to-Text**: Audio transcript generation.
- **Embeddings**: Content vectorization for RAG.

## 📁 Project Structure

```
QAAgent-Task-JitenParmar/
├── src/
│   ├── agents/                    # AI Agents
│   │   ├── data_ingestion.py     # Video/doc processing
│   │   ├── test_generator.py     # AI test generation
│   │   └── test_executor.py      # Test execution
│   ├── utils/                     # Utility functions
│   │   ├── video_processor.py    # Video analysis
│   │   ├── llm_client.py         # LLM integration
│   │   └── playwright_converter.py # Script conversion
│   ├── dashboard/                 # Web interface
│   │   ├── app.py                # Main dashboard
│   │   ├── pages/                # Dashboard pages
│   │   └── components/           # Reusable components
│   ├── data/                     # Data storage
│   │   ├── videos/               # Uploaded videos
│   │   ├── transcripts/          # Generated transcripts
│   │   └── test_cases/           # Generated tests
│   └── tests/                    # Test execution
│       ├── generated/            # Generated test scripts
│       └── results/              # Execution results
├── requirements.txt              # Python dependencies
├── pytest.ini                   # Test configuration
├── conftest.py                  # Pytest setup
├── run_app.py                   # Application entry point
└── README.md                    # Documentation
```

## 🚀 Key Features

### 1. **Multi-Modal Input Support**

- 📹 Video files (MP4, AVI, MOV, MKV)
- 🔗 YouTube URLs
- 📄 Documentation (PDF, Word, Markdown)
- ✍️ Manual test scenario input

### 2. **Intelligent Test Generation**

- 🎯 **Core User Flows**: Essential application workflows.
- 🔄 **Edge Cases**: Boundary conditions and error scenarios.
- 📱 **Cross-Browser**: Chrome, Firefox, Safari compatibility.
- ♿ **Accessibility**: WCAG compliance testing.
- ⚡ **Performance**: Load time and responsiveness tests.

### 3. **Advanced Execution Capabilities**

- 🔄 **Parallel Execution**: Multiple tests simultaneously.
- 📊 **Real-time Monitoring**: Live execution dashboard.
- 🎬 **Rich Artifacts**: Screenshots, videos, and traces.
- 🐛 **Debug Support**: Detailed error reporting.

### 4. **Comprehensive Analytics**

- 📈 **Trend Analysis**: Performance over time.
- 🎯 **Success Metrics**: Pass/fail rates and patterns.
- 🔍 **Failure Analysis**: AI-powered root cause analysis.
- 📊 **Custom Dashboards**: Configurable visualizations.

## 💡 Use Cases

### 1. **QA Teams**

- Accelerate test case creation from requirements.
- Reduce manual testing effort.
- Improve test coverage and quality.

### 2. **Development Teams**

- Quick regression testing after code changes.
- Automated smoke tests for deployments.
- Cross-browser compatibility validation.

### 3. **Product Teams**

- Validate user journeys and workflows.
- Ensure feature functionality across platforms.
- Performance and accessibility compliance.

### 4. **Startups & SMEs**

- Cost-effective QA automation.
- Rapid testing setup for MVPs.
- Scalable testing infrastructure.

## 🎯 Getting Started

### 1. **Setup & Installation**

```bash
# Clone the repository
git clone <repository-url>
cd QAAgent-Task-JitenParmar

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt
playwright install

# Set up environment variables
cp .env.example .env
# Add your OpenAI API key to the .env file
```

### 2. **Run the Application**

```bash
# Start the Streamlit dashboard
streamlit run run_app.py

# Access at http://localhost:8501
```

### 3. **Test the Setup**

```bash
# Run sample tests
pytest src/tests/generated/test_sample.py -v
```

## 🔮 Future Enhancements

- 🤖 **Advanced AI Models**: Integration with Claude, Gemini.
- 🔄 **CI/CD Integration**: GitHub Actions, Jenkins support.
- 📱 **Mobile App Testing**: React Native, Flutter support.
- 🌐 **API Testing**: REST/GraphQL endpoint validation.
- 🎯 **Visual Testing**: UI component regression testing.
- 🔒 **Security Testing**: Vulnerability assessment.
- ☁️ **Cloud Deployment**: AWS, Azure, GCP
