"""
This complete implementation provides:

Comprehensive test generation - Functional, edge cases, accessibility, and performance tests
Context-aware generation - Uses RAG retrieval for video-based context
Multiple output formats - JSON and Markdown reports
Robust error handling - Fallback parsing for malformed responses
Template-based prompts - Structured prompts for consistent outputs
Integration ready - Works with data ingestion and Playwright converter
Detailed logging - Comprehensive error tracking and info logging
"""

import json
import logging
from typing import Dict, List, Optional
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Template Constants
FUNCTIONAL_TEMPLATE = """You are an expert QA engineer specializing in frontend test automation. 
Generate comprehensive functional test cases for the given user flow.

Return ONLY a valid JSON array of test cases with this exact structure:
[
  {
    "title": "Clear, descriptive test title",
    "description": "Detailed test description",
    "steps": ["Step 1", "Step 2", "Step 3"],
    "expected_result": "Expected outcome",
    "priority": "High|Medium|Low",
    "test_data": "Required test data",
    "preconditions": "Setup requirements"
  }
]

Focus on:
- Happy path scenarios
- Critical user journeys
- Form validations
- Navigation flows
- Data submission and retrieval
- User interactions (clicks, inputs, selections)

Make tests specific, actionable, and directly related to the user flow."""

EDGE_CASE_TEMPLATE = """You are an expert QA engineer specializing in edge case testing.
Generate edge cases and boundary condition tests for the given user flow.

Return ONLY a valid JSON array of edge cases with this exact structure:
[
  {
    "title": "Edge case title",
    "description": "What edge condition this tests",
    "steps": ["Step 1", "Step 2"],
    "expected_result": "Expected behavior",
    "priority": "High|Medium|Low",
    "edge_condition": "Specific boundary/edge condition",
    "risk_level": "High|Medium|Low"
  }
]

Focus on:
- Boundary value testing (min/max inputs)
- Invalid data scenarios
- Network failures and timeouts
- Browser compatibility issues
- Concurrent user actions
- Data corruption scenarios
- System resource limitations
- Security edge cases"""

ACCESSIBILITY_TEMPLATE = """You are an accessibility testing expert following WCAG 2.1 guidelines.
Generate accessibility test cases for the given user flow.

Return ONLY a valid JSON array of accessibility tests with this exact structure:
[
  {
    "title": "Accessibility test title",
    "description": "Accessibility requirement being tested",
    "steps": ["Step 1", "Step 2"],
    "expected_result": "Accessible behavior expected",
    "priority": "High|Medium|Low",
    "wcag_guideline": "Relevant WCAG guideline",
    "assistive_technology": "Screen reader|Keyboard navigation|Voice control"
  }
]

Focus on:
- Keyboard navigation
- Screen reader compatibility
- Color contrast and visual accessibility
- Focus management
- Alternative text for images
- Form label associations
- Semantic HTML structure
- ARIA attributes"""

PERFORMANCE_TEMPLATE = """You are a performance testing expert.
Generate performance test scenarios for the given user flow.

Return ONLY a valid JSON array of performance tests with this exact structure:
[
  {
    "title": "Performance test title",
    "description": "Performance aspect being tested",
    "steps": ["Step 1", "Step 2"],
    "expected_result": "Performance criteria",
    "priority": "High|Medium|Low",
    "metric": "Load time|Response time|Memory usage",
    "threshold": "Performance threshold (e.g., < 2 seconds)"
  }
]

Focus on:
- Page load times
- API response times
- Memory consumption
- CPU usage
- Network efficiency
- Large dataset handling
- Concurrent user load
- Mobile performance"""

class TestGeneratorAgent:
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize Test Generator Agent"""
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        self.retriever = None  # Set from data ingestion
        self.test_cases_dir = Path("src/data/test_cases")
        self.test_cases_dir.mkdir(parents=True, exist_ok=True)
        
        # Test case templates
        self.test_templates = {
            "functional": self._get_functional_template(),
            "edge_case": self._get_edge_case_template(),
            "accessibility": self._get_accessibility_template(),
            "performance": self._get_performance_template()
        }
    
    def set_retriever(self, retriever):
        """Set the retriever from data ingestion agent"""
        self.retriever = retriever
        logger.info("Retriever set for context-aware test generation")
    
    def generate_test_cases(self, user_flow: str, context: str = "") -> Dict:
        """Generate comprehensive test cases for a user flow"""
        try:
            logger.info(f"Generating test cases for user flow: {user_flow[:50]}...")
            
            # Get relevant context from video content if retriever is available
            if self.retriever:
                context_docs = self.retriever.get_relevant_documents(user_flow)
                context = "\n".join([doc.page_content for doc in context_docs[:3]])
            
            # Generate different types of test cases
            functional_tests = self._generate_functional_tests(user_flow, context)
            edge_cases = self.create_edge_cases(user_flow, context)
            accessibility_tests = self.add_accessibility_tests(user_flow)
            performance_tests = self._generate_performance_tests(user_flow)
            
            # Combine all test cases
            all_test_cases = {
                "functional": functional_tests,
                "edge_cases": edge_cases,
                "accessibility": accessibility_tests,
                "performance": performance_tests
            }
            
            # Format and save output
            formatted_output = self.format_output(all_test_cases, user_flow)
            
            return {
                "success": True,
                "test_cases": all_test_cases,
                "formatted_output": formatted_output,
                "total_tests": sum(len(tests) for tests in all_test_cases.values())
            }
            
        except Exception as e:
            logger.error(f"Error generating test cases: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _generate_functional_tests(self, user_flow: str, context: str) -> List[Dict]:
        """Generate functional test cases"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.test_templates["functional"]),
            ("human", f"User Flow: {user_flow}\n\nContext from video: {context}")
        ])
        
        response = self.llm(prompt.format_messages())
        
        try:
            # Parse the JSON response
            test_cases = json.loads(response.content)
            if isinstance(test_cases, dict) and "test_cases" in test_cases:
                return test_cases["test_cases"]
            elif isinstance(test_cases, list):
                return test_cases
            else:
                return []
        except json.JSONDecodeError:
            # Fallback parsing if JSON is malformed
            return self._parse_fallback_response(response.content, "functional")
    
    def create_edge_cases(self, base_flow: str, context: str = "") -> List[Dict]:
        """Generate edge cases and boundary conditions"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.test_templates["edge_case"]),
            ("human", f"Base Flow: {base_flow}\n\nContext: {context}")
        ])
        
        response = self.llm(prompt.format_messages())
        
        try:
            edge_cases = json.loads(response.content)
            if isinstance(edge_cases, dict) and "edge_cases" in edge_cases:
                return edge_cases["edge_cases"]
            elif isinstance(edge_cases, list):
                return edge_cases
            else:
                return []
        except json.JSONDecodeError:
            return self._parse_fallback_response(response.content, "edge_case")
    
    def add_accessibility_tests(self, user_flow: str) -> List[Dict]:
        """Add accessibility test scenarios"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.test_templates["accessibility"]),
            ("human", f"User Flow: {user_flow}")
        ])
        
        response = self.llm(prompt.format_messages())
        
        try:
            accessibility_tests = json.loads(response.content)
            if isinstance(accessibility_tests, dict) and "accessibility_tests" in accessibility_tests:
                return accessibility_tests["accessibility_tests"]
            elif isinstance(accessibility_tests, list):
                return accessibility_tests
            else:
                return []
        except json.JSONDecodeError:
            return self._parse_fallback_response(response.content, "accessibility")
    
    def _generate_performance_tests(self, user_flow: str) -> List[Dict]:
        """Generate performance test scenarios"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.test_templates["performance"]),
            ("human", f"User Flow: {user_flow}")
        ])
        
        response = self.llm(prompt.format_messages())
        
        try:
            performance_tests = json.loads(response.content)
            if isinstance(performance_tests, dict) and "performance_tests" in performance_tests:
                return performance_tests["performance_tests"]
            elif isinstance(performance_tests, list):
                return performance_tests
            else:
                return []
        except json.JSONDecodeError:
            return self._parse_fallback_response(response.content, "performance")
    
    def format_output(self, test_cases: Dict, user_flow: str) -> Dict:
        """Format as JSON and Markdown"""
        # Create structured output
        formatted_data = {
            "metadata": {
                "user_flow": user_flow,
                "generated_at": str(Path(__file__).stat().st_mtime),
                "total_tests": sum(len(tests) for tests in test_cases.values())
            },
            "test_suites": test_cases
        }
        
        # Save as JSON
        json_file = self.test_cases_dir / f"test_cases_{hash(user_flow) % 10000}.json"
        with open(json_file, "w") as f:
            json.dump(formatted_data, f, indent=2)
        
        # Generate Markdown report
        markdown_content = self._generate_markdown_report(test_cases, user_flow)
        markdown_file = self.test_cases_dir / f"test_report_{hash(user_flow) % 10000}.md"
        with open(markdown_file, "w") as f:
            f.write(markdown_content)
        
        return {
            "json_file": str(json_file),
            "markdown_file": str(markdown_file),
            "markdown_content": markdown_content
        }
    
    def _generate_markdown_report(self, test_cases: Dict, user_flow: str) -> str:
        """Generate a comprehensive markdown report"""
        markdown = f"""# Test Cases Report
        
## User Flow
{user_flow}

## Summary
- **Total Test Cases**: {sum(len(tests) for tests in test_cases.values())}
- **Functional Tests**: {len(test_cases.get('functional', []))}
- **Edge Cases**: {len(test_cases.get('edge_cases', []))}
- **Accessibility Tests**: {len(test_cases.get('accessibility', []))}
- **Performance Tests**: {len(test_cases.get('performance', []))}

"""
        
        for category, tests in test_cases.items():
            if tests:
                markdown += f"\n## {category.replace('_', ' ').title()} Tests\n\n"
                for i, test in enumerate(tests, 1):
                    markdown += f"### Test {i}: {test.get('title', 'Untitled Test')}\n\n"
                    markdown += f"**Description**: {test.get('description', 'No description')}\n\n"
                    
                    if 'steps' in test:
                        markdown += "**Steps**:\n"
                        for step_num, step in enumerate(test['steps'], 1):
                            markdown += f"{step_num}. {step}\n"
                        markdown += "\n"
                    
                    if 'expected_result' in test:
                        markdown += f"**Expected Result**: {test['expected_result']}\n\n"
                    
                    if 'priority' in test:
                        markdown += f"**Priority**: {test['priority']}\n\n"
                    
                    markdown += "---\n\n"
        
        return markdown
    
    def _parse_fallback_response(self, content: str, test_type: str) -> List[Dict]:
        """Fallback parser for non-JSON responses"""
        # Simple parsing logic for when LLM doesn't return proper JSON
        lines = content.split('\n')
        test_cases = []
        
        current_test = {}
        for line in lines:
            line = line.strip()
            if line.startswith('Title:') or line.startswith('Test:'):
                if current_test:
                    test_cases.append(current_test)
                current_test = {"title": line.split(':', 1)[1].strip(), "type": test_type}
            elif line.startswith('Description:'):
                current_test["description"] = line.split(':', 1)[1].strip()
            elif line.startswith('Expected:'):
                current_test["expected_result"] = line.split(':', 1)[1].strip()
        
        if current_test:
            test_cases.append(current_test)
        
        return test_cases
    
    def _generate_functional_tests(self, user_flow: str, context: str) -> List[Dict]:
        """Generate functional test cases"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", FUNCTIONAL_TEMPLATE),  # Direct reference
            ("human", f"User Flow: {user_flow}\n\nContext from video: {context}")
        ])
        
        response = self.llm(prompt.format_messages())
        
        try:
            test_cases = json.loads(response.content)
            if isinstance(test_cases, dict) and "test_cases" in test_cases:
                return test_cases["test_cases"]
            elif isinstance(test_cases, list):
                return test_cases
            else:
                return []
        except json.JSONDecodeError:
            return self._parse_fallback_response(response.content, "functional")