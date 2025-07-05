import pytest
from playwright.sync_api import Page

def test_basic_navigation(page: Page):
    """Basic test to verify Playwright setup"""
    page.goto("https://www.google.com")
    assert page.title() == "Google"
    
def test_search_functionality(page: Page):
    """Test search functionality"""
    page.goto("https://www.google.com")
    page.fill('textarea[name="q"]', "Playwright testing")
    page.press('textarea[name="q"]', "Enter")
    page.wait_for_load_state("networkidle")
    assert "Playwright" in page.title()

@pytest.mark.slow
def test_multiple_tabs(page: Page):
    """Test opening multiple tabs"""
    page.goto("https://www.example.com")
    assert "Example Domain" in page.title()