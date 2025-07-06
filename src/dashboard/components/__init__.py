"""
Dashboard Components Module - Reusable UI components

This module contains reusable Streamlit components:
- Sidebar navigation and notifications
- Chart and visualization components  
- Report generation utilities
- Custom UI elements and layouts
"""

from .sidebar import (
    render_sidebar
)

# from .charts import (
#     render_test_execution_overview,
#     render_test_trend_analysis,
#     render_test_category_breakdown,
#     render_performance_metrics,
#     render_error_analysis,
#     render_test_coverage_heatmap,
#     render_browser_compatibility_chart,
#     render_real_time_dashboard,
#     render_custom_metric_card
# )

# from .reports import (
#     generate_test_execution_report,
#     export_to_csv,
#     export_to_json,
#     render_report_generation_interface,
#     create_download_link,
#     generate_pdf_report,
#     send_email_report
# )

__all__ = [
    # Sidebar components
    "render_sidebar"
    
    # # Chart components
    # "render_test_execution_overview",
    # "render_test_trend_analysis",
    # "render_test_category_breakdown", 
    # "render_performance_metrics",
    # "render_error_analysis",
    # "render_test_coverage_heatmap",
    # "render_browser_compatibility_chart",
    # "render_real_time_dashboard",
    # "render_custom_metric_card",
    
    # # Report components
    # "generate_test_execution_report",
    # "export_to_csv",
    # "export_to_json", 
    # "render_report_generation_interface",
    # "create_download_link",
    # "generate_pdf_report",
    # "send_email_report"
]

# Component configuration
CHART_THEME = "streamlit"
DEFAULT_CHART_HEIGHT = 400
DEFAULT_TABLE_HEIGHT = 300
MAX_EXPORT_ROWS = 10000
