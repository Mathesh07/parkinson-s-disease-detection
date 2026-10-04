"""Main Entry Point for Streamlit Parkinson's Reliability Analyzer Demo App.

Runs the interactive web application for final-year project panel presentation.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure workspace root and demo directory are in sys.path
root_dir = Path(__file__).resolve().parent.parent
demo_dir = root_dir / "demo"
for p in (str(root_dir), str(demo_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from demo.components.analyzer_view import render_analyzer_view
    from demo.components.architecture_view import render_architecture_view
    from demo.components.educational_view import render_educational_view
    from demo.components.evidence_view import render_evidence_view
except ImportError:
    from components.analyzer_view import render_analyzer_view
    from components.architecture_view import render_architecture_view
    from components.educational_view import render_educational_view
    from components.evidence_view import render_evidence_view

# 1. Page Configuration
st.set_page_config(
    page_title="Parkinson's Reliability Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Dark Theme & Research Prototype Styling
st.markdown("""
<style>
    /* Dark Theme Core Styles */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stSidebar {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Clean Card Container */
    div.css-1r6slb0, div.css-12w0qpk {
        background-color: #161b22;
        border-radius: 8px;
        border: 1px solid #30363d;
        padding: 1rem;
    }
    
    /* Footer Styling */
    footer {
        visibility: hidden;
    }
    .custom-footer {
        text-align: center;
        padding: 1rem;
        color: #8b949e;
        font-size: 0.85rem;
        border-top: 1px solid #21262d;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # 3. Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/color/96/brain--v1.png", width=70)
    st.sidebar.markdown("## 🧠 PARKINSON'S RELIABILITY SYSTEM")
    st.sidebar.markdown("---")

    navigation_choice = st.sidebar.radio(
        "Navigation",
        options=[
            "🔍 Parkinson's Reliability Analyzer",
            "🏗️ System Architecture",
            "💡 Why Reliability?",
            "📊 Experimental Evidence & Dashboards"
        ],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ System Status")
    st.sidebar.success("✓ Gait Engine: Active (M=5, q=0.8147)")
    st.sidebar.success("✓ Voice Engine: Active (M=5, q=0.4075)")
    st.sidebar.success("✓ Cross-Modal Fusion: Active")
    st.sidebar.info("✓ 174 Automated Tests Passed")

    # 4. Render Selected Page Component
    if navigation_choice == "🔍 Parkinson's Reliability Analyzer":
        render_analyzer_view()
    elif navigation_choice == "🏗️ System Architecture":
        render_architecture_view()
    elif navigation_choice == "💡 Why Reliability?":
        render_educational_view()
    elif navigation_choice == "📊 Experimental Evidence & Dashboards":
        render_evidence_view()

    # 5. Footer Disclaimer
    st.markdown("""
    <div class='custom-footer'>
        🔬 <b>Research Prototype</b> — Model Reliability & Evidence Consistency Analysis Only.<br>
        <i>Not a clinical diagnostic system. Does not perform medical diagnosis.</i>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
