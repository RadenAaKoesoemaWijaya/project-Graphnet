import streamlit as st

def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app with optimized selectors"""
    st.markdown("""
<style>
    :root {
        --primary-color: #1E40AF;
        --secondary-color: #3B82F6;
        --bg-color: #F8FAFC;
        --text-color: #334155;
        --border-color: #E2E8F0;
        --success-color: #10B981;
        --error-color: #EF4444;
        --warning-color: #F59E0B;
        --info-color: #3B82F6;
    }

    /* Main background and text colors */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
    }

    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: var(--primary-color) !important;
        font-weight: 600 !important;
        letter-spacing: -0.025em;
    }

    h1 {
        border-bottom: 2px solid var(--secondary-color);
        padding-bottom: 12px;
        margin-bottom: 24px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid var(--border-color);
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.05);
    }

    [data-testid="stSidebar"] .stMarkdown p {
        color: var(--text-color);
    }

    /* Button styling */
    .stButton>button {
        background-color: #FFFFFF;
        color: var(--primary-color);
        border: 1px solid var(--primary-color);
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .stButton>button:hover {
        background-color: var(--secondary-color);
        color: #FFFFFF;
        border-color: var(--secondary-color);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.3);
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: var(--primary-color) !important;
        font-size: 2.2rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: #64748B !important;
    }

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: var(--secondary-color);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.1);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding: 4px;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #64748B;
        border-radius: 6px;
        border: none;
        padding: 10px 20px;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: bold;
        box-shadow: 0 4px 6px -1px rgba(30, 64, 175, 0.2);
    }

    /* Dataframe styling */
    .dataframe {
        border: 1px solid var(--border-color) !important;
        border-radius: 8px;
        overflow: hidden;
    }

    .dataframe th {
        background-color: #F1F5F9 !important;
        color: var(--primary-color) !important;
        border-bottom: 2px solid var(--border-color) !important;
        font-weight: 600 !important;
    }

    .dataframe td {
        color: var(--text-color) !important;
        background-color: #FFFFFF !important;
        border-bottom: 1px solid #F1F5F9 !important;
    }

    .dataframe tr:nth-child(even) td {
        background-color: var(--bg-color) !important;
    }

    /* Message styling */
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid var(--success-color);
        color: #047857;
        border-radius: 8px;
    }

    .stError {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid var(--error-color);
        color: #B91C1C;
        border-radius: 8px;
    }

    .stWarning {
        background-color: rgba(245, 158, 11, 0.1);
        border: 1px solid var(--warning-color);
        color: #B45309;
        border-radius: 8px;
    }

    .stInfo {
        background-color: rgba(59, 130, 246, 0.1);
        border: 1px solid var(--info-color);
        color: #1D4ED8;
        border-radius: 8px;
    }

    /* Custom containers */
    .highlight-container {
        background-color: #FFFFFF;
        border-left: 4px solid var(--error-color);
        padding: 20px;
        margin: 15px 0;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        color: var(--text-color);
    }

    .results-container {
        background-color: #FFFFFF;
        border-left: 4px solid var(--success-color);
        padding: 20px;
        margin: 15px 0;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        color: var(--text-color);
    }

    /* Input label styling */
    .stSelectbox label, .stRadio label, .stTextInput label,
    .stNumberInput label, .stFileUploader label {
        color: var(--text-color) !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

def custom_container(content, container_type="highlight"):
    """Create a custom styled container for content"""
    if container_type == "highlight":
        st.markdown(f'<div class="highlight-container">{content}</div>', unsafe_allow_html=True)
    elif container_type == "results":
        st.markdown(f'<div class="results-container">{content}</div>', unsafe_allow_html=True)
