import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="CRM System",
    page_icon="📊",
)

# Title of the main page
st.title("AI based CRM System")

# Sidebar message
st.sidebar.success("Select a page above.")

# Introduction page content
st.write("""
# Welcome to the CRM System
This CRM system is designed to help businesses manage their customer relationships effectively. 
It provides features for tracking leads, managing customer interactions, and analyzing customer data.

### Key Features:
- **Lead Scoring**: Lead scoring is a methodology used to rank prospects against a scale that represents the perceived value each lead represents to the organization.
- **Lead Source Identification**: Lead source identification is the process of determining where your leads are coming from, such as online campaigns, referrals, or events
- **Sales Forecasting**: Sales forecasting is the process of estimating future sales. It helps businesses make informed decisions about resource allocation and goal setting.
""")

# Additional content or explanations can be added here

