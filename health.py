"""Simple health check for server monitoring."""
import streamlit as st
st.set_page_config(page_title="Health Check")
st.json({"status": "ok", "stage": "sarsp_langed"})