"""Cached singletons so models / DB connections are created once per server process."""
import streamlit as st

from src.pipeline import VeriRAGPipeline


@st.cache_resource(show_spinner="Starting VeriRAG...")
def get_pipeline() -> VeriRAGPipeline:
    return VeriRAGPipeline()
