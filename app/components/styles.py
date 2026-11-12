"""Global CSS: a consistent indigo/teal palette, cards and badges."""
import streamlit as st

PALETTE = {"primary": "#4F46E5", "accent": "#0D9488", "ok": "#16A34A", "warn": "#D97706", "bad": "#DC2626", "muted": "#64748B"}

_CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
h1, h2, h3 {letter-spacing: -0.02em;}
.vr-hero {background: linear-gradient(135deg, #4F46E5 0%, #0D9488 100%); color: #fff; padding: 1.6rem 2rem;
          border-radius: 16px; margin-bottom: 1.2rem;}
.vr-hero h1 {color: #fff; margin: 0; font-size: 2rem;}
.vr-hero p {margin: .3rem 0 0; opacity: .92;}
.vr-card {border: 1px solid rgba(100,116,139,.25); border-radius: 14px; padding: 1rem 1.2rem; height: 100%;
          background: rgba(148,163,184,.07);}
.vr-card .label {font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; color: #64748B;}
.vr-card .value {font-size: 1.8rem; font-weight: 700; line-height: 1.2;}
.vr-card .sub {font-size: .85rem; color: #64748B;}
.vr-pill {display:inline-block; padding: .15rem .65rem; border-radius: 999px; font-size: .78rem; font-weight: 600; color:#fff;}
.vr-passage {border-left: 4px solid #4F46E5; padding: .5rem .9rem; background: rgba(79,70,229,.06);
             border-radius: 0 8px 8px 0; font-size: .92rem; white-space: pre-wrap;}
.vr-note {font-size: .82rem; color: #64748B;}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
