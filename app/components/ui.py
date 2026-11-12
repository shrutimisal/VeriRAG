"""Reusable Streamlit rendering helpers."""
from __future__ import annotations

import html
import re

import streamlit as st

from app.components.styles import PALETTE
from src.citations.verifier import DISCLAIMER
from src.models import AnswerResult, ClaimVerdict
from src.utils.text import MARKER_RE

LABEL_STYLE = {  # label -> (colour, emoji)
    "Supported": (PALETTE["ok"], "✅"),
    "Partially Supported": (PALETTE["warn"], "🟡"),
    "Unsupported": (PALETTE["muted"], "⚪"),
    "Contradicted": (PALETTE["bad"], "❌"),
    "Unverified": ("#94A3B8", "❔"),
}


def card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="vr-card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(str(value))}</div><div class="sub">{html.escape(sub)}</div></div>',
        unsafe_allow_html=True,
    )


def pill(text: str, colour: str) -> str:
    return f'<span class="vr-pill" style="background:{colour}">{html.escape(text)}</span>'


def render_answer_text(result: AnswerResult) -> None:
    """Show the answer, turning [n] markers into badges (red if the number was invented)."""
    valid = {c.ref for c in result.retrieved}

    def repl(m: re.Match) -> str:
        parts = []
        for n in (int(x) for x in m.group(1).split(",")):
            parts.append(f":blue-badge[{n}]" if n in valid else f":red-badge[{n}?]")
        return " ".join(parts)

    st.markdown(MARKER_RE.sub(repl, result.answer))


def render_citations(result: AnswerResult) -> None:
    if not result.citations:
        if result.found:
            st.warning("No valid citations were produced for this answer.")
        return
    st.markdown("**Sources**")
    for c in result.citations:
        with st.expander(f"[{c.ref}] {c.document} — page {c.page}  (similarity {c.score:.2f})"):
            st.markdown(f'<div class="vr-passage">{html.escape(c.passage)}</div>', unsafe_allow_html=True)


def render_retrieved(result: AnswerResult) -> None:
    """All retrieved chunks, cited or not, with their scores."""
    cited = {c.ref for c in result.citations}
    for r in result.retrieved:
        tag = " · cited" if r.ref in cited else ""
        with st.expander(f"#{r.ref} {r.document} — page {r.page} · similarity {r.score:.3f} · distance {r.distance:.3f}{tag}"):
            st.markdown(f'<div class="vr-passage">{html.escape(r.text)}</div>', unsafe_allow_html=True)


def render_verification(verdicts: list[ClaimVerdict]) -> None:
    st.info(DISCLAIMER, icon="ℹ️")
    if not verdicts:
        st.write("No claims to verify.")
        return
    counts: dict[str, int] = {}
    for v in verdicts:
        counts[v.label] = counts.get(v.label, 0) + 1
    st.markdown(
        " ".join(pill(f"{LABEL_STYLE.get(k, LABEL_STYLE['Unverified'])[1]} {k}: {n}", LABEL_STYLE.get(k, LABEL_STYLE["Unverified"])[0])
                 for k, n in counts.items()),
        unsafe_allow_html=True,
    )
    for v in verdicts:
        colour, emoji = LABEL_STYLE.get(v.label, LABEL_STYLE["Unverified"])
        with st.container(border=True):
            st.markdown(f"{emoji} **{v.claim.text}**")
            refs = ", ".join(f"[{r}]" for r in v.claim.refs) or "none"
            st.markdown(pill(v.label, colour) + f' &nbsp; <span class="vr-note">cited: {refs}</span>', unsafe_allow_html=True)
            if v.claim.uncited:
                st.caption("⚠️ This sentence had no citation, so it was checked against all retrieved passages.")
            st.write(v.explanation or "—")
