"""Launch with: python -m streamlit run app.py"""

import os
from io import BytesIO

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

# Read the key before importing/creating the crew; missing secrets stay in the UI.
# app.py — only this block changes
try:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    if not os.environ["GEMINI_API_KEY"].strip():
        raise ValueError("Empty API key")
except (KeyError, StreamlitSecretNotFoundError, ValueError, TypeError):
    st.error('Add a non-empty GEMINI_API_KEY to Streamlit secrets, then restart.')
    st.stop()

from pypdf import PdfReader
from main import run_recruitment_crew


def extract_text(upload) -> str:
    """Read UTF-8 text or a text-based PDF entirely in memory."""
    try:
        data = upload.getvalue()
        if upload.name.lower().endswith(".pdf"):
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                raise ValueError("Upload an unencrypted PDF.")
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            text = data.decode("utf-8-sig")
    except Exception:
        raise ValueError(
            f"Could not read {upload.name}. Use UTF-8 TXT or an unencrypted text PDF."
        ) from None
    if not text.strip():
        raise ValueError(f"{upload.name} has no readable text. Scanned PDFs need OCR first.")
    return text.strip()


st.title("CrewAI HR Recruiter")
st.caption("AI-assisted review. Scores of 60+ receive five interview questions.")
st.caption("Uploaded text is sent to the configured model provider when you click Run.")

mode = st.radio("Job description source", ["Paste text", "Upload file"], horizontal=True)
jd_text = ""
jd_file = None
if mode == "Paste text":
    jd_text = st.text_area("Job description", height=200)
else:
    jd_file = st.file_uploader("Job description", type=["txt", "pdf"], key="jd")

uploads = st.file_uploader(
    "Resumes", type=["txt", "pdf"], accept_multiple_files=True, key="resumes"
)

if st.button("Run", type="primary"):
    try:
        jd = extract_text(jd_file) if jd_file is not None else jd_text.strip()
        if not jd:
            raise ValueError("Paste or upload a job description.")
        if not uploads:
            raise ValueError("Upload at least one resume.")
        resumes = [extract_text(upload) for upload in uploads]
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    try:
        with st.spinner("Sourcing, screening, preparing questions and ranking..."):
            result = run_recruitment_crew(jd, resumes)
    except Exception as exc:
        st.error(
            "Recruitment could not finish. Check your API key, model access, quota "
            "and connection, or retry if the model returned an invalid result."
        )
        st.caption(f"Debug: {type(exc).__name__}: {exc}")  # remove once stable
        st.stop()

    candidates = result["shortlist"]
    st.subheader("Ranked candidates")
    st.dataframe(
        [{key: c[key] for key in ("id", "score", "justification")} for c in candidates],
        hide_index=True,
        width="stretch",
    )
    filenames = {f"candidate_{i}": upload.name for i, upload in enumerate(uploads, 1)}
    for candidate in candidates:
        with st.expander(
            f"{candidate['id']} | {filenames[candidate['id']]} | {candidate['score']}/100"
        ):
            if candidate["questions"]:
                for i, question in enumerate(candidate["questions"], 1):
                    st.write(f"{i}. {question}")
            else:
                st.write("Below 60: no interview questions generated.")
