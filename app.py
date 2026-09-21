"""Launch with: python -m streamlit run app.py"""

import os
import re
from io import BytesIO

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

try:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    if not os.environ["GEMINI_API_KEY"].strip():
        raise ValueError("Empty API key")
except (KeyError, StreamlitSecretNotFoundError, ValueError, TypeError):
    st.error('Add a non-empty GEMINI_API_KEY to Streamlit secrets, then restart.')
    st.stop()

from pypdf import PdfReader
from main import run_recruitment_crew, evaluate_interview_answers, send_decision_email

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


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


def extract_email(resume_text: str) -> str:
    match = EMAIL_RE.search(resume_text)
    return match.group(0) if match else ""


def get_smtp_config() -> dict:
    try:
        return {
            "host": st.secrets["SMTP_HOST"],
            "port": st.secrets["SMTP_PORT"],
            "user": st.secrets["SMTP_USER"],
            "password": st.secrets["SMTP_PASSWORD"],
        }
    except (KeyError, StreamlitSecretNotFoundError):
        return {}


st.title("CrewAI HR Recruiter")
st.caption("Upload your resume and the job description. If you match, you'll get "
           "interview questions; either way, you'll get a decision by email.")

st.session_state.setdefault("stage", "intake")

# ------------------------------------------------------------------- intake
if st.session_state.stage == "intake":
    jd_text = st.text_area("Job description", height=200)
    resume_file = st.file_uploader("Your resume (CV)", type=["txt", "pdf"], key="own_resume")

    resume_text = ""
    detected_email = ""
    if resume_file is not None:
        try:
            resume_text = extract_text(resume_file)
            detected_email = extract_email(resume_text)
        except ValueError as exc:
            st.error(str(exc))

    email = detected_email
    if resume_file is not None and not detected_email:
        email = st.text_input("We couldn't find an email in your CV — enter it here")

    if st.button("Submit application", type="primary"):
        try:
            if not jd_text.strip():
                raise ValueError("Paste the job description.")
            if resume_file is None or not resume_text:
                raise ValueError("Upload your resume.")
            if not email or "@" not in email:
                raise ValueError("A valid email is required (add one to your CV or enter it above).")
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        try:
            with st.spinner("Reviewing your resume against the job description..."):
                result = run_recruitment_crew(jd_text.strip(), [resume_text])
        except Exception as exc:
            st.error("Could not finish the review. Check your connection and retry.")
            st.caption(f"Debug: {type(exc).__name__}: {exc}")  # remove once stable
            st.stop()

        candidate = result["shortlist"][0]
        st.session_state.jd = jd_text.strip()
        st.session_state.resume = resume_text
        st.session_state.email = email
        st.session_state.questions = candidate["questions"]
        st.session_state.answers = []
        st.session_state.qa_index = 0

        if candidate["questions"]:
            st.session_state.matched = True
            st.session_state.stage = "qa"
        else:
            smtp_config = get_smtp_config()
            if smtp_config:
                try:
                    send_decision_email(email, False, smtp_config)
                except Exception as exc:
                    st.warning(f"Could not send the decision email: {exc}")
            else:
                st.warning("SMTP secrets are not configured — email was not sent.")
            st.session_state.matched = False
            st.session_state.passed = False
            st.session_state.stage = "done"
        st.rerun()

# ------------------------------------------------------------------- qa
elif st.session_state.stage == "qa":
    questions = st.session_state.questions
    i = st.session_state.qa_index
    st.subheader(f"Question {i + 1} of {len(questions)}")
    st.write(questions[i])

    answer_mode = st.radio(
        "Answer type", ["Type / paste", "Upload file"], horizontal=True, key=f"mode_{i}"
    )
    if answer_mode == "Type / paste":
        answer = st.text_area("Your answer", key=f"answer_{i}")
    else:
        answer_file = st.file_uploader(
            "Upload your answer (.txt or .pdf)", type=["txt", "pdf"], key=f"file_{i}"
        )
        answer = ""
        if answer_file is not None:
            try:
                answer = extract_text(answer_file)
            except ValueError as exc:
                st.error(str(exc))

    if st.button("Submit answer", type="primary"):
        if not answer.strip():
            st.error("Enter or upload an answer before submitting.")
            st.stop()
        st.session_state.answers.append(answer.strip())
        if i + 1 < len(questions):
            st.session_state.qa_index += 1
            st.rerun()
        else:
            qa_pairs = [
                {"question": q, "answer": a}
                for q, a in zip(questions, st.session_state.answers)
            ]
            try:
                with st.spinner("Evaluating your answers..."):
                    evaluation = evaluate_interview_answers(
                        st.session_state.jd, st.session_state.resume, qa_pairs
                    )
            except Exception as exc:
                st.error("Could not evaluate your answers. Check your connection and retry.")
                st.caption(f"Debug: {type(exc).__name__}: {exc}")  # remove once stable
                st.stop()

            smtp_config = get_smtp_config()
            if smtp_config:
                try:
                    send_decision_email(
                        st.session_state.email, evaluation["passed"], smtp_config
                    )
                except Exception as exc:
                    st.warning(f"Could not send the decision email: {exc}")
            else:
                st.warning("SMTP secrets are not configured — email was not sent.")

            st.session_state.passed = evaluation["passed"]
            st.session_state.stage = "done"
            st.rerun()

# ------------------------------------------------------------------- done
elif st.session_state.stage == "done":
    if not st.session_state.get("matched", True):
        st.warning("Your profile does not match the job requirements.")
    elif st.session_state.passed:
        st.success("Congratulations, you have been selected for a physical interview.")
    else:
        st.info(
            "Thank you for your time. Unfortunately, you have not been "
            "selected for an interview at this time."
        )
    st.caption(f"A copy of this decision was emailed to {st.session_state.email}.")
    if st.button("Start over"):
        for key in ("stage", "jd", "resume", "email", "questions", "answers",
                    "qa_index", "passed", "matched"):
            st.session_state.pop(key, None)
        st.rerun()
