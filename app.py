"""Launch with: python -m streamlit run app.py"""

import os
from io import BytesIO

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

# Read the key before importing/creating the crew; missing secrets stay in the UI.
try:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    if not os.environ["GEMINI_API_KEY"].strip():
        raise ValueError("Empty API key")
except (KeyError, StreamlitSecretNotFoundError, ValueError, TypeError):
    st.error('Add a non-empty GEMINI_API_KEY to Streamlit secrets, then restart.')
    st.stop()

from pypdf import PdfReader
from main import run_recruitment_crew, evaluate_interview_answers, send_decision_email


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
st.caption("AI-assisted review. Scores of 60+ receive five interview questions.")
st.caption("Uploaded text is sent to the configured model provider when you click Run.")

audience = st.radio(
    "Who's using this?",
    ["Recruiter — compare multiple candidates", "Candidate — take the interview"],
    horizontal=True,
)

# ---------------------------------------------------------------- Recruiter mode
if audience == "Recruiter — compare multiple candidates":
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

# ---------------------------------------------------------------- Candidate mode
else:
    st.session_state.setdefault("stage", "intake")

    if st.session_state.stage == "intake":
        jd_text = st.text_area("Job description", height=200)
        resume_file = st.file_uploader("Your resume", type=["txt", "pdf"], key="own_resume")
        email = st.text_input("Your email address")

        if st.button("Check my fit", type="primary"):
            try:
                jd = jd_text.strip()
                if not jd:
                    raise ValueError("Paste the job description.")
                if resume_file is None:
                    raise ValueError("Upload your resume.")
                if "@" not in email:
                    raise ValueError("Enter a valid email address.")
                resume = extract_text(resume_file)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

            try:
                with st.spinner("Reviewing your resume against the job description..."):
                    result = run_recruitment_crew(jd, [resume])
            except Exception as exc:
                st.error("Could not finish the review. Check your connection and retry.")
                st.caption(f"Debug: {type(exc).__name__}: {exc}")  # remove once stable
                st.stop()

            candidate = result["shortlist"][0]
            st.session_state.jd = jd
            st.session_state.resume = resume
            st.session_state.email = email
            st.session_state.questions = candidate["questions"]
            st.session_state.answers = []
            st.session_state.qa_index = 0

            if candidate["questions"]:
                st.session_state.stage = "qa"
            else:
                # Screening score was below 60: reject immediately, no interview stage.
                smtp_config = get_smtp_config()
                if smtp_config:
                    try:
                        send_decision_email(email, False, smtp_config)
                    except Exception as exc:
                        st.warning(f"Could not send the decision email: {exc}")
                else:
                    st.warning("SMTP secrets are not configured — email was not sent.")
                st.session_state.stage = "done"
                st.session_state.passed = False
            st.rerun()

    elif st.session_state.stage == "qa":
        questions = st.session_state.questions
        i = st.session_state.qa_index
        st.subheader(f"Question {i + 1} of {len(questions)}")
        st.write(questions[i])
        answer = st.text_area("Your answer", key=f"answer_{i}")

        if st.button("Submit answer", type="primary"):
            if not answer.strip():
                st.error("Enter an answer before submitting.")
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

                st.session_state.stage = "done"
                st.session_state.passed = evaluation["passed"]
                st.rerun()

    elif st.session_state.stage == "done":
        if st.session_state.passed:
            st.success("Congratulations, you have been selected for an interview.")
        else:
            st.info(
                "Thank you for your time. Unfortunately, you have not been "
                "selected for an interview at this time."
            )
        st.caption(f"A copy of this decision was emailed to {st.session_state.email}.")
        if st.button("Start over"):
            for key in ("stage", "jd", "resume", "email", "questions", "answers",
                        "qa_index", "passed"):
                st.session_state.pop(key, None)
            st.rerun()
