import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()


class ResumeAnalysis(BaseModel):
    summary: str
    candidate_skills: list[str]
    required_skills: list[str]
    matching_skills: list[str]
    missing_skills: list[str]
    job_role: str
    job_description_summary: str
    job_responsibilities: list[str]
    skills_to_learn: list[str]
    suggestions: list[str]


st.title("🤖 AI Resume Analyzer")

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf"]
)


if uploaded_file is not None:

    reader = PdfReader(uploaded_file)

    resume_text = ""

    for page in reader.pages:
        text = page.extract_text()

        if text:
            resume_text += text + "\n"

    st.subheader("📄 Resume")

    st.text_area(
        "Extracted Resume",
        resume_text,
        height=300
    )


    if st.button("🤖 Analyze Resume"):

        llm = ChatOpenAI(
            model="gpt-5.6-luna",
            temperature=0
        )

        structured_llm = llm.with_structured_output(
            ResumeAnalysis
        )

        prompt = f"""
You are an expert technical recruiter and resume analyzer.

Analyze BOTH the candidate's resume AND the job description.

Resume:
{resume_text}

Job Description:
{job_description}

Provide:

1. Professional summary of the candidate
2. Candidate technical skills
3. Job description summary
4. Job role
5. Main responsibilities mentioned in the job description
6. Skills required by the job description
7. Matching skills between resume and job description
8. Missing skills from the candidate
9. Skills the candidate should learn
10. Resume improvement suggestions

Only use information available in the resume and job description.
Do not invent candidate experience or skills.
"""

        result = structured_llm.invoke(prompt)


        st.subheader("🧠 AI Resume Analysis")

        st.write("### 📌 Summary")
        st.write(result.summary)

        st.write("### 🛠️ Skills")
        for skill in result.skills:
            st.write(f"- {skill}")

        st.write("### 🚀 Projects")
        for project in result.projects:
            st.write(f"- {project}")

        st.write("### 💪 Strengths")
        for strength in result.strengths:
            st.write(f"- {strength}")

        st.write("### ⚠️ Weaknesses")
        for weakness in result.weaknesses:
            st.write(f"- {weakness}")

        st.write("### 📚 Missing Skills")
        for skill in result.missing_skills:
            st.write(f"- {skill}")

        st.write("### 📝 Suggestions")
        for suggestion in result.suggestions:
            st.write(f"- {suggestion}")