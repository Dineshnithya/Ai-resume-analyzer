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
# Get skills from resume and job description
candidate_skills = {
    skill.lower().strip()
    for skill in result.candidate_skills
}

required_skills = {
    skill.lower().strip()
    for skill in result.required_skills
}

# Find matching and missing skills
matching_skills = candidate_skills & required_skills
missing_skills = required_skills - candidate_skills

# Calculate match percentage
if required_skills:
    match_percentage = round(
        len(matching_skills) / len(required_skills) * 100
    )
else:
    match_percentage = 0
st.subheader("📊 Job Match")

st.metric(
    "Skill Match",
    f"{match_percentage}%"
)

st.write("### ✅ Matching Skills")

for skill in matching_skills:
    st.write(f"- {skill}")

st.write("### ❌ Missing Skills")

for skill in missing_skills:
    st.write(f"- {skill}")

st.write("### 📌 Summary")
st.write(result.summary)

st.write("### 🛠️ Skills")
for skill in result.candidate_skills:
    st.write(f"- {skill}")

st.write("### 🧩 Job Description Summary")
st.write(result.job_description_summary)

st.write("### 🧑‍💼 Job Role")
st.write(result.job_role)

st.write("### 📋 Responsibilities")
for item in result.job_responsibilities:
    st.write(f"- {item}")

st.write("### ✅ Matching Skills")
for skill in result.matching_skills:
    st.write(f"- {skill}")

st.write("### ⚠️ Missing Skills")
for skill in result.missing_skills:
    st.write(f"- {skill}")

st.write("### 📚 Skills to Learn")
for skill in result.skills_to_learn:
    st.write(f"- {skill}")

st.write("### 📝 Suggestions")
for suggestion in result.suggestions:
    st.write(f"- {suggestion}")