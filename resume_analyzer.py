# resume_analyzer.py

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_skills(text: str) -> list[str]:
    """
    Use an LLM to extract key skills from the resume or JD text.
    """
    system_prompt = "You are an assistant that extracts skills from a resume or job description."
    user_prompt = f"Extract all relevant skills from the following text:\n\"\"\"\n{text}\n\"\"\"\nSkills:"
    
    resp = openai.ChatCompletion.create(
        model="gpt-4o",  # or use another suitable model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0
    )
    skills_text = resp.choices[0].message.content.strip()
    # Parse LLM output (e.g. comma-separated)
    skills = [s.strip() for s in skills_text.split(",") if s.strip()]
    return skills

def compare_skills(resume_skills: list[str], jd_skills: list[str]) -> list[str]:
    """
    Returns the list of skills present in JD but missing in resume (gap analysis).
    """
    missing = [skill for skill in jd_skills if skill.lower() not in map(str.lower, resume_skills)]
    return missing

# Example usage:
# resume_text = "..."; jd_text = "..."
# res_skills = extract_skills(resume_text)
# jd_skills = extract_skills(jd_text)
# missing = compare_skills(res_skills, jd_skills)
