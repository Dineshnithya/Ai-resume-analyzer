# app.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from resume_analyzer import extract_skills, compare_skills
from rag import retrieve, answer_with_context, answer_without_context

app = FastAPI(title="AI Resume Analyzer API")

class JDInput(BaseModel):
    job_description: str

class QuestionInput(BaseModel):
    question: str

@app.post("/analyze")
async def analyze_resume(resume: UploadFile = File(...), jd_text: str = ""):
    """
    Endpoint to upload a resume file and optional job description text.
    Returns extracted skills and missing skills vs JD.
    """
    # 1. Read resume text
    try:
        contents = await resume.read()
        text = contents.decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    # 2. Extract skills via LLM
    resume_skills = extract_skills(text)
    result = {"resume_skills": resume_skills}

    # 3. If JD provided, analyze it too
    if jd_text:
        jd_skills = extract_skills(jd_text)
        missing = compare_skills(resume_skills, jd_skills)
        result.update({"jd_skills": jd_skills, "missing_skills": missing})

    return JSONResponse(content=result)

@app.post("/query")
async def query_knowledge(q: QuestionInput):
    """
    Endpoint to query the knowledge base (RAG) or fallback to general LLM.
    Returns the answer and indicates which path was used.
    """
    question = q.question
    hits = retrieve(question, top_k=2)
    if hits and hits[0][1] < 1.0:
        # Found relevant context
        context_docs = "\n---\n".join(doc for doc, _ in hits)
        answer = answer_with_context(question, context_docs)
        source = "RAG"
    else:
        answer = answer_without_context(question)
        source = "LLM"
    return JSONResponse(content={"source": source, "answer": answer})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
