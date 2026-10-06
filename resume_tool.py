# resume_tool.py

from langchain.agents import Tool

def tool_extract_resume(text):
    # calls extract_skills or other analysis
    skills = extract_skills(text)
    return f"Extracted skills: {skills}"

def tool_ask_knowledge(query):
    # calls RAG answer
    answer = answer_with_context(query, retrieve(query, top_k=2))
    return answer

tools = [
    Tool(name="extract_skills", func=tool_extract_resume, description="Extract skills from resume text."),
    Tool(name="ask_knowledge", func=tool_ask_knowledge, description="Answer general queries using RAG.")
]
