import os
from typing import Optional
from fastapi import FastAPI, Request, Form, Query, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

# Import EduGenie AI Modules
from qna import answer_question_with_gemini
from explanation_module import explain_topic
from quiz_module import generate_quiz
from summary_module import summarize_text
from learning_path import get_learning_recommendations

load_dotenv()

app = FastAPI(
    title="EduGenie: AI-Powered Learning Assistant",
    description="Intelligent educational platform powered by Google Gemini and local AI models",
    version="1.0.0"
)

# Static and Template directories
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Request Schemas
class ExplainRequest(BaseModel):
    topic: str

class QuizRequest(BaseModel):
    text: str

class SummarizeRequest(BaseModel):
    text: str

class LearningPathRequest(BaseModel):
    topic: str

class QnARequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Renders the EduGenie main single-page interface.
    """
    load_dotenv(override=True)
    has_api_key = bool(os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_API_KEY") != "your_gemini_api_key_here")
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"has_api_key": has_api_key}
    )


@app.get("/api/status")
async def api_status():
    """
    Status endpoint to check system configuration.
    """
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY", "")
    is_configured = bool(api_key and api_key != "your_gemini_api_key_here")
    return {
        "status": "online",
        "gemini_configured": is_configured,
        "model": os.getenv("GEMINI_MODEL", "models/gemini-3.8-flash")
    }


class SettingsUpdateRequest(BaseModel):
    model: Optional[str] = None
    audience: Optional[str] = None
    tone: Optional[str] = None


@app.get("/api/settings")
async def get_settings():
    """
    Returns current configuration, active model, and available models list.
    """
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY", "")
    current_model = os.getenv("GEMINI_MODEL", "models/gemini-3.8-flash")
    return {
        "gemini_configured": bool(api_key and api_key != "your_gemini_api_key_here"),
        "active_model": current_model,
        "available_models": [
            {"id": "models/gemini-3.8-flash", "name": "Gemini 3.8 Flash", "desc": "Ultra-fast latency, high accuracy & best for interactive study", "tag": "Recommended"},
            {"id": "models/gemini-3.7-flash", "name": "Gemini 3.7 Flash", "desc": "Hybrid reasoning, fast responses & advanced problem-solving", "tag": "High Intelligence"},
            {"id": "models/gemini-pro-latest", "name": "Gemini Pro Latest", "desc": "Deep reasoning, advanced proofs & complex topics", "tag": "Deep Thinking"},
            {"id": "local_lamini", "name": "LaMini-Flan-T5-783M", "desc": "Local HuggingFace model running on your machine", "tag": "Offline Fallback"}
        ]
    }


@app.post("/api/settings")
async def update_settings(payload: SettingsUpdateRequest):
    """
    Updates active model and preferences for the current session.
    """
    if payload.model:
        os.environ["GEMINI_MODEL"] = payload.model
    return {
        "success": True,
        "active_model": os.getenv("GEMINI_MODEL", "models/gemini-3.8-flash"),
        "audience": payload.audience or "General",
        "tone": payload.tone or "Balanced"
    }



# ==========================================
# 1. Q&A Module Endpoints
# ==========================================
@app.get("/qa")
async def qna_get(question: str = Query(..., description="The student's question")):
    """
    Q&A endpoint via GET method as specified in EduGenie architecture.
    """
    if not question.strip():
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Please provide a question."}
        )
    answer = answer_question_with_gemini(question)
    return {"question": question, "answer": answer}


@app.post("/qa")
async def qna_post(payload: Optional[QnARequest] = None, request: Request = None):
    """
    Q&A endpoint supporting JSON and Form submissions.
    """
    question = ""
    if payload and payload.question:
        question = payload.question
    elif request:
        try:
            body = await request.json()
            question = body.get("question", "")
        except Exception:
            form = await request.form()
            question = form.get("question", "")

    if not question.strip():
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Please provide a question."}
        )
    answer = answer_question_with_gemini(question)
    return {"question": question, "answer": answer}


# ==========================================
# 2. Concept Explanation Module Endpoint
# ==========================================
@app.post("/explain")
async def explain_api(request: Request):
    """
    Concept explanation endpoint using LaMini-Flan-T5 / Gemini fallback.
    """
    topic = ""
    try:
        data = await request.json()
        topic = data.get("topic", "")
    except Exception:
        form = await request.form()
        topic = form.get("topic", "")

    if not topic.strip():
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Please provide a topic."}
        )
    explanation = explain_topic(topic)
    return {"topic": topic, "explanation": explanation}


# ==========================================
# 3. Quiz Generation Module Endpoint
# ==========================================
@app.post("/quiz")
async def quiz_api(request: Request):
    """
    Generates 3 MCQs from passage or topic.
    """
    text = ""
    try:
        data = await request.json()
        text = data.get("text", "")
    except Exception:
        form = await request.form()
        text = form.get("text", "")

    if not text.strip():
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Please provide text or topic for quiz."}
        )
    quiz = generate_quiz(text)
    return {"quiz": quiz}


# ==========================================
# 4. Summarization Module Endpoint
# ==========================================
@app.post("/summarize")
async def summarize_api(request: Request):
    """
    Summarizes long educational text.
    """
    text = ""
    try:
        data = await request.json()
        text = data.get("text", "")
    except Exception:
        form = await request.form()
        text = form.get("text", "")

    if not text.strip():
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Please provide text to summarize."}
        )
    summary = summarize_text(text)
    return {"summary": summary}


# ==========================================
# 5. Learning Path Module Endpoints
# ==========================================
@app.get("/learn/recommendations")
async def learning_recommendation_api(topic: str = Query(..., description="The subject or skill to learn")):
    """
    Generates structured learning path via GET method.
    """
    if not topic.strip():
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Please provide a topic."}
        )
    recommendation = get_learning_recommendations(topic)
    return {"topic": topic, "recommendation": recommendation}


@app.post("/learn/recommendations")
async def learning_recommendation_post(request: Request):
    """
    Generates structured learning path via POST method.
    """
    topic = ""
    try:
        data = await request.json()
        topic = data.get("topic", "")
    except Exception:
        form = await request.form()
        topic = form.get("topic", "")

    if not topic.strip():
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Please provide a topic."}
        )
    recommendation = get_learning_recommendations(topic)
    return {"topic": topic, "recommendation": recommendation}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
