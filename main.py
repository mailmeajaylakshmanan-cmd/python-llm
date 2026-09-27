import os
import asyncio
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
        "model": os.getenv("GEMINI_MODEL", "models/gemini-3.5-flash-lite")
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
    current_model = os.getenv("GEMINI_MODEL", "models/gemini-3.5-flash-lite")
    return {
        "gemini_configured": bool(api_key and api_key != "your_gemini_api_key_here"),
        "active_model": current_model,
        "available_models": [
            {"id": "models/gemini-3.5-flash-lite", "name": "Gemini 3.5 Flash-Lite", "desc": "Sub-second latency (1.2s), high throughput, ideal for real-time tutoring", "tag": "Fastest"},
            {"id": "models/gemini-3.8-flash", "name": "Gemini 3.8 Flash", "desc": "Advanced reasoning & deep explanations", "tag": "High Accuracy"},
            {"id": "models/gemini-3.7-flash", "name": "Gemini 3.7 Flash", "desc": "Hybrid mathematical reasoning", "tag": "Hybrid"},
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
        "active_model": os.getenv("GEMINI_MODEL", "models/gemini-3.5-flash-lite"),
        "audience": payload.audience or "General",
        "tone": payload.tone or "Balanced"
    }


# ==========================================
# 1. Q&A Module Endpoints
# ==========================================
@app.get("/qa")
async def qna_get(question: str = Query(..., description="The student's question")):
    if not question.strip():
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Please provide a question."}
        )
    answer = await asyncio.to_thread(answer_question_with_gemini, question)
    return {"question": question, "answer": answer}


@app.post("/qa")
async def qna_post(payload: Optional[QnARequest] = None, request: Request = None):
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
    answer = await asyncio.to_thread(answer_question_with_gemini, question)
    return {"question": question, "answer": answer}


# ==========================================
# 2. Concept Explanation Module Endpoint
# ==========================================
@app.post("/explain")
async def explain_api(request: Request):
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
    explanation = await asyncio.to_thread(explain_topic, topic)
    return {"topic": topic, "explanation": explanation}


# ==========================================
# 3. Quiz Generation Module Endpoint
# ==========================================
@app.post("/quiz")
async def quiz_api(request: Request):
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
    quiz = await asyncio.to_thread(generate_quiz, text)
    return {"quiz": quiz}


# ==========================================
# 4. Summarization Module Endpoint
# ==========================================
@app.post("/summarize")
async def summarize_api(request: Request):
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
    summary = await asyncio.to_thread(summarize_text, text)
    return {"summary": summary}


# ==========================================
# 5. Learning Path Module Endpoints
# ==========================================
@app.get("/learn/recommendations")
async def learning_recommendation_api(topic: str = Query(..., description="The subject or skill to learn")):
    if not topic.strip():
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Please provide a topic."}
        )
    recommendation = await asyncio.to_thread(get_learning_recommendations, topic)
    return {"topic": topic, "recommendation": recommendation}


@app.post("/learn/recommendations")
async def learning_recommendation_post(request: Request):
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
    recommendation = await asyncio.to_thread(get_learning_recommendations, topic)
    return {"topic": topic, "recommendation": recommendation}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
