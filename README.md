# EduGenie: Google Gemini Powered Learning Assistant 🎓✨

EduGenie is an intelligent, lightweight AI-powered educational assistant that simplifies learning through generative AI. Designed for students and learners across all academic levels, EduGenie combines cloud-based LLM reasoning (Google Gemini) with local AI model efficiency.

---

## 🚀 Features & Modules

1. **Ask EduGenie (Q&A)**: Instant, precise answers to academic and conceptual questions (`/qa`).
2. **Concept Explainer**: Simplifies difficult concepts into digestible, jargon-free explanations for school students and beginners (`/explain`).
3. **Interactive Quiz Generator**: Dynamically generates 3 context-aware MCQs with 4 options, automated JSON cleanup, instant client-side grading, and score tracking (`/quiz`).
4. **Smart Summarization**: Condenses lengthy textbook passages and articles into clear revision key points (`/summarize`).
5. **Learning Path Roadmap**: Constructs step-by-step personalized learning paths from Beginner to Advanced with timelines and recommended resources (`/learn/recommendations`).

---

## 🛠️ Architecture & Tech Stack

- **Backend**: FastAPI (Python 3.10+), Uvicorn ASGI Server
- **Frontend**: HTML5, Vanilla CSS, Jinja2 Templating, Asynchronous Fetch API
- **AI Models**:
  - **Google Gemini 1.5 Flash / 1.5 Pro** via `google-generativeai`
  - **LaMini-Flan-T5-783M** (Local CPU inference via Hugging Face `transformers` & `torch`) with seamless Gemini fallback

---

## 📂 Project Structure

```
c:\python-LLM\
├── main.py                  # FastAPI application & route endpoints
├── qna.py                   # Question answering module (Gemini)
├── explanation_module.py    # Concept explanation module (LaMini / Gemini)
├── quiz_module.py           # Quiz generation & JSON parsing module
├── summary_module.py        # Text summarization module
├── learning_path.py         # Learning roadmap module
├── requirements.txt         # Dependencies list
├── .env.example             # Configuration template
├── .env                     # Your environment variables
├── templates/
│   └── index.html           # Modern interactive single-page interface
└── static/
    └── style.css            # Responsive styles & animations
```

---

## ⚡ Quickstart Guide

### 1. Configure Gemini API Key
Open `.env` and set your API key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
```
*(Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey))*

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run EduGenie
```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
Open your browser and visit: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 📡 API Endpoints

| Method | Endpoint | Description | Sample Query / Body |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Web Application UI | - |
| `GET` / `POST` | `/qa` | Question & Answer | `?question=Which is the largest ocean?` |
| `POST` | `/explain` | Concept Explanation | `{"topic": "Quantum Computing"}` |
| `POST` | `/quiz` | MCQ Quiz Generation | `{"text": "Pythagoras Theorem..."}` |
| `POST` | `/summarize` | Text Summarization | `{"text": "The Industrial Revolution..."}` |
| `GET` / `POST` | `/learn/recommendations` | Learning Path Roadmap | `?topic=SQL` |
| `GET` | `/api/status` | Configuration Status | - |
