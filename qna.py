import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def get_configured_model():
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY is not set. Please add your key to the .env file.")
    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "models/gemini-3.5-flash-lite")
    return genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "max_output_tokens": 1024,
            "temperature": 0.7
        }
    )

def answer_question_with_gemini(question: str) -> str:
    """
    Answers academic and general knowledge questions using Google Gemini with fast latency.
    """
    try:
        model = get_configured_model()
        response = model.generate_content(question)
        if response and response.text:
            return response.text.strip()
        return "No response received from the model."
    except Exception as e:
        return f"Error in Q&A: {str(e)}"
