import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def summarize_text(text: str) -> str:
    """
    Summarizes long educational text or passages into clear, concise key points for revision.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        return "GEMINI_API_KEY is not set. Please add your key to .env file to enable summarization."

    try:
        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "models/gemini-3.8-flash")
        model = genai.GenerativeModel(model_name=model_name)

        prompt = f"""You are an educational summarizer. Summarize the following educational text into a concise, easy-to-understand summary.
Focus on retaining core facts and concepts while eliminating unnecessary fluff and redundancy. 
Use bullet points and clear sections where appropriate for quick student revision.

Text:
{text}
"""
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        return "No summary could be generated."
    except Exception as e:
        return f"Error in Summary: {str(e)}"
