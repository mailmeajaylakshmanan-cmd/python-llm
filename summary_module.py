import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def summarize_text(text: str) -> str:
    """
    Summarizes educational text into concise takeaways and key revision points.
    """
    if not text or not text.strip():
        return "Please provide text to summarize."

    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        return "GEMINI_API_KEY is not set. Please add your key to .env file."

    try:
        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "models/gemini-3.5-flash-lite")
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "max_output_tokens": 1024,
                "temperature": 0.5
            }
        )

        prompt = f"""You are an expert study assistant.
Summarize the following educational text into:
1. Core Takeaway (1-2 sentences)
2. Essential Key Points (3-5 concise bullet points)
3. Key Terminology or Definitions (if applicable)

Text:
{text}
"""
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        return "No summary could be generated."
    except Exception as e:
        return f"Error in Summary: {str(e)}"
