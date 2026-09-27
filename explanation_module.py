import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def explain_topic(topic: str) -> str:
    """
    Explains educational concepts in a simple, beginner-friendly format with fast latency.
    """
    if not topic or not topic.strip():
        return "Please provide a valid topic to explain."

    try:
        load_dotenv(override=True)
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key or api_key == "your_gemini_api_key_here":
            return (
                "⚠️ API Key Required: Please add your Gemini API key in the .env file "
                "(GEMINI_API_KEY=your_key)."
            )
        
        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "models/gemini-3.5-flash-lite")
        gemini_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "max_output_tokens": 1200,
                "temperature": 0.7
            }
        )
        
        prompt = (
            f"Explain the concept of '{topic}' in a simple, clear, and highly engaging way "
            f"for a school student or beginner. Keep it concise, structured with bullet points, and accessible without unnecessary jargon."
        )
        response = gemini_model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        return "No explanation could be generated for this topic."
    except Exception as e:
        return f"Error in Explanation: {str(e)}"
