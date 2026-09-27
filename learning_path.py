import os
import traceback
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def get_learning_recommendations(topic: str) -> str:
    """
    Generates a personalized, step-by-step learning roadmap for any subject with fast latency.
    """
    if not topic or not topic.strip():
        return "Please provide a valid topic for the roadmap."

    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        return "GEMINI_API_KEY is not set. Please add your key to .env file to generate learning paths."

    prompt = f"""You are an expert AI tutor and educational mentor.
A student wants to learn: {topic}.

Generate a concise, structured, and actionable learning roadmap that includes:
1. Learning Objectives
2. Level 1: Beginner (Foundations, core topics, timeline, exercises)
3. Level 2: Intermediate (Practical applications, deeper concepts, timeline)
4. Level 3: Advanced (Mastery, real-world projects, timeline)
5. Recommended Resources (Tutorials, docs, practice projects)

Format your response in clean Markdown with clear headings and bullet points.
"""
    try:
        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "models/gemini-3.5-flash-lite")
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "max_output_tokens": 1500,
                "temperature": 0.7
            }
        )

        response = model.generate_content(prompt)
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        elif hasattr(response, "parts") and response.parts:
            return response.parts[0].text.strip()
        else:
            return "Could not extract content from Gemini response."
    except Exception as e:
        traceback.print_exc()
        return f"Error occurred in Learning Recommendations: {str(e)}"
