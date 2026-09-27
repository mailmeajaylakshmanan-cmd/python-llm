import os
import re
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from typing import List, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def clean_json_block(text: str) -> str:
    """
    Cleans markdown code blocks (```json ... ```) from Gemini responses.
    """
    cleaned = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    cleaned = re.sub(r'```\s*', '', cleaned)
    return cleaned.strip()

def generate_quiz(text: str) -> List[Dict[str, Any]]:
    """
    Generates 3 multiple-choice questions (MCQs) with 4 options and the correct answer.
    """
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        return [{"error": "GEMINI_API_KEY is not set. Please add your key to .env file."}]

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

        prompt = f"""You are a quiz generator for students.
From the following topic or passage, create exactly 3 multiple-choice questions. 

Requirements for each question:
- "question": A clear question string.
- "options": A list of exactly 4 distinct plausible options as strings.
- "answer": The correct answer string which MUST EXACTLY match one of the items in the "options" list.

Format your output ONLY as valid JSON array, strictly like this:
[
  {{
    "question": "What is ...?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": "Option A"
  }}
]

Topic or Passage:
{text}
"""
        response = model.generate_content(prompt)
        if not response or not response.text:
            return [{"error": "Empty response from Gemini model."}]

        raw_text = response.text.strip()
        cleaned_text = clean_json_block(raw_text)

        # Parse JSON
        parsed = json.loads(cleaned_text)
        if isinstance(parsed, dict):
            if "quiz" in parsed and isinstance(parsed["quiz"], list):
                parsed = parsed["quiz"]
            elif "questions" in parsed and isinstance(parsed["questions"], list):
                parsed = parsed["questions"]
            else:
                parsed = [parsed]

        return parsed
    except json.JSONDecodeError as jde:
        return [{"error": f"Failed to parse quiz JSON: {str(jde)}"}]
    except Exception as e:
        return [{"error": f"Error in Quiz generation: {str(e)}"}]
