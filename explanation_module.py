import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Global cached handles
_explain_tokenizer = None
_explain_model = None
_local_model_loaded = False
_local_model_attempted = False

def load_local_lamini():
    """
    Attempts to load the local MBZUAI/LaMini-Flan-T5-783M model.
    Checks if local weights are cached or if USE_LOCAL_LAMINI is enabled.
    """
    global _explain_tokenizer, _explain_model, _local_model_loaded, _local_model_attempted
    if _local_model_attempted:
        return _local_model_loaded, _explain_tokenizer, _explain_model

    _local_model_attempted = True
    use_local_env = os.getenv("USE_LOCAL_LAMINI", "false").lower() in ("true", "1", "yes")
    
    if not use_local_env:
        # Default to Gemini for instant cloud responses
        _local_model_loaded = False
        return False, None, None

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # type: ignore
        import torch  # type: ignore
        model_name = "MBZUAI/LaMini-Flan-T5-783M"
        print(f"Loading local model: {model_name}...")
        _explain_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _explain_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _local_model_loaded = True
        print("Local LaMini-Flan-T5 model loaded successfully.")
    except Exception as e:
        print(f"Notice: Local LaMini model not active ({e}). Using Gemini model.")
        _local_model_loaded = False

    return _local_model_loaded, _explain_tokenizer, _explain_model

def explain_topic(topic: str) -> str:
    """
    Explains educational concepts in a simple, beginner-friendly format.
    Uses LaMini-Flan-T5-783M locally when enabled, or Google Gemini.
    """
    if not topic or not topic.strip():
        return "Please provide a valid topic to explain."

    is_loaded, tokenizer, model = load_local_lamini()
    
    # 1. Use local LaMini model if loaded
    if is_loaded and tokenizer and model:
        try:
            input_text = f"Explain the concept of '{topic}' in a simple and clear way for a school student:"
            inputs = tokenizer(input_text, return_tensors="pt")
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True
            )
            explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if explanation:
                return explanation
        except Exception as e:
            print(f"Local model generation encountered an error: {e}. Trying Gemini...")

    # 2. Use Gemini with tailored educational prompt
    try:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key or api_key == "your_gemini_api_key_here":
            return (
                "⚠️ API Key Required: Please add your Gemini API key in the .env file "
                "(GEMINI_API_KEY=your_key)."
            )
        
        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "models/gemini-3.8-flash")
        gemini_model = genai.GenerativeModel(model_name=model_name)
        
        prompt = (
            f"Explain the concept of '{topic}' in a simple, clear, and highly engaging way "
            f"for a school student or beginner. Keep it concise, accessible, and breakdown the core ideas without jargon."
        )
        response = gemini_model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        return "No explanation could be generated for this topic."
    except Exception as e:
        return f"Error in Explanation: {str(e)}"
