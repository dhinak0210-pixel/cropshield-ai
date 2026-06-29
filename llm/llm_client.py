# ============================================
# llm/llm_client.py
# Connects to different LLM APIs
# Falls back automatically if one fails
# ============================================

import os
import logging
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Helper to load .env manually or via dotenv across multiple possible locations
def load_env():
    # Load via python-dotenv if possible
    load_dotenv()
    
    # Defensive check for parent directories or subdirectory .env
    paths = [".env", "../.env", "plant-disease-detection/.env"]
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, val = line.split("=", 1)
                            val_clean = val.strip().strip('"').strip("'")
                            os.environ[key.strip()] = val_clean
            except Exception as e:
                pass

    # Try to load keys from Streamlit secrets if running inside streamlit
    try:
        import streamlit as st
        for key in ["GROQ_API_KEY", "GEMINI_API_KEY", "HUGGINGFACE_TOKEN", "GROQ_MODEL", "GEMINI_MODEL"]:
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:
        pass

# Initialize environment keys
load_env()


class LLMClient:
    """
    Universal LLM client that supports:
    - Groq (fastest, free)
    - Gemini (best quality, free)
    - HuggingFace (open source, free)
    - Ollama (local, offline)
    
    Auto-falls back if primary fails
    """

    def __init__(self, provider: str = "groq"):
        self.provider = provider
        self.client   = None
        self._setup_client()

    def _setup_client(self):
        """Initialize LLM client"""

        if self.provider == "groq":
            self._setup_groq()
        elif self.provider == "gemini":
            self._setup_gemini()
        elif self.provider == "ollama":
            self._setup_ollama()
        elif self.provider == "huggingface":
            self._setup_huggingface()
        else:
            raise ValueError(
                f"Unknown provider: {self.provider}\n"
                f"Choose: groq, gemini, ollama, huggingface"
            )

    def _setup_groq(self):
        """Setup Groq client (fastest free option)"""
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key or api_key == "your_groq_api_key_here":
                raise ValueError("GROQ_API_KEY not found in .env")
            self.client = Groq(api_key=api_key)
            self.model  = os.getenv(
                "GROQ_MODEL", "llama-3.3-70b-versatile"
            )
            print(f"✅ Groq client ready: {self.model}")
        except ImportError:
            raise ImportError("Run: pip install groq")

    def _setup_gemini(self):
        """Setup Gemini client using the new google-genai SDK."""
        try:
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key or api_key == "your_gemini_api_key_here":
                raise ValueError("GEMINI_API_KEY not found in .env")
            self.client = genai.Client(api_key=api_key)
            self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            print(f"✅ Gemini client ready: {self.model}")
        except ImportError:
            raise ImportError("Run: pip install google-genai")

    def _setup_ollama(self):
        """Setup Ollama (local, no internet needed)"""
        try:
            import ollama
            self.client = ollama
            self.model  = "llama3"
            print(f"✅ Ollama local client ready")
            print(f"   Make sure Ollama is running!")
        except ImportError:
            raise ImportError("Run: pip install ollama")

    def _setup_huggingface(self):
        """Setup HuggingFace Inference API"""
        try:
            from huggingface_hub import InferenceClient
            token = os.getenv("HUGGINGFACE_TOKEN")
            self.client = InferenceClient(token=token)
            self.model  = "mistralai/Mistral-7B-Instruct-v0.3"
            print(f"✅ HuggingFace client ready")
        except ImportError:
            raise ImportError(
                "Run: pip install huggingface-hub"
            )

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate text from any LLM provider
        
        Args:
            prompt: User message
            system_prompt: AI role/context
            temperature: 0=focused, 1=creative
            max_tokens: Max response length
            
        Returns:
            Generated text string
        """
        try:
            if self.provider == "groq":
                return self._groq_generate(
                    prompt, system_prompt,
                    temperature, max_tokens
                )
            elif self.provider == "gemini":
                return self._gemini_generate(
                    prompt, system_prompt,
                    temperature, max_tokens
                )
            elif self.provider == "ollama":
                return self._ollama_generate(
                    prompt, system_prompt,
                    temperature, max_tokens
                )
            elif self.provider == "huggingface":
                return self._huggingface_generate(
                    prompt, system_prompt,
                    temperature, max_tokens
                )

        except Exception as e:
            # Auto-fallback to simpler response
            print(f"⚠️  LLM error: {e}")
            return self._fallback_response(prompt, e)

    def _groq_generate(
        self, prompt, system_prompt,
        temperature, max_tokens
    ) -> str:
        """Generate using Groq API"""

        messages = []
        if system_prompt:
            messages.append({
                "role"   : "system",
                "content": system_prompt
            })
        messages.append({
            "role"   : "user",
            "content": prompt
        })

        response = self.client.chat.completions.create(
            model       = self.model,
            messages    = messages,
            temperature = temperature,
            max_tokens  = max_tokens
        )
        return response.choices[0].message.content

    def _gemini_generate(
        self, prompt, system_prompt,
        temperature, max_tokens
    ) -> str:
        """Generate using Gemini API (google-genai SDK)."""
        from google.genai import types

        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n"
        full_prompt += prompt

        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        return response.text

    def _ollama_generate(
        self, prompt, system_prompt,
        temperature, max_tokens
    ) -> str:
        """Generate using local Ollama"""

        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt}
            ],
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )
        return response['message']['content']

    def _huggingface_generate(
        self, prompt, system_prompt,
        temperature, max_tokens
    ) -> str:
        """Generate using HuggingFace API"""

        full_prompt = f"{system_prompt}\n\n{prompt}"
        response = self.client.text_generation(
            full_prompt,
            model       = self.model,
            temperature = temperature,
            max_new_tokens = max_tokens
        )
        return response

    def _fallback_response(self, prompt: str, error: Exception) -> str:
        """Return helpful message if LLM fails"""
        return (
            f"⚠️ **LLM Generation Error**: {str(error)}\n\n"
            "Please verify that your API key is correct and you have not exceeded your rate limits. "
            "You can configure your keys in the `.env` file or Streamlit/HuggingFace space secrets."
        )


# ─── GEMINI VISION CLIENT ────────────────────
class GeminiVisionClient:
    """
    Use Gemini to visually analyze plant disease images.
    Uses the new google-genai SDK to avoid protobuf conflicts.
    """

    def __init__(self):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        self._client = genai.Client(api_key=api_key)
        self._model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        print("✅ Gemini Vision client ready")

    def analyze_plant_image(
        self,
        image_path: str,
        predicted_disease: str,
        confidence: float
    ) -> str:
        """
        Let Gemini visually analyze the plant image.
        Returns detailed visual analysis.
        """
        import base64
        from google.genai import types

        with open(image_path, "rb") as f:
            img_bytes = f.read()

        # Detect mime type
        ext = image_path.lower().rsplit(".", 1)[-1]
        mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png", "webp": "image/webp"}
        mime_type = mime_map.get(ext, "image/jpeg")

        prompt = f"""You are an expert plant pathologist.

This image shows a plant leaf. My AI model detected:
Disease: {predicted_disease}
Confidence: {confidence:.1f}%

Please analyze this image and provide:
1. Visual confirmation of what you see
2. Specific symptoms visible in the image
3. Disease stage assessment (early/mid/late)
4. Any other issues you notice
5. Confidence in the diagnosis

Be specific about what you see in the image.
Keep response under 200 words."""

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
                prompt,
            ],
        )
        return response.text


# ─── FACTORY FUNCTION ────────────────────────
def get_llm_client(
    provider: Optional[str] = None
) -> Optional[LLMClient]:
    """
    Get LLM client based on available API keys
    Auto-selects best available option
    """
    # Force reload of env or streamlit secrets to be safe
    load_env()

    if provider:
        try:
            return LLMClient(provider)
        except Exception as e:
            print(f"Failed to load requested LLM provider '{provider}': {e}")
            return None

    def is_valid_key(val: Optional[str], default_placeholder: str) -> bool:
        if not val:
            return False
        val_strip = val.strip()
        return val_strip != "" and val_strip != default_placeholder

    # Auto-select based on available keys
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    if is_valid_key(groq_key, "your_groq_api_key_here"):
        print("🚀 Using Groq (fastest)")
        try:
            return LLMClient("groq")
        except Exception as e:
            print(f"⚠️ Groq client initialization failed: {e}")

    if is_valid_key(gemini_key, "your_gemini_api_key_here"):
        print("🔵 Using Gemini")
        try:
            return LLMClient("gemini")
        except Exception as e:
            print(f"⚠️ Gemini client initialization failed: {e}")

    if is_valid_key(hf_token, "your_hf_token_here"):
        print("🤗 Using HuggingFace")
        try:
            return LLMClient("huggingface")
        except Exception as e:
            print(f"⚠️ HuggingFace client initialization failed: {e}")

    # Fallback to local Ollama only if it is actually reachable
    try:
        import ollama
        client = LLMClient("ollama")
        client.client.list()
        print("💻 Using Ollama (local)")
        return client
    except Exception:
        print("⚠️ Local Ollama not running or python package missing. LLM offline.")

    return None


# ─── BACKWARD COMPATIBILITY CLASS ────────────
class GeminiClient:
    """
    Backward-compatible wrapper.
    Routes generate_text to the best available LLM provider automatically.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.llm_client = get_llm_client()
        if api_key or (model_name and model_name != "gemini-2.5-flash"):
            try:
                from google import genai
                self.llm_client = LLMClient("gemini")
                _api_key = api_key or os.getenv("GEMINI_API_KEY", "")
                self.llm_client.client = genai.Client(api_key=_api_key)
                self.llm_client.model = model_name
            except Exception as e:
                logger.warning(f"Failed to force custom Gemini client: {e}")

    def generate_text(self, prompt: str, system_instruction: Optional[str] = None, history: Optional[List[Dict[str, Any]]] = None) -> str:
        if history:
            chat_context = ""
            for msg in history:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                chat_context += f"{role_label}: {msg['content']}\n"
            full_prompt = f"{chat_context}User: {prompt}"
        else:
            full_prompt = prompt

        return self.llm_client.generate(
            prompt=full_prompt,
            system_prompt=system_instruction or "",
            temperature=0.2
        )
