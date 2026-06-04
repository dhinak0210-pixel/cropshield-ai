# ============================================
# llm/chat_handler.py
# Manages conversation history with AI
# ============================================

import logging
from typing import List, Dict, Optional, Any
from llm.llm_client import LLMClient, get_llm_client

logger = logging.getLogger(__name__)


class ChatHandler:
    """
    Manages multi-turn conversation with AI
    about plant diseases
    
    Features:
    - Remembers conversation history
    - Context-aware responses
    - Disease-specific knowledge
    - Suggested questions
    """

    SYSTEM_PROMPT = """
    You are PlantDoc AI, a friendly plant disease 
    expert assistant for farmers and gardeners.
    
    You have access to the current plant disease 
    diagnosis from our AI model.
    
    Your personality:
    - Helpful and encouraging
    - Simple, clear language
    - Practical advice
    - Empathetic to farmer concerns
    
    Rules:
    - Always stay on topic (plant health)
    - Recommend professional help for severe cases
    - Never make up specific product prices
    - Always mention safety for chemical treatments
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        # Resolve LLM client instance
        client = llm_client or get_llm_client()
        # If it's a GeminiClient compatibility wrapper, extract the LLMClient instance
        if hasattr(client, "llm_client"):
            client = client.llm_client
            
        self.llm             = client
        self.conversation    = []
        self.disease_context = {}

    def set_disease_context(
        self,
        plant: str,
        disease: str,
        confidence: float,
        severity: str
    ):
        """Set current disease context for chat"""

        self.disease_context = {
            "plant"     : plant,
            "disease"   : disease,
            "confidence": confidence,
            "severity"  : severity
        }

        # Add context to conversation start
        context_message = f"""
        New plant disease detected:
        - Plant: {plant}
        - Disease: {disease}
        - Confidence: {confidence:.1f}%
        - Severity: {severity}
        
        I'm ready to answer questions about this.
        """

        self.conversation = [{
            "role"   : "assistant",
            "content": context_message
        }]

    def chat(self, user_message: str) -> str:
        """Send message and get AI response"""

        # Add user message to history
        self.conversation.append({
            "role"   : "user",
            "content": user_message
        })

        # Build context-aware prompt
        context = ""
        if self.disease_context:
            context = f"""
            Current diagnosis context:
            Plant: {self.disease_context['plant']}
            Disease: {self.disease_context['disease']}
            Severity: {self.disease_context['severity']}
            
            Conversation so far:
            """

        # Build conversation string
        conv_str = ""
        for msg in self.conversation[-6:]:  # Last 6 messages
            role = "Farmer" if msg["role"] == "user" \
                else "PlantDoc AI"
            conv_str += f"{role}: {msg['content']}\n\n"

        full_prompt = f"""
        {context}
        {conv_str}
        
        PlantDoc AI: [Respond to the farmer's message above]
        """

        # Generate response
        response = self.llm.generate(
            prompt=full_prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.7,
            max_tokens=400
        )

        # Add to conversation history
        self.conversation.append({
            "role"   : "assistant",
            "content": response
        })

        return response

    def get_suggested_questions(self) -> List[str]:
        """Get suggested questions based on disease"""

        if not self.disease_context:
            return [
                "How do I prevent plant diseases?",
                "What are signs of a healthy plant?",
                "When should I water my plants?"
            ]

        disease = self.disease_context.get("disease", "")
        plant   = self.disease_context.get("plant", "")

        return [
            f"Is {disease} contagious to other plants?",
            f"What organic treatment works for {disease}?",
            f"How long does {disease} treatment take?",
            f"Can I still eat {plant} if it has {disease}?",
            "What's the best time to spray treatment?",
            "How do I prevent this next season?"
        ]

    def clear_history(self):
        """Clear conversation history"""
        self.conversation    = []
        self.disease_context = {}

    # ─── BACKWARD COMPATIBILITY METHODS ─────────
    def get_response(self, user_message: str, history: List[Dict[str, Any]]) -> str:
        """Backward-compatible mapping of chat history context."""
        logger.info(f"Re-synchronizing chat history of length {len(history)} for PlantDoc AI.")
        
        # Re-build self.conversation history
        self.conversation = []
        for msg in history:
            self.conversation.append({
                "role": msg["role"],
                "content": msg["content"]
            })
            
        return self.chat(user_message)

    @staticmethod
    def get_default_welcome_message() -> str:
        """Returns a welcoming greeting to initiate conversations."""
        return (
            "🌿 **Welcome to the CropShield AI Agronomy Lab!** \n\n"
            "I am your AI Agronomy Advisor. You can ask me any questions about: \n"
            "- Specific details about your diagnostic report.\n"
            "- Organic or chemical plant care techniques.\n"
            "- Soil amendment, pest control, and watering optimization.\n\n"
            "*How can I assist your crop management today?*"
        )
