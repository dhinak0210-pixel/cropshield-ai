# ============================================
# llm/disease_advisor.py
# AI-powered plant disease expert
# Uses LLM to give intelligent advice
# ============================================

import logging
from typing import Dict, List, Tuple
from llm.llm_client import LLMClient

logger = logging.getLogger(__name__)

def parse_class_name(class_name: str) -> Tuple[str, str]:
    """
    Parses a dataset class name into user-friendly crop and disease strings.
    E.g. 'Tomato___Bacterial_spot' -> ('Tomato', 'Bacterial spot')
         'Apple___healthy' -> ('Apple', 'Healthy')
    """
    # Replace triple or double underscores with single divider
    normalized = class_name.replace("___", "::").replace("__", "::")
    
    if "::" in normalized:
        parts = normalized.split("::")
        crop = parts[0].replace("_", " ").title()
        disease = parts[1].replace("_", " ").title()
    else:
        # Fallback
        crop = "Plant"
        disease = class_name.replace("_", " ").title()
        
    # Clean up healthy tag
    if disease.lower() == "healthy":
        disease = "Healthy Leaf Specimen"
        
    return crop, disease


class PlantDiseaseAdvisor:
    """
    AI Plant Disease Expert powered by LLM
    
    Features:
    - Detailed disease explanation
    - Step-by-step treatment plans
    - Prevention strategies
    - Farmer-friendly advice
    - Multi-language support
    """

    # System prompt makes LLM act as plant expert
    SYSTEM_PROMPT = """
    You are an expert agricultural plant pathologist 
    with 20 years of experience helping farmers.
    
    Your expertise:
    - Plant disease identification and treatment
    - Organic and chemical treatment options
    - Crop protection strategies
    - Sustainable farming practices
    
    Your communication style:
    - Clear and simple language (farmer-friendly)
    - Practical, actionable advice
    - Organized with bullet points
    - Include urgency level
    - Always mention safety precautions
    
    Always structure responses with clear sections.
    Be empathetic - farmers depend on their crops.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def explain_disease(
        self,
        disease_name: str,
        plant_name: str,
        confidence: float
    ) -> str:
        """Get detailed AI explanation of disease"""

        prompt = f"""
        A farmer's plant has been diagnosed with:
        
        Plant  : {plant_name}
        Disease: {disease_name}
        AI Confidence: {confidence:.1f}%
        
        Please provide a comprehensive explanation:
        
        1. 🔬 WHAT IS THIS DISEASE?
           (Simple explanation, 2-3 sentences)
        
        2. ⚠️ HOW SERIOUS IS IT?
           (Severity: Low/Medium/High/Critical)
           (Will it spread? How fast?)
        
        3. 🔍 WHAT TO LOOK FOR:
           (3-4 specific visual symptoms)
        
        4. 📊 ECONOMIC IMPACT:
           (How much crop loss if untreated?)
        
        5. ⏰ URGENCY:
           (How quickly must farmer act?)
        
        Keep the total response under 300 words.
        Use simple language a farmer can understand.
        """

        return self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.3,  # Lower = more factual
            max_tokens=500
        )

    def get_treatment_plan(
        self,
        disease_name: str,
        plant_name: str,
        severity: str = "medium"
    ) -> str:
        """Generate step-by-step treatment plan"""

        prompt = f"""
        Create a detailed treatment plan for:
        
        Plant   : {plant_name}
        Disease : {disease_name}
        Severity: {severity}
        
        Provide treatment in this format:
        
        🚨 IMMEDIATE ACTIONS (Do within 24 hours):
        - Step 1: ...
        - Step 2: ...
        
        💊 TREATMENT OPTIONS:
        
        Option A - Organic/Natural:
        - Product/Method: ...
        - How to apply: ...
        - How often: ...
        - Cost estimate: Low/Medium/High
        
        Option B - Chemical Treatment:
        - Recommended fungicide/pesticide: ...
        - Dosage: ...
        - Application method: ...
        - Safety precautions: ...
        - Waiting period before harvest: ...
        
        📅 TREATMENT SCHEDULE:
        Week 1: ...
        Week 2: ...
        Week 3: ...
        
        ⚠️ WARNING SIGNS (when to get expert help):
        - ...
        
        Keep under 400 words. Be very specific.
        """

        return self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=700
        )

    def get_prevention_tips(
        self,
        disease_name: str,
        plant_name: str
    ) -> str:
        """Get prevention strategies"""

        prompt = f"""
        Provide prevention strategies for:
        Plant  : {plant_name}
        Disease: {disease_name}
        
        Structure your response as:
        
        🌱 BEFORE PLANTING:
        - Tip 1
        - Tip 2
        - Tip 3
        
        🌿 DURING GROWING SEASON:
        - Tip 1
        - Tip 2
        - Tip 3
        
        💧 WATERING AND NUTRITION:
        - Best practices
        
        🧹 GARDEN HYGIENE:
        - Cleanup practices
        
        🌦️ WEATHER MONITORING:
        - Conditions to watch
        
        📋 RESISTANT VARIETIES:
        - Better plant varieties to consider
        
        Keep under 300 words.
        """

        return self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=500
        )

    def answer_question(
        self,
        question: str,
        disease_context: Dict
    ) -> str:
        """Answer farmer's specific question about disease"""

        context = f"""
        Current plant disease context:
        - Plant  : {disease_context.get('plant', 'Unknown')}
        - Disease: {disease_context.get('disease', 'Unknown')}
        - Severity: {disease_context.get('severity', 'Unknown')}
        """

        prompt = f"""
        {context}
        
        Farmer's question: {question}
        
        Answer this specific question clearly and practically.
        - Be direct and helpful
        - Use simple language
        - If you need more info to answer properly, ask
        - Keep under 200 words
        """

        return self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.5,
            max_tokens=400
        )

    def generate_full_report(
        self,
        disease_name: str,
        plant_name: str,
        confidence: float,
        severity: str
    ) -> str:
        """Generate complete farm report"""

        prompt = f"""
        Generate a complete plant disease assessment report:
        
        ════════════════════════════════════
        PLANT DISEASE ASSESSMENT REPORT
        ════════════════════════════════════
        
        Plant    : {plant_name}
        Disease  : {disease_name}
        Confidence: {confidence:.1f}%
        Severity : {severity}
        
        Please write a complete professional report with:
        
        1. EXECUTIVE SUMMARY (2-3 sentences)
        
        2. DISEASE OVERVIEW
           - What it is
           - Cause (fungal/bacterial/viral)
           - How it spreads
        
        3. OBSERVED SYMPTOMS
           - Visual symptoms to look for
           - Progression stages
        
        4. IMMEDIATE RECOMMENDATIONS
           - Actions for next 24-48 hours
        
        5. TREATMENT PLAN
           - Week by week plan
           - Products to use
        
        6. PREVENTION FOR NEXT SEASON
           - 5 key prevention tips
        
        7. ESTIMATED CROP LOSS (if untreated)
           - Percentage estimate
        
        8. WHEN TO CONSULT EXPERT
           - Warning signs
        
        Format professionally. Use clear headings.
        This report will be saved as PDF for the farmer.
        """

        return self.llm.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=1200
        )

    def translate_advice(
        self,
        text: str,
        language: str
    ) -> str:
        """Translate advice to local language"""

        prompt = f"""
        Translate this plant disease advice to {language}.
        
        Keep all technical terms but make them 
        understandable in {language}.
        
        Text to translate:
        {text}
        
        Provide only the translation, no explanation.
        """

        return self.llm.generate(
            prompt=prompt,
            system_prompt="You are an expert translator.",
            temperature=0.1,
            max_tokens=800
        )


# ─── BACKWARD COMPATIBILITY CLASS ────────────
class DiseaseAdvisor(PlantDiseaseAdvisor):
    """
    Backward-compatible wrapper for the original frontend integration.
    Allows existing Streamlit interface to query without breaking.
    """
    def __init__(self, client=None):
        from llm.llm_client import get_llm_client
        llm_client = client or get_llm_client()
        # If it's a GeminiClient compatibility wrapper, extract the LLMClient instance
        if hasattr(llm_client, "llm_client"):
            llm_client = llm_client.llm_client
        super().__init__(llm_client)

    def generate_report(self, class_name: str, confidence: float) -> str:
        crop, disease = parse_class_name(class_name)
        severity = "high" if confidence > 90 else "medium"
        return self.generate_full_report(disease, crop, confidence, severity)
