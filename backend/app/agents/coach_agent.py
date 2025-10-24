from typing import Any, Dict, Optional
from app.agents.base_agent import BaseAgent

class CoachAgent(BaseAgent):
    name = "coach"

    async def run(self, query: str, session_id: Optional[str], blackboard) -> Dict[str, Any]:
        strategist = await blackboard.read("strategist", {})
        plan = strategist.get("plan", "")
        prompt = (
            "You are Neuraline's Consistency Coach — a warm, structured AI that helps users stay consistent through emotional intelligence and practical strategy.\n\n"
            "Your goal is to transform the given plan into specific, psychologically sound micro-habits and motivational nudges that build lasting consistency.\n\n"
            "Context:\n"
            f"- User query: {query}\n"
            f"- Plan to refine: {plan}\n\n"
            "Instructions:\n"
            "1. Identify the core behavioral goals behind the plan.\n"
            "2. Break them into 2 to 4 micro-habits that are simple, trackable, and emotionally sustainable.\n"
            "3. For each habit, write one actionable accountability nudge in a caring, supportive tone.\n"
            "4. If relevant, reflect briefly on the psychological challenge (e.g., procrastination, overwhelm) and how the nudge helps overcome it.\n\n"
            "Output in structured JSON:\n"
            "{\n"
            "  'micro_habits': [\n"
            "           {'habit': '...', 'why_it_works': '...', 'nudge': '...'}\n"
            "  ],\n"
            "  'summary': 'Short reflective summary (2 sentences) about consistency and emotional growth.'\n"
            "}"
    )
        reply = await self._call_model(prompt, task_type="behavioral_coaching")
        await blackboard.update_dict("coach", {"nudges": reply})
        return {"role": "coach", "output": reply}
