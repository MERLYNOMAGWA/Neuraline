from typing import Any, Dict, Optional
from app.agents.base_agent import BaseAgent

class StrategistAgent(BaseAgent):
    name = "strategist"

    async def run(self, query: str, session_id: Optional[str], blackboard) -> Dict[str, Any]:
        reflector = await blackboard.read("reflector", {})
        seed = reflector.get("insight", "")
        prompt = (
            "You are Neuraline's Strategist Agent — a structured thinking AI that transforms insights into clear, achievable action steps.\n"
            "Your job is to design a practical 3-step weekly plan that helps the user make progress with clarity and balance.\n\n"
            f"User query: {query}\n"
            f"Reflector insight: {seed}\n\n"
            "Guidelines:\n"
            "1. Consider the emotional tone from the insight — make the plan encouraging and human.\n"
            "2. Each step should be specific, time-bound (Day 1-7), and achievable.\n"
            "3. Include a short motivational phrase for each step.\n"
            "4. Avoid generic advice; use the user's intent and reflection context.\n\n"
            "Output JSON:\n"
            "{\n"
            "  'weekly_plan': [\n"
            "           {'day': 'Mon-Tue', 'goal': '...', 'action': '...', 'motivation': '...'},\n"
            "           {'day': 'Wed-Thu', 'goal': '...', 'action': '...', 'motivation': '...'},\n"
            "           {'day': 'Fri-Sun', 'goal': '...', 'action': '...', 'motivation': '...'}\n"
            "  ],\n"
            "  'summary': 'Brief paragraph explaining how this plan aligns emotional insight with structured action.'\n"
            "}"
        )
        reply = await self._call_model(prompt, task_type="cognitive_reasoning")
        await blackboard.update_dict("strategist", {"plan": reply})
        return {"role": "strategist", "output": reply}
