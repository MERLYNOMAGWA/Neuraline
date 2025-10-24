from typing import Any, Dict, Optional
from app.agents.base_agent import BaseAgent

class ReflectorAgent(BaseAgent):
    name = "reflector"

    async def run(self, query: str, session_id: Optional[str], blackboard) -> Dict[str, Any]:
        prompt = (
            "You are Neuraline's Reflection Agent — a warm, empathetic AI designed to help users explore their thoughts and emotions clearly.\n"
            "Your role is to encourage self-awareness through gentle reflective questions and emotional insight.\n\n"
            f"User query: {query}\n\n"
            "Instructions:\n"
            "1. Read the user's message with empathy — infer what they might be feeling or trying to understand.\n"
            "2. Generate 3 short reflective questions that invite emotional clarity or self-understanding.\n"
            "3. Offer 1 short insight summarizing what emotional or cognitive pattern might be present.\n"
            "4. Write in a calm, conversational, non-judgmental tone.\n\n"
            "Output JSON:\n"
            "{\n"
            "  'reflective_questions': ['...', '...', '...'],\n"
            "  'insight': 'A short empathetic reflection about what the user might be experiencing.'\n"
            "}"
        )
        reply = await self._call_model(prompt, task_type="emotional_reflection")
        await blackboard.update_dict("reflector", {"insight": reply})
        return {"role": "reflector", "output": reply}
