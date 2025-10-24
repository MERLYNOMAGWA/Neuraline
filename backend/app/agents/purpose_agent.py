from typing import Any, Dict, Optional
from app.agents.base_agent import BaseAgent

class PurposeAgent(BaseAgent):
    name = "purpose"

    async def run(self, query: str, session_id: Optional[str], blackboard) -> Dict[str, Any]:
        ref = await blackboard.read("reflector", {})
        strat = await blackboard.read("strategist", {})
        prompt = (
            "You are Neuraline's Purpose Agent — an empathetic AI guide who helps users align daily actions with their deeper life values.\n"
            "Your goal is to interpret the user's plan and reflection, then craft 2 to 3 sentences that connect their actions to purpose and inner motivation.\n\n"
            "Context:\n"
            f"- Reflection insight: {ref.get('insight', 'No reflection provided')}\n"
            f"- Strategic plan: {strat.get('plan', 'No plan available')}\n"
            f"- User query: {query}\n\n"
            "Guidelines:\n"
            "1. Identify the underlying values or long-term goals implied by the plan.\n"
            "2. Express how following this plan supports personal growth, contribution, or self-fulfillment.\n"
            "3. Write in a warm, reflective, and motivational tone — like a mentor helping the user reconnect with their 'why.'\n\n"
            "Output JSON:\n"
            "{\n"
            "  'purpose_alignment': '2 to 3 sentence reflection connecting the plan to purpose and values',\n"
            "  'core_value': 'word or phrase representing the underlying purpose (e.g., growth, discipline, balance)'\n"
            "}"
        )
        reply = await self._call_model(prompt, task_type="purpose_alignment")
        await blackboard.update_dict("purpose", {"alignment": reply})
        return {"role": "purpose", "output": reply}