import asyncio
import logging
from typing import List, Dict, Optional

from app.agents.blackboard import Blackboard
from app.agents.reflector_agent import ReflectorAgent
from app.agents.strategist_agent import StrategistAgent
from app.agents.coach_agent import CoachAgent
from app.agents.purpose_agent import PurposeAgent
from app.agents.evaluator import EvaluatorAgent
from app.core.logging_config import log_event

logger = logging.getLogger(__name__)

class CoordinatorAgent:
    """Orchestrates agents: supports parallel and chain execution + meta-eval."""

    def __init__(self, retriever=None, model_router=None, memory_store=None, timeout: int = 20):
        self.blackboard = Blackboard()
        self.retriever = retriever
        self.model_router = model_router
        self.memory_store = memory_store
        self.timeout = timeout

        self.reflector = ReflectorAgent(retriever, model_router, memory_store)
        self.strategist = StrategistAgent(retriever, model_router, memory_store)
        self.coach = CoachAgent(retriever, model_router, memory_store)
        self.purpose = PurposeAgent(retriever, model_router, memory_store)
        self.evaluator = EvaluatorAgent()

        self.routing_table = {
            "emotional_reflection": ["reflector", "purpose"],
            "cognitive_reasoning": ["strategist", "coach", "purpose"],
            "behavioral_coaching": ["coach", "reflector"],
            "rag_query": ["reflector", "strategist", "purpose"],
            "general_chat": ["reflector", "strategist", "coach", "purpose"],
        }

    async def _run_agent(self, agent, query, session_id):
        try:
            return await asyncio.wait_for(agent.run(query, session_id, self.blackboard), timeout=self.timeout)
        except asyncio.TimeoutError:
            log_event("AGENT_TIMEOUT", f"{agent.name} timed out for session={session_id}")
            logger.warning(f"{agent.name} timed out")
            return {"role": agent.name, "output": f"{agent.name} timed out."}
        except Exception as e:
            log_event("AGENT_ERROR", f"{agent.name} error: {e}")
            logger.exception(e)
            return {"role": agent.name, "output": f"{agent.name} failed: {e}"}

    async def run_parallel(self, query: str, task_type: str, session_id: Optional[str]) -> Dict:
        """Run all agents in routing_table[task_type] concurrently and return evaluator summary."""
        agent_names = self.routing_table.get(task_type, ["reflector", "strategist"])
        name_map = {
            "reflector": self.reflector,
            "strategist": self.strategist,
            "coach": self.coach,
            "purpose": self.purpose,
        }
        jobs = [self._run_agent(name_map[n], query, session_id) for n in agent_names if n in name_map]
        results = await asyncio.gather(*jobs)
        eval_result = await self.evaluator.evaluate(query, results)
        snapshot = await self.blackboard.dump()
        log_event("COORDINATOR_PARALLEL", f"session={session_id} task={task_type} snapshot_keys={list(snapshot.keys())}")
        return {"results": results, "eval": eval_result, "snapshot": snapshot}

    async def run_chain(self, query: str, chain: List[str], session_id: Optional[str]) -> Dict:
        """Run agents sequentially following chain order (strings of agent names)."""
        name_map = {
            "reflector": self.reflector,
            "strategist": self.strategist,
            "coach": self.coach,
            "purpose": self.purpose,
        }
        results = []
        for n in chain:
            agent = name_map.get(n)
            if not agent:
                continue
            res = await self._run_agent(agent, query, session_id)
            results.append(res)
        eval_result = await self.evaluator.evaluate(query, results)
        snapshot = await self.blackboard.dump()
        log_event("COORDINATOR_CHAIN", f"session={session_id} chain={chain} snapshot_keys={list(snapshot.keys())}")
        return {"results": results, "eval": eval_result, "snapshot": snapshot}