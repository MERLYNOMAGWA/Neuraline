import asyncio
import logging
from app.agents.coordinator import CoordinatorAgent
from app.services.retriever import ContextRetriever
from app.services.model_router import ModelRouter
from app.services.memory.chroma_memory import ChromaConversationMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    retriever = ContextRetriever()
    model_router = ModelRouter()
    memory_store = ChromaConversationMemory()

    coord = CoordinatorAgent(retriever=retriever, model_router=model_router, memory_store=memory_store)

    session = "demo_user_1"
    query = "I keep procrastinating my side project and feel stuck. Help me plan."

    print("--- Parallel run (default routing) ---")
    res = await coord.run_parallel(query, task_type="cognitive_reasoning", session_id=session)
    print("EVAL best role:", res["eval"]["best"]["role"])
    print("Combined summary:\n", res["eval"]["combined"][:800])

    print("\n--- Chain run (Reflector -> Strategist -> Coach -> Purpose) ---")
    chain = ["reflector", "strategist", "coach", "purpose"]
    res2 = await coord.run_chain(query, chain=chain, session_id=session)
    print("Chain combined:\n", res2["eval"]["combined"][:1000])

if __name__ == "__main__":
    asyncio.run(main())