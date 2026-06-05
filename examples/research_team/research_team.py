from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from kernel.process import AgentMessage

from .agents import (
    CriticAgent,
    PlannerAgent,
    ResearchBenefitsAgent,
    ResearchMarketAgent,
    ResearchRisksAgent,
    SynthesizerAgent,
)


class DemoBus:
    def __init__(self) -> None:
        self.mailboxes: dict[str, asyncio.Queue[AgentMessage]] = {}
        self.pid_to_name: dict[int, str] = {}

    def attach(self, pid: int, agent: Any) -> None:
        agent.pid = pid
        agent.agent_name = agent.name
        agent.bus = self
        agent.stop_event = asyncio.Event()
        self.pid_to_name[pid] = agent.name
        self.mailboxes[agent.name] = asyncio.Queue()

    def send_message(self, message: AgentMessage) -> None:
        receiver = self.pid_to_name[int(message.receiver)]
        self.mailboxes[receiver].put_nowait(message)

    async def recv_message(self, agent_name: str) -> AgentMessage:
        return await self.mailboxes[agent_name].get()


async def run_demo() -> dict[str, Any]:
    print("=" * 64)
    print("Agent OS Research Team Demo")
    print("=" * 64)
    print("[SUPERVISOR] Research Team Started")
    print("\n--- IPC Workflow ---")
    bus = DemoBus()
    research_agents = [
        ResearchBenefitsAgent(),
        ResearchRisksAgent(),
        ResearchMarketAgent(),
    ]
    planner = PlannerAgent()
    synthesizer = SynthesizerAgent()
    critic = CriticAgent()
    agents = [planner, *research_agents, synthesizer, critic]
    for pid, agent in enumerate(agents, start=100):
        bus.attach(pid, agent)

    planner.research_pids = {
        "ResearchBenefitsAgent": research_agents[0].pid,
        "ResearchRisksAgent": research_agents[1].pid,
        "ResearchMarketAgent": research_agents[2].pid,
    }
    for agent in research_agents:
        agent.synthesizer_pid = synthesizer.pid
    synthesizer.critic_pid = critic.pid
    tasks = [asyncio.create_task(agent.run()) for agent in agents]
    try:
        deadline = asyncio.get_running_loop().time() + 2.0
        while not all(hasattr(agent, "assignment_received") for agent in research_agents):
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("planner assignments were not delivered")
            await asyncio.sleep(0.01)
        print("[SUPERVISOR] Step 3 complete: planner assignments delivered")
        while len(synthesizer.results) != 3:
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("research results were not delivered")
            await asyncio.sleep(0.01)
        print("[SUPERVISOR] Step 4 complete: research results delivered")
        while not hasattr(critic, "report_received"):
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("synthesized report was not delivered")
            await asyncio.sleep(0.01)
        print("[SUPERVISOR] Step 5 complete: synthesized report delivered")
        while not hasattr(critic, "review"):
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("critic review was not generated")
            await asyncio.sleep(0.01)
        print("\n--- Final Result ---")
        print("=" * 64)
        print("[SUPERVISOR] Workflow Complete")
        print(f"[SUPERVISOR] Final Score: {critic.review.score}/10")
        print("=" * 64)
        hierarchy = {
            "supervisor": "ResearchTeamSupervisor",
            "children": [
                "PlannerAgent",
                "ResearchBenefitsAgent",
                "ResearchRisksAgent",
                "ResearchMarketAgent",
                "SynthesizerAgent",
                "CriticAgent",
            ],
        }
        return {
            "assignments": [agent.assignment_received for agent in research_agents],
            "research_results": dict(synthesizer.results),
            "synthesized_report": critic.report_received,
            "critic_review": critic.review,
            "hierarchy": hierarchy,
            "research_agents": research_agents,
            "synthesizer": synthesizer,
            "critic": critic,
        }
    finally:
        for agent in agents:
            agent.stop_event.set()
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task


if __name__ == "__main__":
    asyncio.run(run_demo())
