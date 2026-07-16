from dataclasses import asdict

from agentos import AgentProcess

from ..contracts import ResearchAssignment
from ..data import TOPIC
from ..runtime_events import record_agent_work


class PlannerAgent(AgentProcess):
    name = "Planner"

    def __init__(self) -> None:
        super().__init__()
        self.research_pids: dict[str, int] = {}

    def create_assignments(self) -> list[ResearchAssignment]:
        return [
            ResearchAssignment(TOPIC, "Benefits", "ResearchBenefitsAgent"),
            ResearchAssignment(TOPIC, "Risks", "ResearchRisksAgent"),
            ResearchAssignment(TOPIC, "Market Trends", "ResearchMarketAgent"),
        ]

    async def on_start(self) -> None:
        with record_agent_work(self):
            print(f"[Planner] Creating research plan for: {TOPIC}")
            for assignment in self.create_assignments():
                print(
                    f"[Planner] Sending assignment: {assignment.focus_area} -> "
                    f"{assignment.destination}"
                )
                self.send(
                    self.research_pids[assignment.destination],
                    {"assignment": asdict(assignment)},
                )
