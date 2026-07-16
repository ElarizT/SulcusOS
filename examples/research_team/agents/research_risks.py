from dataclasses import asdict

from agentos import AgentProcess

from ..contracts import ResearchAssignment, ResearchResult
from ..data import RISKS
from ..runtime_events import record_agent_work


class ResearchRisksAgent(AgentProcess):
    name = "Research-Risks"

    async def on_message(self, message) -> None:
        with record_agent_work(self):
            assignment = ResearchAssignment(**message.payload["assignment"])
            if assignment.focus_area != "Risks":
                raise ValueError(f"unexpected focus area: {assignment.focus_area}")
            self.assignment_received = assignment
            print(f"[Research-Risks] Received assignment: {assignment.focus_area}")
            result = ResearchResult(focus_area=assignment.focus_area, findings=RISKS)
            self.result_sent = result
            print(f"[Research-Risks] Produced {len(result.findings)} findings")
            print("[Research-Risks] Sending result -> SynthesizerAgent")
            self.send(self.synthesizer_pid, {"result": asdict(result)})
