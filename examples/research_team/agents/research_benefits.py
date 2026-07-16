from dataclasses import asdict

from agentos import AgentProcess

from ..contracts import ResearchAssignment, ResearchResult
from ..data import BENEFITS
from ..runtime_events import record_agent_work


class ResearchBenefitsAgent(AgentProcess):
    name = "Research-Benefits"

    async def on_message(self, message) -> None:
        with record_agent_work(self):
            assignment = ResearchAssignment(**message.payload["assignment"])
            if assignment.focus_area != "Benefits":
                raise ValueError(f"unexpected focus area: {assignment.focus_area}")
            self.assignment_received = assignment
            print(f"[Research-Benefits] Received assignment: {assignment.focus_area}")
            result = ResearchResult(focus_area=assignment.focus_area, findings=BENEFITS)
            self.result_sent = result
            print(f"[Research-Benefits] Produced {len(result.findings)} findings")
            print("[Research-Benefits] Sending result -> SynthesizerAgent")
            self.send(self.synthesizer_pid, {"result": asdict(result)})
