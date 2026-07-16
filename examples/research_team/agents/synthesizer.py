from dataclasses import asdict

from agentos import AgentProcess

from ..contracts import ResearchResult, SynthesizedReport
from ..data import TOPIC
from ..runtime_events import record_agent_work


class SynthesizerAgent(AgentProcess):
    name = "Synthesizer"
    required_focus_areas = {"Benefits", "Risks", "Market Trends"}

    def __init__(self) -> None:
        super().__init__()
        self.results: dict[str, ResearchResult] = {}
        self.report_sent: SynthesizedReport | None = None

    def create_report(self) -> SynthesizedReport | None:
        if set(self.results) != self.required_focus_areas:
            return None
        return SynthesizedReport(
            topic=TOPIC,
            benefits=self.results["Benefits"].findings,
            risks=self.results["Risks"].findings,
            market=self.results["Market Trends"].findings,
            summary=(
                "AI in healthcare is progressing rapidly, with strong potential "
                "benefits, meaningful risks, and growing market adoption."
            ),
        )

    async def on_message(self, message) -> None:
        result = ResearchResult(**message.payload["result"])
        self.results[result.focus_area] = result
        print(f"[Synthesizer] Received result: {result.focus_area}")
        if set(self.results) == self.required_focus_areas and self.report_sent is None:
            with record_agent_work(self):
                report = self.create_report()
                if report is None:
                    raise RuntimeError("synthesis started without all required results")
                self.report_sent = report
                print("[Synthesizer] All research results received")
                print("[Synthesizer] Created synthesized report")
                print("[Synthesizer] Sending report -> CriticAgent")
                self.send(self.critic_pid, {"report": asdict(report)})
