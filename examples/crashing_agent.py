from kernel.process import AgentProcess


class CrashingAgent(AgentProcess):
    name = "CrashingAgent"

    async def on_start(self) -> None:
        raise RuntimeError("intentional example crash")
