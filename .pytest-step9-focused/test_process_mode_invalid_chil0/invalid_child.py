
from kernel.process import AgentProcess

class InvalidChild(AgentProcess):
    name = "InvalidChild"
    mailbox_size = -1
