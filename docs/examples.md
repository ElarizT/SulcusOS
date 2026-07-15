# Examples

All paths are relative to the repository root.

| Purpose | Command or path | Offline | Credentials | Native core |
| --- | --- | :---: | :---: | :---: |
| First run | `python examples\public_api_quickstart.py` | ✓ | — | — |
| Runtime capabilities | `python examples\runtime_capabilities_demo.py` | ✓ | — | — |
| Agent/process basics | `examples\hello_agent.py`, `examples\echo_agent.py` | ✓ | — | Required by bundled host |
| IPC basics | `examples\ipc_ping_pong_quickstart.py` | ✓ | — | ✓ |
| Supervision basics | `examples\supervisor_quickstart.py` | ✓ | — | ✓ |
| LLM/tool runtime | `python examples\manual_tool_execution_phase5_smoke_test.py` | ✓ | — | — |
| Multi-round tool loop | `python examples\agent_tool_loop_multi_round_demo.py` | ✓ | — | — |
| Permissions | `python examples\agent_tool_loop_tool_permission_demo.py` | ✓ | — | — |
| Resource limits | `python examples\agent_tool_loop_resource_limits_demo.py` | ✓ | — | — |
| Resumable approval | `python examples\agent_tool_loop_approval_resume_demo.py` | ✓ | — | — |
| Persistent checkpoint | `python -m examples.agent_tool_loop_persistent_checkpoint_demo` | ✓ | — | — |
| OpenAI-compatible smoke test | `python examples\agent_tool_loop_phase6_smoke_test.py` | | `AGENTOS_LLM_API_KEY` | — |
| Flagship demo | `sulcus demo research-team` | ✓ | — | — |
| Full process dashboard | `python main.py` | ✓ | — | ✓ plus dashboard extra |

## Suggested path

Run the first-run example, then the multi-round, limits, approval, and
persistent-checkpoint demos. Use the flagship demo to see those controls in a
larger workflow. Move to IPC/supervision examples only after installing the
native core and dashboard dependencies.

The OpenAI-compatible smoke tests make real network requests and are optional.
All deterministic demos use scripted providers or bundled data and make no
claim about model quality.
