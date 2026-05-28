import asyncio
import os
import sys

try:
    print("1. Testing Native Core Component Bindings...")
    from agent_os_core import RustKernel, NativeIPCBus, ContextMemoryManager, WasmSandboxManager
    kernel = RustKernel()
    bus = NativeIPCBus(kernel)
    memory = ContextMemoryManager(max_active_tokens=8000)
    sandbox = WasmSandboxManager()
    print("   -> Native Core: OK")

    print("\n2. Testing Utility Toolchain Import...")
    from kernel.toolchain import compile_agent_script
    print("   -> Toolchain: OK")

    print("\n3. Ingesting Shell Environment Parameters...")
    provider = os.getenv("AGENT_OS_LLM_PROVIDER")
    model = os.getenv("AGENT_OS_LLM_MODEL")
    base_url = os.getenv("AGENT_OS_LLM_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"   -> Provider: {provider}\n   -> Model: {model}\n   -> Base URL: {base_url}")

    print("\n4. Initializing Asynchronous LLM Manager...")
    from kernel.llm import AsyncLLMManager, LLMConfig
    config = LLMConfig(provider=provider, model_name=model, api_key=api_key, base_url=base_url)
    llm = AsyncLLMManager(config)
    print("   -> LLM Manager: OK")

except Exception as e:
    import traceback
    print("\n SYSTEM INITIALIZATION CRASH DETECTED:")
    traceback.print_exc()
    sys.exit(1)

async def test_cloud_ping():
    try:
        print("\n5. Dispatching Test Frame to Gemini Cloud API...")
        response = await llm.generate_response(
            system_prompt="You are a system verification daemon. Reply with exactly 'PING OK'.", 
            active_context=["Hello Kernel"]
        )
        print(f"   -> Cloud Response: {response.text}")
        print("\n ALL SUB-SYSTEMS VERIFIED SUCCESSFULLY!")
    except Exception as e:
        import traceback
        print("\n CLOUD NETWORK TRANSACTION CRASH DETECTED:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_cloud_ping())
