"""Sidekick — main entry point."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agents import create_worker

config = {"configurable": {"thread_id": "sidekick-2"}}

memory = InMemorySaver()
worker = create_worker(memory)

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break

    messages = [{"role": "user", "content": user_input}]
    response = worker.invoke({"messages": messages}, config, version="v2")

    # Handle HITL interrupts
    while getattr(response, "interrupts", None):
        for interrupt in response.interrupts:
            decisions = []
            for action in interrupt.value.get("action_requests", []):
                print(f"\n[Tool Approval Required]")
                print(f"Tool: {action.get('name')}")
                print(f"Arguments: {action.get('args')}")
                decision = input("Do you approve? (yes/no): ").strip().lower()
                if decision in ("yes", "y"):
                    decisions.append({"type": "approve"})
                else:
                    decisions.append({
                        "type": "reject",
                        "message": "User rejected this tool call."
                    })
            response = worker.invoke(
                Command(resume={"decisions": decisions}),
                config,
                version="v2",
            )

    # Print todo list and final worker output
    state = worker.get_state(config)
    for todo in state.values.get("todos", []):
        print(f"[{todo['status']}] {todo['content']}")

    worker_output = state.values["messages"][-1].content
    print(f"\nWorker:\n{worker_output}\n")
