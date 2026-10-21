"""Worker agent definitions."""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    TodoListMiddleware,
    ToolErrorMiddleware,
    ModelCallLimitMiddleware
)

from clients import llm,nararouter,groq,iamhc
from tools import tools


WORKER_SYSTEM_PROMPT = (
    "you're a helpful assistant and you use the tools given to you to aid the user "
    "with whatever query or request they have you must always create a todo list for "
    "all the tasks that you're about to perform and then proceed accordingly never "
    "approach the task without a todo list if you are not able to do a certain task "
    "say that instead of faking it and provide the user with a way that they can do "
    "it on their own"
)

HITL_INTERRUPT_ON = {
    "write_file": {"allowed_decisions": ["approve", "reject"]},
    "edit_file":  {"allowed_decisions": ["approve", "reject"]},
    "delete_file": {"allowed_decisions": ["approve", "reject"]},
    "append_file": {"allowed_decisions": ["approve", "reject"]},
}


def _tool_error_handler(exc: Exception, req) -> str:
    tool_name = req.tool.name if req.tool else req.tool_call.get("name", "unknown")
    return f"Tool '{tool_name}' failed: {type(exc).__name__}: {exc}"


def create_worker(checkpointer):
    """Create and return the worker agent."""
    return create_agent(
        model=llm,
        system_prompt=WORKER_SYSTEM_PROMPT,
        tools=tools,
        middleware=[
            ToolErrorMiddleware(_tool_error_handler),
            TodoListMiddleware(),
            HumanInTheLoopMiddleware(interrupt_on=HITL_INTERRUPT_ON),
            ModelCallLimitMiddleware(run_limit=30)
        ],
        checkpointer=checkpointer,
    )


__all__ = ["create_worker"]
