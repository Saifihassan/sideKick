"""The sidekick worker/evaluator agent loop.

Exposes :func:`main`, which runs an interactive chat loop where a worker
agent answers the user and an evaluator agent critiques the answer,
re-prompting the worker until the success criteria are met (or the retry
budget is exhausted).
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from .tools import tools

load_dotenv(override=True)

MAX_RETRIES = 3
SYSTEM_PROMPT = (
    "you're a helpful assistant and you use the tools that are given to you "
    "to help with the users queries"
)


class EvaluatorOutput(BaseModel):
    """Structured verdict returned by the evaluator model."""

    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(
        description="Whether the success criteria have been met"
    )
    user_input_needed: bool = Field(
        description=(
            "True if the assistant has a question, needs clarification, or is "
            "stuck and needs the user"
        )
    )


def build_llm() -> ChatOpenAI:
    """Create the primary worker/evaluator model from environment settings."""
    return ChatOpenAI(
        base_url=os.getenv("VELONA_BASE_URL"),
        api_key=os.getenv("VELONA_API_KEY"),
        model="deepseek/deepseek-v4.1-flash",
    )


EVALUATOR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an evaluator assessing the response and actions of an AI "
            "worker agent.\n"
            "Analyze whether the worker agent's output completely and accurately "
            "satisfies the user's request.\n"
            "- If the response fully satisfies the user request, set "
            "success_criteria_met=True.\n"
            "- If the response is incomplete, incorrect, or missing requested "
            "actions (e.g. creating files, searching web, comparing), set "
            "success_criteria_met=False and provide actionable feedback "
            "explaining what needs to be fixed.\n"
            "- If the worker asks a genuine clarifying question or needs user "
            "input, set user_input_needed=True and success_criteria_met=True."
        ),
    ),
    (
        "human",
        "Original User Request:\n{user_input}\n\nWorker Agent Output:\n{worker_output}",
    ),
])


def _print_todos(response: dict) -> None:
    for todo in response.get("todos", []):
        print(f"[{todo['status']}] {todo['content']}")


def run_agent_turn(worker, evaluator, user_input: str, config: dict) -> str:
    """Run one user turn through the worker/evaluator retry loop.

    Returns the final worker output text.
    """
    response = worker.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
    )
    _print_todos(response)

    retries = 0
    while retries < MAX_RETRIES:
        worker_output = response["messages"][-1].content
        ev_response = evaluator.invoke({
            "user_input": user_input,
            "worker_output": worker_output,
        })

        if ev_response.success_criteria_met or ev_response.user_input_needed:
            return worker_output

        retries += 1
        print(f"\n[Evaluator] Attempt {retries}/{MAX_RETRIES} did not meet criteria.")
        print(f"[Feedback] {ev_response.feedback}")
        print("[Worker] Retrying with evaluator feedback...\n")

        feedback_prompt = (
            "The evaluator assessed your response against the original user "
            "request:\n"
            f"Original User Request: {user_input}\n\n"
            f"Evaluator Feedback: {ev_response.feedback}\n\n"
            "Please address this feedback, execute any required tools, and "
            "revise your response."
        )
        response = worker.invoke(
            {"messages": [{"role": "user", "content": feedback_prompt}]},
            config,
        )
        _print_todos(response)

    # Retry budget exhausted: re-check the final output and report if still bad.
    if retries >= MAX_RETRIES:
        final_output = response["messages"][-1].content
        final_eval = evaluator.invoke({
            "user_input": user_input,
            "worker_output": final_output,
        })
        if not final_eval.success_criteria_met and not final_eval.user_input_needed:
            print(f"\n[Evaluator] Maximum retries ({MAX_RETRIES}) reached. Stopping task.")
            print(f"[Final Feedback] {final_eval.feedback}\n")

    return response["messages"][-1].content


def main() -> None:
    """Run the interactive sidekick agent loop."""
    llm = build_llm()
    evaluator = EVALUATOR_PROMPT | llm.with_structured_output(EvaluatorOutput)

    with SqliteSaver.from_conn_string("sidekick.db") as memory:
        worker = create_agent(
            model=llm,
            system_prompt=SYSTEM_PROMPT,
            tools=tools,
            middleware=[TodoListMiddleware()],
            checkpointer=memory,
        )
        config = {"configurable": {"thread_id": "sidekick"}}

        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            final_output = run_agent_turn(worker, evaluator, user_input, config)
            print(final_output)


if __name__ == "__main__":
    main()
