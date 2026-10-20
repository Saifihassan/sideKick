from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.agents.middleware import TodoListMiddleware, HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
import os
from tools import tools

from pydantic import BaseModel, Field

load_dotenv(override=True)

llm = ChatOpenAI(
    base_url=os.getenv("VELONA_BASE_URL"),
    api_key= os.getenv("VELONA_API_KEY"),
    model= "upstage/solar-mini4"
)
class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(
        description="True if the assistant has a question, needs clarification, or is stuck and needs the user"
    )


worker = create_agent(
    model=llm,
    system_prompt="you're a helpful assistant and you use the tools given to you to aid the user with whatever query or request they have you must always create a todo list for all the tasks that you're about to perform and then proceed accordingly never aproach the task without a todo list if you are not able to do a certain task say that instead of faking it and provide the user with a way that they can do it on their own ",
    tools=tools,
    middleware=[
        TodoListMiddleware(),
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": {
                    "allowed_decisions": ["approve", "reject"]
                },
                "edit_file": {
                    "allowed_decisions": ["approve", "reject"]
                },
                "delete_file": {
                    "allowed_decisions": ["approve", "reject"]
                },
                "append_file": {
                    "allowed_decisions": ["approve", "reject"]
                }
            }
        ),
    ],
    checkpointer=InMemorySaver(),
)

evaluator = create_agent(
    model=llm,
    system_prompt=(
        "You are an evaluator assessing the response of a worker agent.\n"
        "Judge whether the worker agent's output satisfies the user's request.\n"
        "Provide clear feedback and determine whether the success criteria have been met "
        "or if additional user input is required."
    ),
    response_format=EvaluatorOutput
)

MAX_RETRIES = 3
config = {"configurable": {"thread_id": "sidekick"}}

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break

    messages = [{"role": "user", "content": user_input}]
    retries = 0

    while retries < MAX_RETRIES:
        response = worker.invoke({"messages": messages}, config, version="v2")

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
                    version="v2"
                )

        state = worker.get_state(config)
        for todo in state.values.get("todos", []):
            print(f"[{todo['status']}] {todo['content']}")

        worker_output = state.values["messages"][-1].content
        print(f"\nWorker:\n{worker_output}\n")

        eval_result = evaluator.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": f"User Request: {user_input}\nWorker Output: {worker_output}"
                }
            ]
        })
        feedback: EvaluatorOutput = eval_result["structured_response"]
        print(f"Evaluator Feedback:\n{feedback.feedback}")
        print(f"Success Criteria Met: {feedback.success_criteria_met}")
        print(f"User Input Needed: {feedback.user_input_needed}\n")

        if feedback.success_criteria_met:
            break

        retries += 1
        if retries >= MAX_RETRIES:
            print("Max retries reached.\n")
            break

        messages = [
            {"role": "user", "content": f"Evaluator feedback: {feedback.feedback}\nPlease revise your answer."}
        ]    


    
