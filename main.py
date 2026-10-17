from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from dotenv import load_dotenv
import os
from tools import tools

load_dotenv(override=True)

llm = ChatOpenAI(
    base_url=os.getenv("VELONA_BASE_URL"),
    api_key= os.getenv("VELONA_API_KEY"),
    model= "stepfun/step-3.5-flash"
)


worker = create_agent(
    model=llm,
    system_prompt="you're a helpflu assistant and you use the tools that are given to you to help with the users queries",
    tools=tools   
)



while True:
    
    user_input = input("You: ")
    if user_input == "exit":
        break   
    response = worker.invoke({"messages":[{"role": "user", "content": user_input}]})
    print(response["messages"][-1].content)    


    
