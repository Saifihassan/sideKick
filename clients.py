"""LLM client connections."""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

llm = ChatOpenAI(
    base_url=os.getenv("VELONA_BASE_URL"),
    api_key=os.getenv("VELONA_API_KEY"),
    model="upstage/solar-mini4",
)
iamhc = ChatOpenAI(
    base_url=os.getenv("IAMHC_BASE_URL"),
    api_key=os.getenv("IAMHC_API_KEY"),
    model="MiniMax-M3.1-Flash",
)
nararouter = ChatOpenAI(
    base_url=os.getenv("NARAROUTER_BASE_URL"),
    api_key=os.getenv("NARAROUTER_API_KEY"),
    model="nemotron-3.5-lightning-free",
)

groq = ChatOpenAI(
    base_url=os.getenv("GROQ_BASE_URL"),
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
)

__all__ = ["llm", "nararouter", "groq"]
