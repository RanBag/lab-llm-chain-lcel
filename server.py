
# run directly from the terminal
#!/usr/bin/env python
from typing import TypedDict

from langchain_core.runnables import RunnableLambda
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes

# 1. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# 2. Input Schema
class InputSchema(TypedDict):
    language: str
    text: str
    api_key: str

# 3. Build chain with dynamic API key
def build_chain_with_key(inputs: InputSchema):
    api_key = inputs["api_key"]
    model = ChatOpenAI(api_key=api_key)
    parser = StrOutputParser()
    chain = prompt_template | model | parser
    return chain.invoke(inputs)

# 4. Make Runnable
chain = RunnableLambda(build_chain_with_key)

# 2. Create model
model = ChatOpenAI()

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = prompt_template | model | parser


# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route


print(r"""
   __                          __                     
  / /___  ____ _____ ___  ____/ /__  ____ _____  ___  
 / / __ \/ __ `/ __ `__ \/ __  / _ \/ __ `/ __ \/ _ \ 
/ / /_/ / /_/ / / / / / / /_/ /  __/ /_/ / / / /  __/ 
/_/\____/\__,_/_/ /_/ /_/\__,_/\___/\__,_/_/ /_/\___/  
🐦  Welcome to Randa's LangServe Playground!
""")


add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

    uvicorn.run("server:app", host="localhost", port=8000, reload=True)
