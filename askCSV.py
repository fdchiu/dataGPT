import pandas as pd
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.agents.agent_toolkits.conversational_retrieval.tool import (
    create_retriever_tool,
)
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain_experimental.tools import PythonAstREPLTool
from langchain.vectorstores import FAISS
from langsmith import Client
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 20)
os.environ["LANGSMITH_API_KEY"] = os.getenv('LANGSMITH_API_KEY')

embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.load_local("titanic_data", embedding_model)
retriever_tool = create_retriever_tool(
    vectorstore.as_retriever(), "person_name_search", "Search for a person by name"
)

"""
'You are working with a pandas dataframe in Python. 
The name of the dataframe is `df`. 
You should use the tools below to answer the question posed of you:

python_repl_ast: A Python shell. Use this to execute python commands. 
Input should be a valid python command. When using this tool, sometimes output is abbreviated - make sure it does not look abbreviated before using it in your answer.
nUse the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [python_repl_ast]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

This is the result of `print(df.head())`:
{df_head}
Begin!
Question: {input}
{agent_scratchpad}'
"""

TEMPLATE = """You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
It is important to understand the attributes of the dataframe before working with it. This is the result of running `df.head().to_markdown()`

<df>
{dhead}
</df>

You are not meant to use only these rows to answer questions - they are meant as a way of telling you about the shape and schema of the dataframe.
You also do not have use only the information here to answer questions - you can run intermediate queries to do exporatory data analysis to give you more information as needed.

You have a tool called `person_name_search` through which you can lookup a person by name and find the records corresponding to people with similar name as the query.
You should only really use this if your search term contains a persons name. Otherwise, try to solve it with code.

For example:

<question>How old is Jane?</question>
<logic>Use `person_name_search` since you can use the query `Jane`</logic>

<question>Who has id 320</question>
<logic>Use `python_repl` since even though the question is about a person, you don't know their name so you can't include it.</logic>
"""


class PythonInputs(BaseModel):
    query: str = Field(description="code snippet to run")


if __name__ == "__main__":
    df = pd.read_csv("titanic.csv")
    template = TEMPLATE.format(dhead=df.head().to_markdown())

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ("human", "{input}"),
        ]
    )

    def get_chain():
        repl = PythonAstREPLTool(
            locals={"df": df},
            name="python_repl",
            description="Runs code and returns the output of the final line",
            args_schema=PythonInputs,
        )
        tools = [repl, retriever_tool]
        agent = OpenAIFunctionsAgent(
            llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125"), prompt=prompt, tools=tools
        )
        agent_executor = AgentExecutor(
            agent=agent, tools=tools, max_iterations=5, early_stopping_method="generate"
        )
        return agent_executor

    agent = get_chain()
    #output = agent.invoke({"input": "who are the survivors"})
    output = agent.invoke({"input": "how many female survivors"})
    print(output['output'])
    """ client = Client()
        eval_config = RunEvalConfig(
        evaluators=["qa"],
    )
    chain_results = run_on_dataset(
        client,
        dataset_name="Titanic CSV Data",
        llm_or_chain_factory=get_chain,
        evaluation=eval_config,
    )
    """
