import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import os
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent, create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI, OpenAI
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
import asyncio
from langchain.vectorstores import FAISS
from langsmith import Client
from pydantic import BaseModel, Field
from langchain_experimental.tools import PythonAstREPLTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents.agent_toolkits.conversational_retrieval.tool import (
    create_retriever_tool,
)
from langchain.chains import create_extraction_chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers.json import SimpleJsonOutputParser
import json

load_dotenv()

pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 20)

class PythonInputs(BaseModel):
    query: str = Field(description="code snippet to run")

class TrackableAssistantAgent(AssistantAgent):
    """
    A custom AssistantAgent that tracks the messages it receives.

    This is done by overriding the `_process_received_message` method.
    """

    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            st.markdown(message)
        return super()._process_received_message(message, sender, silent)


class TrackableUserProxyAgent(UserProxyAgent):
    """
    A custom UserProxyAgent that tracks the messages it receives.

    This is done by overriding the `_process_received_message` method.
    """

    def _process_received_message(self, message, sender, silent):
        with st.chat_message(sender.name):
            if message.lower().find("api_key") == -1:
                st.markdown(message)
        return super()._process_received_message(message, sender, silent)

selected_model = "gpt-3.5-turbo-0125" #"gpt-4-1106-preview"
selected_key = os.getenv("OPENAI_API_KEY")

def csvAgent():
    agent = create_csv_agent(
    ChatOpenAI(temperature=0,model=selected_model),
    ["./salesData/order-details.csv", ], #"./salesData/orders.csv","./salesData/prodcuts.csv
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #agent_type=AgentType.OPENAI_FUNCTIONS
    )
    return agent

def pandaFrameAgent(dfs):
    # Create a pandas dataframe agent with the GPT-3.5-turbo API model
    df = pd.read_csv("./salesData/products.csv")
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125"),
        dfs,
        verbose=True,
        max_iterations = 100,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    return agent

# Now you can use the agent to interact with the dataframe
    #result = agent.run('What is the longest name?')

def plot(option, data):
    selected_plot = option #st.sidebar.selectbox("Choose a plot type", plot_options)

    if selected_plot == "Bar plot":
        x_axis = st.sidebar.selectbox("Select x-axis", data.columns)
        y_axis = st.sidebar.selectbox("Select y-axis", data.columns)
        st.write("Bar plot:")
        fig, ax = plt.subplots()
        sns.barplot(x=data[x_axis], y=data[y_axis], ax=ax)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)

    elif selected_plot == "Scatter plot":
        x_axis = st.sidebar.selectbox("Select x-axis", data.columns)
        y_axis = st.sidebar.selectbox("Select y-axis", data.columns)
        st.write("Scatter plot:")
        fig, ax = plt.subplots()
        sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)
        st.pyplot(fig)

    elif selected_plot == "Histogram":
        column = st.sidebar.selectbox("Select a column", data.columns)
        bins = st.sidebar.slider("Number of bins", 5, 100, 20)
        st.write("Histogram:")
        fig, ax = plt.subplots()
        sns.histplot(data[column], bins=bins, ax=ax)
        st.pyplot(fig)

    elif selected_plot == "Box plot":
        column = st.sidebar.selectbox("Select a column", data.columns)
        st.write("Box plot:")
        fig, ax = plt.subplots()
        sns.boxplot(data[column], ax=ax)
        st.pyplot(fig)

Template0 = """
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

TEMPLATE = """You are working with a pandas dataframe in Python. There are 3 dataframes: 'data_products', 'data_orders', 'data_orderdetails'.
'data_prodcuts' has information of all products including id, name etc., 'data_orders' has orders about orderid, shipping address customerid, order date etc., 
while 'data_orderDetails' has order id, product id, unit price, order date etc.. Use these dataframe to solve user query.

The name of the dataframes are `data_products`, 'data_orders', 'data_orderDetails'.
It is important to understand the attributes of the dataframe before working with it. This is the result of running `df.head().to_markdown()`

<df>
{dhead}
</df>

You are not meant to use only these rows to answer questions - they are meant as a way of telling you about the shape and schema of the dataframe.
You also do not have use only the information here to answer questions - you can run intermediate queries to do exporatory data analysis to give you more information as needed.

"""
embedding_model = OpenAIEmbeddings()

# Will be used in the future
#vectorstore = FAISS.load_local("titanic_data", embedding_model)
#retriever_tool = create_retriever_tool(
#    vectorstore.as_retriever(), "person_name_search", "Search for a person by name")

def get_chain(prompt, dfs):
    repl = PythonAstREPLTool(
        locals = dfs,  #{"df": df},
        name="python_repl",
        description="Runs code and returns the output of the final line",
        args_schema=PythonInputs,
    )
    tools = [repl]  #retriever_tool
    agent = OpenAIFunctionsAgent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125"), prompt=prompt, tools=tools
    )
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, max_iterations=5, early_stopping_method="generate"
    )
    return agent_executor

def process_data(data):
    action_data = {}
    if 'text' in data and data['text']:
        extracted_data = data['text'][0] if isinstance(data['text'], list) else data['text']

        for key in ['plot', 'show','filedName', 'extra_info']:            
            if key in extracted_data and extracted_data[key]:
                action_data[key] = extracted_data[key]
    return action_data

schema = {
        "properties": {
            "plot": {"type": "string"},
            "show": {"type": "string"},
            "extra_info": {"type": "string"}
        },    
}

schemaPlotAxis = {
        "properties": {
            "x_axis": {"type": "string"},
            "y_axis": {"type": "string"},
        },    
        "required": ["x_axis", "y_axis"]
}

extractionChain = create_extraction_chain(schema, ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125"))

extractionChainPlotAxis = create_extraction_chain(schemaPlotAxis, ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125"))

def getAxis(inputString, dataColumnNames):
    localPrompt = """
        Your task is to extract x-axis and y-axis from {inputString} those can be found in the dataframe column names:
        {dataColumnNames}
        
        Respond with json out with values for 'x_axis'  and 'y_axis' .

        Some examples:
        in 'orderDate vs orderQuantity', 'x_axis' = 'orderDate', 'y_axis' = 'quantity'.

    """ 
    prompt = PromptTemplate(
        input_variables=["inputString", "dataColumnNames"],
        template=localPrompt
    )

    #json_parser = SimpleJsonOutputParser()

    llm_chain = LLMChain(
        llm=OpenAI(openai_api_key=selected_key),
        prompt=prompt,
    ) 

    print('getAxis')
    print(inputString)
    #print(dataColumnNames)
    output = llm_chain.invoke({"inputString": inputString, "dataColumnNames": dataColumnNames})[
        'text'].strip()
    if output:
        axis = extractionChainPlotAxis.invoke(output)
        return axis
    #print(output)
    return None

def main():

    st.title("dataGPT - Chat with Your Data")
    st.header("Chat and Plot With CSV Data")
    st.write("Scroll down to enter questions")
    #data_file = st.file_uploader("Upload CSV", type=["csv"])
        # Initialize the chat history in the session_state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    #if data_file is not None:
    #    data_orders = pd.read_csv("./salesData/orders.csv")
    #    data_orderDetails = pd.read_csv("./salesData/order-details.csv")
    #    data_products= pd.read_csv("./salesData/products.csv")
    #else:  # Add this block
        # Replace with your fixed file path
    #    fixed_file_path = os.getcwd()+"/moviesDB.csv"  
        #data = pd.read_csv(fixed_file_path)        
    data_orders = pd.read_csv("./salesData/orders.csv")
    data_orderDetails = pd.read_csv("./salesData/order-details.csv")
    data_products = pd.read_csv("./salesData/products.csv")

    dheads = [data_products.head().to_markdown(), data_orders.head().to_markdown(), data_orderDetails.head().to_markdown()]
    print(dheads)
    template = TEMPLATE.format(dhead=dheads)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ("human", "{input}"),
        ]
    )

    st.write("Product Data Overview:")
    st.write(data_products.head())
    st.write("Order Data Overview:")
    st.write(data_orders.head())
    st.write("Order Details Data Overview:")
    st.write(data_orderDetails.head())
    st.divider()
    st.subheader("Sample questions: ")
    st.write("‒ Plot product name vs unit price")
    st.write("‒ List all products")
    st.write("‒ How many orders?")
    st.write("‒ Show product name vs supplier ID")
    st.write("‒ How many orders on 1996-07-08?")
    st.write("‒ What's the total order amount?")
    st.write("‒ What's the total order amount for USA?")
    st.write("‒ Plot order date vs quantity")
    
    st.sidebar.header("Chat History")
    plot_options = ["Bar plot", "Scatter plot", "Histogram", "Box plot"]

    # Display the chat_history in a chat-like format using streamlit-chat
    with st.sidebar:
        histries = st.container()
        for i, (sender, message_text) in enumerate(st.session_state.chat_history):
            if sender == "user":
                histries.chat_message("user").write(message_text)
            else:
                histries.chat_message("assistant").write(message_text['output'])
    with st.container():
        #csv_agent = csvAgent()
        #pandaAgent = pandaFrameAgent([data_orders, data_orderDetails, data_products])
        pythonAstRepelAgent = get_chain(prompt, {'data_products':data_products, 'data_orders': data_orders, 'data_orderDetails': data_orderDetails})
        user_input = st.chat_input("Ask question about Product and Orders")
    # only run if user input is not empty and model and key are selected
        if user_input:
            if not selected_key or not selected_model:
                st.warning("You must provide valid OpenAI API key and choose preferred model", icon="⚠️")
                st.stop()
        # setup request timeout and config list
            llm_config = {
                "timeout": 600,
                "config_list": [
                    {"model": selected_model, "api_key": selected_key},
                ],
                "temperature": 0,  # temperature of 0 means deterministic output
            }
        # create an AssistantAgent instance named "assistant"
            assistant = TrackableAssistantAgent(name="assistant", llm_config=llm_config,
                system_message="For coding tasks, you use etherscan api with this key when needed: QSKXUEAZY7B5ZM9D9W1WYAXEQ4Z5QYD8FP . Wrap the generated code in python __main__ . Use python plot for display if possible. Reply TERMINATE when the task is done.",
                                            )        

        # create a UserProxyAgent instance named "user"
        # human_input_mode is set to "NEVER" to prevent the agent from asking for user input
            user_proxy = TrackableUserProxyAgent(
                name="user",
                human_input_mode="NEVER",
                llm_config=llm_config,
                #is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),
                code_execution_config={"work_dir": "coding", "use_docker": False}
            )

            # Create an event loop: this is needed to run asynchronous functions
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Define an asynchronous function: this is needed to use await
            if "chat_initiated" not in st.session_state:
                st.session_state.chat_initiated = False  # Initialize the session state

            if not st.session_state.chat_initiated:
                def initiate_chat():
                    #answer = csv_agent.run(user_input)
                    #answer = pandaAgent(user_input)

                    try:
                        data = extractionChain.invoke(user_input)
                        action_data = process_data(data)
                        print("action_data")
                        print(action_data)
                        # plot

                    except Exception as e:
                        print("exception:")
                        print(e)

                        # normal operations
                    plotInfo = None
                    if action_data.get('plot') is not None:
                        plotInfo = action_data['plot']
                    else:
                        if action_data.get('show') is not None:
                            plotInfo =  action_data['show']
                    if plotInfo is not None:
                        columnNames = data_products.columns.values.tolist()+data_orders.columns.values.tolist()+data_orderDetails.columns.values.tolist()
                        #print(columnNames)
                        axis = getAxis(plotInfo, columnNames)
                        if axis:
                            print(axis)
                            print(type(axis))
                            
                            x_axis = axis['text'][0]['x_axis']
                            y_axis =  axis['text'][0]['y_axis']
                        else:
                            x_axis = "productName"
                            y_axis = "unitPrice"
                        print(x_axis)
                        st.write("Bar plot:")
                        try:
                            fig, ax = plt.subplots()
                            if x_axis in data_products:
                                xData = data_products
                            elif x_axis in data_orders:
                                xData = data_orders
                            elif x_axis in data_orderDetails: 
                                xData = data_orderDetails
                            else:
                                xData = None
                            if y_axis in data_products:
                                yData = data_products
                            elif y_axis in data_orders:
                                yData = data_orders
                            elif y_axis in data_orderDetails: 
                                yData = data_orderDetails
                            else:
                                yData = None
                            if xData is not None and yData is not None:
                                sns.barplot(x=xData[x_axis], y=yData[y_axis], ax=ax)
                                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
                                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                                st.pyplot(fig)
                            else:
                                raise Exception("Requested data not found in csv file")
                        except Exception as e:    
                            st.write("Sorry cannot plot using the specified axsis")
                    else:
                        answer = pythonAstRepelAgent.invoke({'input':user_input})
                        st.session_state.chat_history.append(("user", user_input))
                        st.session_state.chat_history.append(("agent", answer))
            
                        st.write(answer['output'])

                    st.stop()
                # Run the asynchronous function within the event loop
                loop.run_until_complete(initiate_chat())

                # Close the event loop
                loop.close()

                st.session_state.chat_initiated = True  # Set the state to True after running the chat
    
if __name__ == "__main__":
    main()


Test_notused = """
                async def initiate_chat():
                    #await user_proxy.a_initiate_chat(
                    await user_proxy.a_initiate_chat(    
                        assistant,
                        message=user_input,
                        max_consecutive_auto_reply=5,
                        is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),
                    )
                    st.stop()  # Stop code execution after termination command
"""

Not_used  = """
You have a tool called `person_name_search` through which you can lookup a person by name and find the records corresponding to people with similar name as the query.
You should only really use this if your search term contains a persons name. Otherwise, try to solve it with code.

For example:

<question>How old is Jane?</question>
<logic>Use `person_name_search` since you can use the query `Jane`</logic>

<question>Who has id 320</question>
<logic>Use `python_repl` since even though the question is about a person, you don't know their name so you can't include it.</logic>
"""