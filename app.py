import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import os
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent

load_dotenv()

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

def main():
    st.title("Machine Learning")

    st.header("Upload your CSV data file")
    data_file = st.file_uploader("Upload CSV", type=["csv"])

    if data_file is not None:
        data = pd.read_csv(data_file)
    else:  # Add this block
        # Replace with your fixed file path
        fixed_file_path = os.getcwd()+"/moviesDB.csv"  
        data = pd.read_csv(fixed_file_path)        
    st.write("Data overview:")
    st.write(data.head())

    st.sidebar.header("Visualizations")
    plot_options = ["Bar plot", "Scatter plot", "Histogram", "Box plot"]

    with st.container():
        user_input = st.text_input("User Input")
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

                async def initiate_chat():
                    #await user_proxy.a_initiate_chat(
                    await user_proxy.a_initiate_chat(    
                        assistant,
                        message=user_input,
                        max_consecutive_auto_reply=5,
                        is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),
                    )
                    st.stop()  # Stop code execution after termination command

                # Run the asynchronous function within the event loop
                loop.run_until_complete(initiate_chat())

                # Close the event loop
                loop.close()

                st.session_state.chat_initiated = True  # Set the state to True after running the chat
    
if __name__ == "__main__":
    main()