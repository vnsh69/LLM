import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent,initialize_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine, text
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain_community.llms import Ollama
import sqlite3

st.set_page_config(page_title="chat with your database")
st.title("chat with your database")

MYSQL = "USE_MYSQL"

dq_url = MYSQL
mysql_host = st.sidebar.text_input("enter your host name")
mysql_user = st.sidebar.text_input("enter your username")
mysql_password = st.sidebar.text_input("enter your password", type="password")
create_new = st.checkbox("Create a new database")
mysql_db = st.sidebar.text_input("enter your database name")

llm_choice = st.sidebar.selectbox("Choose LLM Provider", ["Ollama (local)", "Groq (cloud)"])
llm = None

if llm_choice == "Groq (cloud)":
    api_key = st.sidebar.text_input("Enter your Groq API key", type="password")
    if not api_key:
        st.warning("Please enter your Groq API key")
        st.stop()
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192")

elif llm_choice == "Ollama (local)":
    ollama_model = st.sidebar.selectbox("Choose Ollama model", ["gemma3:latest", "mistral:latest", "deepseek-r1:14b"])
    llm = Ollama(model=ollama_model)

def execute_sql(query: str):
    return db.run(query)

@st.cache_resource(ttl = "2h")
def configure_db(dq_url, mysql_host, mysql_user, mysql_password, mysql_db):
    if not (mysql_host and mysql_user and mysql_password and mysql_db):
        st.error("pls provide all info")
        st.stop()
    if not create_new:
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))
    if create_new:
        engine = create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}")
        with engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE {mysql_db}"))
        st.success(f"Database '{mysql_db}' created successfully!")
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))


db = configure_db(dq_url, mysql_host, mysql_user, mysql_password, mysql_db)

custom_tool = Tool.from_function(
    name="run_sql_query",
    func=execute_sql,
    description="Executes raw SQL queries including INSERT, UPDATE, DELETE, SELECT."
)

toolkit = SQLDatabaseToolkit(db = db, llm=llm).get_tools()

all_tools = toolkit + [custom_tool]

agent = initialize_agent(
    llm =llm,
    tools = all_tools,
    verbose = True,
    agent_type = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True
)

if "messages" not in st.session_state or st.sidebar.button("clear history"):
    st.session_state.messages = [{"role":"assistant", "content":"how can i help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
user_query = st.chat_input(placeholder = "ask or create database with ai agent")

if user_query:
    st.session_state.messages.append({"role":"user", "content": user_query})
    st.chat_message("user").write(user_query)
    response = agent.run(user_query)
    st.session_state.messages.append({"role":"assistant", "content": response})
    st.chat_message("assistant").write(response)
    