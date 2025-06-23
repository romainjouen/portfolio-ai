import streamlit as st
import os

# Import required components from our SQL system
from openai import OpenAI
from swarm import Swarm
from sql_agents import sql_router_agent, rss_feed_agent, user_agent, analytics_agent


def init_agents(selected_llm: str):
    # Update the model for all agents
    for agent in [sql_router_agent, rss_feed_agent, user_agent, analytics_agent]:
        agent.model = selected_llm


# Initialize conversation state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.title('SQL Swarm Agent UI')

# Sidebar for LLM selection
st.sidebar.header('LLM Selection')
selected_llm = st.sidebar.selectbox('Choose LLM', ['qwen2.5-coder:7b', 'qwen2.5:3b'])
init_agents(selected_llm)

st.markdown('This interface allows you to interact with the SQL system. Enter your query below and the appropriate SQL agent will respond.')

# User input form
with st.form(key='query_form'):
    user_query = st.text_input('Enter your SQL query or question:')
    submit_button = st.form_submit_button(label='Send Query')

if submit_button and user_query:
    # Append user's query to the conversation history
    st.session_state['messages'].append({"role": "user", "content": user_query})

    # Initialize the client with ollama settings
    ollama_client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
    client = Swarm(client=ollama_client)

    # Run the agent starting with the router agent
    response = client.run(
        agent=sql_router_agent,
        messages=st.session_state['messages'],
        context_variables={},
        stream=False,
        debug=False
    )

    # Append the agent's response messages
    st.session_state['messages'].extend(response.messages)
    st.experimental_rerun()

st.markdown('---')
st.header('Conversation')

# Display conversation history
for message in st.session_state['messages']:
    role = message.get('role', '').capitalize()
    content = message.get('content', '')
    if role == 'User':
        st.markdown(f"**User:** {content}")
    elif role == 'Assistant':
        st.markdown(f"**Assistant:** {content}")
    else:
        st.markdown(f"**{role}:** {content}") 