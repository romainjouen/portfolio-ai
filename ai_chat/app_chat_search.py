import streamlit as st
from src.utils.chat_interface import ChatInterface
from src.utils.web_search_agent import WebSearchAgent
from src.config.config import Config
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_interface' not in st.session_state:
        st.session_state.chat_interface = ChatInterface()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_model' not in st.session_state:
        st.session_state.current_model = "gemini20"  # Default model key from config
    if 'current_provider' not in st.session_state:
        st.session_state.current_provider = "gemini_gemini20"  # Provider format: family_model
    if 'web_search_agent' not in st.session_state:
        st.session_state.web_search_agent = None  # Initialize as None first
    if 'selected_search_engines' not in st.session_state:
        st.session_state.selected_search_engines = ['duckduckgo', 'brave']  # Default engines

def display_chat_message(role, content):
    """Display a chat message with the appropriate styling"""
    with st.chat_message(role):
        # For assistant messages, remove "assistant:" prefix if present
        if role == "assistant" and content.startswith("assistant:"):
            content = content[len("assistant:"):].lstrip()
        st.markdown(content)

def main():
    st.title("AI Chat Interface")
    
    # Initialize session state
    initialize_session_state()
    
    # Load config
    config = Config()
    
    # Update stats display function definition
    def update_stats(containers):
        current_stats = st.session_state.chat_interface.get_conversation_stats()
        containers['messages'].metric("Messages", current_stats['total_messages'])
        containers['tokens'].metric("Tokens", current_stats['total_tokens'])
        containers['cost'].metric("Cost", f"${current_stats['total_cost']:.5f}")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Settings")
        
        # Model settings section
        st.subheader("Model Settings")
        
        # Two-stage model selection
        model_family = st.selectbox(
            "Select Model Family",
            options=list(config.model_families.keys()),
            format_func=lambda x: config.model_families[x]['description'],
            index=list(config.model_families.keys()).index('gemini') if 'gemini' in config.model_families else 0
        )
        
        # Get available models for selected family
        family_models = config.model_families[model_family]['models']
        model_name = st.selectbox(
            "Select Model",
            options=list(family_models.keys()),
            format_func=lambda x: family_models[x]['description'],
            index=list(family_models.keys()).index('gemini20') if 'gemini20' in family_models else 0
        )
        
        # Create provider string
        provider = f"{model_family}_{model_name}"
        
        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1
        )
        
        # Search Engine Settings
        st.subheader("Search Engine Settings")
        
        available_engines = {
            'brave': 'Brave Search',
            'duckduckgo': 'DuckDuckGo',
            'google': 'Google Search',
            'serper': 'Serper',
            'serpapi': 'SerpAPI'
        }
        
        # Create checkboxes for each engine
        selected_engines = []
        for engine_key, engine_name in available_engines.items():
            is_checked = st.checkbox(
                engine_name,
                value=engine_key in st.session_state.selected_search_engines,
                key=f"checkbox_{engine_key}"
            )
            if is_checked:
                selected_engines.append(engine_key)
        
        # Show warning if no engines selected
        if not selected_engines:
            st.warning("⚠️ Please select at least one search engine")
        
        # Number of search results slider
        st.divider()
        search_results_limit = st.slider(
            "Number of results per search",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Set the maximum number of results to return from each search engine"
        )
        
        # Update selected engines in session state
        if selected_engines != st.session_state.selected_search_engines:
            st.session_state.selected_search_engines = selected_engines
            st.session_state.web_search_agent = WebSearchAgent(
                search_engines=selected_engines,
                results_limit=search_results_limit
            )
        
        # Store current model in session state and initialize/reinitialize components if needed
        if st.session_state.current_provider != provider:
            st.session_state.current_provider = provider
            st.session_state.chat_interface = ChatInterface()
            st.session_state.web_search_agent = WebSearchAgent(
                search_engines=st.session_state.selected_search_engines,
                results_limit=search_results_limit
            )
            st.session_state.messages = []
            st.rerun()
        
        # Initialize web_search_agent if it's None
        if st.session_state.web_search_agent is None:
            st.session_state.web_search_agent = WebSearchAgent(
                search_engines=st.session_state.selected_search_engines,
                results_limit=search_results_limit
            )
        
        # Reset conversation button
        if st.button("Reset Conversation"):
            st.session_state.messages = []
            st.session_state.chat_interface.reset_conversation()
            st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to display
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        
        # Check if this is a search request
        if prompt.strip().startswith("@search"):
            search_query = prompt[7:].strip()  # Remove "@search " prefix
            
            try:
                # Perform web search
                search_results = asyncio.run(st.session_state.web_search_agent.process(search_query))
                
                # Format search results for LLM
                formatted_message = (
                    f"I want you to help me with this search query: {search_query}\n\n"
                    f"Here are the search results from the web:\n\n"
                )
                for idx, result in enumerate(search_results, 1):
                    formatted_message += f"{idx}. {result['title']}\n"
                    formatted_message += f"   URL: {result['url']}\n"
                    formatted_message += f"   {result['snippet']}\n\n"
                
                formatted_message += config.prompts['search']['response_format']
                
                # Get response from LLM
                response = st.session_state.chat_interface.chat(
                    message=formatted_message,
                    provider=provider,
                    temperature=temperature
                )
                
                # Format sources for display
                sources_text = "\n\n---\n**Sources:**\n"
                for idx, result in enumerate(search_results, 1):
                    sources_text += f"{idx}. [{result['title']}]({result['url']})\n"
                
                # Combine response with sources
                response = response + sources_text
                
            except Exception as e:
                response = f"I apologize, but I encountered an error while searching: {str(e)}"
        else:
            # Regular chat response
            response = st.session_state.chat_interface.chat(
                message=prompt,
                provider=provider,
                temperature=temperature
            )
        
        # Add assistant response to display
        st.session_state.messages.append({"role": "assistant", "content": response})
        display_chat_message("assistant", response)

    # Display usage statistics at the bottom
    st.divider()  # Add a visual separator
    current_stats = st.session_state.chat_interface.get_conversation_stats()
    stats_cols = st.columns(4)
    stats_cols[0].metric("Messages", current_stats['total_messages'])
    stats_cols[1].metric("Input Tokens", current_stats['input_tokens'])
    stats_cols[2].metric("Output Tokens", current_stats['output_tokens'])
    stats_cols[3].metric("Total Cost", f"${current_stats['total_cost']:.5f}")

if __name__ == "__main__":
    main()