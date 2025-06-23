import streamlit as st
from src.utils.chat_interface import ChatInterface
from src.config.config import Config
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_interface' not in st.session_state:
        st.session_state.chat_interface = ChatInterface()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_model' not in st.session_state:
        st.session_state.current_model = "Gemini Pro"  # Default model

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
        
        # Two-stage model selection
        model_family = st.selectbox(
            "Select Model Family",
            options=list(config.model_families.keys()),
            format_func=lambda x: config.model_families[x]['description']
        )
        
        # Get available models for selected family
        family_models = config.model_families[model_family]['models']
        model_name = st.selectbox(
            "Select Model",
            options=list(family_models.keys()),
            format_func=lambda x: family_models[x]['description']
        )
        
        # Create provider string
        provider = f"{model_family}_{model_name}"
        
        # Store current model in session state
        if 'current_provider' not in st.session_state or st.session_state.current_provider != provider:
            st.session_state.current_provider = provider
            st.session_state.chat_interface = ChatInterface()
            st.session_state.messages = []
            st.rerun()
        
        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1
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
        
        # Get model response
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