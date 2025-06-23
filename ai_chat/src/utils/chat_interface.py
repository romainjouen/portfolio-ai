from typing import List, Dict, Optional
from .llm_provider import LLMProvider
import json

class ChatInterface:
    def __init__(self, default_provider: str = "gemini-pro"):
        """Initialize chat interface with a default provider"""
        self.llm_provider = LLMProvider()
        self.default_provider = default_provider
        self.conversation_history: List[Dict] = []
        
    def reset_conversation(self):
        """Clear the conversation history"""
        self.conversation_history = []
        
    def add_system_message(self, content: str):
        """Add a system message to the conversation"""
        self.conversation_history.append({
            "role": "system",
            "content": content
        })
        
    def add_user_message(self, content: str):
        """Add a user message to the conversation"""
        self.conversation_history.append({
            "role": "user",
            "content": content
        })
        
    def add_assistant_message(self, content: str):
        """Add an assistant message to the conversation"""
        self.conversation_history.append({
            "role": "assistant",
            "content": content
        })

    def chat(self, message: str, provider: Optional[str] = None, temperature: float = 0.7) -> str:
        """
        Send a message and get a response using the specified provider
        
        Args:
            message: The user's message
            provider: The LLM provider to use (defaults to self.default_provider)
            temperature: Temperature setting for response generation
            
        Returns:
            The assistant's response
        """
        try:
            # Use default provider if none specified
            provider = provider or self.default_provider
            
            # Add user message to history
            self.add_user_message(message)
            
            # Get response from LLM
            response = self.llm_provider.process_query(
                provider=provider,
                messages=self.conversation_history,
                temperature=temperature
            )
            
            if response:
                # Add assistant response to history
                self.add_assistant_message(response)
                return response
            else:
                error_msg = f"Failed to get response from {provider}"
                print(error_msg)
                return error_msg
                
        except Exception as e:
            error_msg = f"Error in chat: {str(e)}"
            print(error_msg)
            return error_msg
            
    def get_conversation_stats(self) -> Dict:
        """Get statistics about the current conversation"""
        return {
            "total_messages": len(self.conversation_history),
            "input_tokens": self.llm_provider.input_tokens,
            "output_tokens": self.llm_provider.output_tokens,
            "total_cost": round(self.llm_provider.cumulative_cost, 5)
        }