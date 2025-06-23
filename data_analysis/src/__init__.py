"""
Main package for the Text to SQL and Python Data Analysis application.

This package provides unified agents for SQL generation and Python code generation,
along with their corresponding UI components.
"""

# Import from data_analysis package
from .data_analysis import PythonAgent, SQLAgent, BasePythonAgent, BaseSQLAgent, load_config

# Import UI components
from .ui_components.ui import PythonAgentUI, SQLAgentUI, LLMProviderSelector, ConfigurationUI

# Import utilities
from .utils.output_manager import OutputManager

__all__ = [
    # Core agents
    'PythonAgent',
    'SQLAgent',
    
    # UI components
    'PythonAgentUI', 
    'SQLAgentUI', 
    'LLMProviderSelector',
    'ConfigurationUI',
    
    # Base classes and utilities
    'BasePythonAgent',
    'BaseSQLAgent',
    'load_config',
    'OutputManager'
] 