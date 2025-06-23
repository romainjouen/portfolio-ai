import json
import re
import time
import io
import sys
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any, Union
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


def load_config(file_path: str = 'conf/config.json') -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Failed to load config from {file_path}: {str(e)}")
        return {}


class BasePythonAgent(ABC):
    """Base class for Python agents with common functionality."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self._get_default_model_name()
        self._model = None
    
    @abstractmethod
    def _get_default_model_name(self) -> str:
        """Get the default model name for this provider."""
        pass
    
    @property
    @abstractmethod
    def model(self):
        """Lazy load the model - implementation specific."""
        pass
    
    def _analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze DataFrame structure and content."""
        return {
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "sample_rows": df.head(3).to_dict(orient="records"),
            "shape": df.shape,
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"
        }
    
    def _create_python_prompt(self) -> str:
        """Create the Python code generation prompt template."""
        return """
        You are a Python data science expert. Generate ONLY executable Python code.
        
        DataFrame Info:
        - Columns: {df_columns}
        - Types: {df_dtypes}  
        - Sample: {df_sample}
        - Shape: {df_shape}
        
        Requirements:
        1. Use the existing DataFrame variable 'df'
        2. Include necessary imports at the top
        3. Use streamlit-compatible outputs (st.pyplot(), st.plotly_chart(), etc.)
        4. Create plt.figure() before matplotlib plots
        5. Use clear variable names and add comments
        6. Handle potential errors gracefully
        
        User request: {query}
        
        Return ONLY Python code:
        ```python
        """
    
    def _extract_code(self, text: Union[str, Any]) -> str:
        """Extract clean Python code from model response."""
        # Handle different response formats
        if hasattr(text, 'content'):
            text = text.content
        elif not isinstance(text, str):
            text = str(text)
        
        # Extract code from markdown blocks
        code_match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
            
        # Fallback to cleaned text
        return text.strip()
    
    def _create_safe_namespace(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create a safe execution namespace with necessary imports and data."""
        return {
            # Core libraries
            'df': df,
            'pd': pd,
            'plt': plt,
            'sns': sns,
            'st': st,
            
            # Additional useful libraries
            'np': __import__('numpy'),
            'json': json,
            'time': time,
            're': re,
            
            # Plotting libraries
            'plotly': __import__('plotly.express', fromlist=['express']),
            'px': __import__('plotly.express', fromlist=['express']),
            
            # ML libraries (optional, with error handling)
            **self._import_optional_libraries()
        }
    
    def _import_optional_libraries(self) -> Dict[str, Any]:
        """Import optional libraries with error handling."""
        optional_libs = {}
        
        # Try to import sklearn
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error
            optional_libs.update({
                'train_test_split': train_test_split,
                'LinearRegression': LinearRegression,
                'mean_squared_error': mean_squared_error
            })
        except ImportError:
            pass
            
        return optional_libs
    
    @abstractmethod
    def generate_code(self, df: pd.DataFrame, user_query: str) -> Tuple[str, Dict[str, Any]]:
        """Generate Python code based on DataFrame and user query - model specific."""
        pass
    
    def execute_code(self, code: str, df: pd.DataFrame) -> Tuple[bool, str]:
        """Execute Python code in a controlled environment."""
        # Capture stdout
        old_stdout = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # Create safe namespace
            namespace = self._create_safe_namespace(df)
            
            # Execute code
            exec(code, namespace)
            
            # Get captured output
            output = captured_output.getvalue()
            return True, output
            
        except Exception as e:
            error_msg = f"Execution Error: {type(e).__name__}: {str(e)}"
            return False, error_msg
            
        finally:
            # Restore stdout
            sys.stdout = old_stdout


def create_display_function(agent_class, provider_name: str):
    """Factory function to create display functions for specific providers."""
    def display_python_agent(sql_results_df: Optional[pd.DataFrame]) -> None:
        """Display the Python agent interface."""
        try:
            from ..ui_components.ui import PythonAgentUI
            agent = agent_class()
            ui = PythonAgentUI(agent, provider_name)
            ui.display(sql_results_df)
        except Exception as e:
            st.error(f"❌ Python Agent Error: {str(e)}")
            st.info(f"🔧 Check your {provider_name} configuration and dependencies")
    
    return display_python_agent 