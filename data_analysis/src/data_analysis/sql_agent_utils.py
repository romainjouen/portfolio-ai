import json
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any, Union
import pandas as pd
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import create_engine, inspect, text


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json file."""
    try:
        with open('conf/config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load config: {str(e)}")
        return {}


class BaseSQLAgent(ABC):
    """Base class for SQL agents with common functionality."""
    
    def __init__(self, db_url: str, model_name: str, semantic_model: Dict[str, Any]):
        self.db_url = db_url
        self.model_name = model_name
        self.semantic_model = semantic_model
        self._model = None
        self._schema = None
    
    @property
    @abstractmethod
    def model(self):
        """Lazy load the model - implementation specific."""
        pass
    
    @property
    def schema(self) -> str:
        """Lazy load and cache the database schema."""
        if self._schema is None:
            self._schema = self._extract_db_schema()
        return self._schema
    
    def _extract_db_schema(self) -> str:
        """Extract database schema from connection."""
        try:
            engine = create_engine(self.db_url)
            inspector = inspect(engine)
            schema = {}
            
            for table in inspector.get_table_names():
                columns = inspector.get_columns(table)
                schema[table] = [col['name'] for col in columns]
                
            return json.dumps(schema, indent=2)
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
            return "{}"
    
    def _format_semantic_info(self) -> Tuple[str, str]:
        """Format semantic model information."""
        tables_info = "\n".join([
            f"- Table: {table['table_name']}\n"
            f"  Description: {table['table_description']}\n"
            f"  Use Case: {table.get('Use Case', 'Not specified')}\n"
            for table in self.semantic_model.get('tables', [])
        ])
        
        relationships_info = "\n".join([
            f"- {rel['from_table']} → {rel['to_table']}: {rel['relationship']}"
            for rel in self.semantic_model.get('relationships', [])
        ])
        
        return tables_info, relationships_info
    
    def _create_sql_prompt(self) -> str:
        """Create the SQL generation prompt template."""
        return """
        You are a SQL generator. Generate ONLY the SQL statement—no explanation.
        Include a semicolon at the end.

        IMPORTANT RULES:
        1. Use EXACT column names from the Database Schema below
        2. For text search with LIKE operations, always use LOWER() function on both column and search value
        3. Example: WHERE LOWER(column_name) LIKE LOWER('%search_term%')
        4. Verify column names exist in the schema before using them
        5. Use appropriate table names and JOIN conditions based on relationships

        Database Schema:
        {schema}

        Tables & Descriptions:
        {tables_info}

        Relationships:
        {relationships_info}

        User question: {query}
        
        Remember: Check the Database Schema above for exact column names, then apply LOWER() for text searches.
        Output (SQL only):
        """
    
    def _clean_sql_output(self, text: Union[str, Any]) -> str:
        """Clean and format SQL output."""
        # Handle different response formats
        if hasattr(text, 'content'):
            text = text.content
        elif not isinstance(text, str):
            text = str(text)
        
        # Remove code blocks
        cleaned = re.sub(r"```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"```", "", cleaned)
        
        # Ensure semicolon ending
        cleaned = cleaned.strip()
        if not cleaned.endswith(';'):
            cleaned += ';'
            
        return cleaned
    
    @abstractmethod
    def generate_sql(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Generate SQL from natural language query - model specific."""
        pass
    
    def execute_sql(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        try:
            # Clean query
            sql_query = re.sub(r'^```sql\s*', '', sql_query.strip())
            sql_query = re.sub(r'\s*```$', '', sql_query)
            
            if not sql_query.rstrip().endswith(';'):
                sql_query += ';'
            
            # Store for debugging
            st.session_state['last_executed_sql'] = sql_query
            
            # Execute
            engine = create_engine(self.db_url)
            with engine.connect() as conn:
                result = conn.execute(text(sql_query))
                df = pd.DataFrame(result.fetchall())
                if not df.empty:
                    df.columns = result.keys()
                return df
                
        except Exception as e:
            st.error(f"SQL Execution Error: {str(e)}")
            st.code(sql_query, language="sql", caption="Failed Query")
            return pd.DataFrame()





def create_display_function(agent_class, provider_name: str):
    """Factory function to create display functions for specific providers."""
    def display_sql_agent(db_url: str, model_name: str, semantic_model: Dict[str, Any]) -> None:
        """Display the SQL agent interface."""
        try:
            from ..ui_components.ui import SQLAgentUI
            agent = agent_class(db_url, model_name, semantic_model)
            ui = SQLAgentUI(agent, provider_name)
            ui.display()
        except Exception as e:
            st.error(f"❌ SQL Agent Error: {str(e)}")
            st.info(f"🔧 Check your database connection and {provider_name} configuration")
    
    return display_sql_agent 