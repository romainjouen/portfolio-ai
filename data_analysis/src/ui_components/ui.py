import pandas as pd
import streamlit as st
from typing import Optional, Dict, Tuple, Any

from ..data_analysis.python_agent_utils import BasePythonAgent, load_config
from ..data_analysis.python_agent import PythonAgent
from ..data_analysis.sql_agent_utils import BaseSQLAgent
from ..data_analysis.sql_agent import SQLAgent


class ConfigurationUI:
    """Configuration tab for global LLM provider and model selection."""
    
    def __init__(self):
        self.config = load_config()
        self.models_config = self.config.get('models', {})
        
        # Initialize session state for global configuration (single model for both agents)
        if 'global_model' not in st.session_state:
            st.session_state.global_model = None
        if 'global_provider' not in st.session_state:
            st.session_state.global_provider = None
    
    def _get_provider_models(self, provider: str) -> list:
        """Get available models for a specific provider."""
        if provider not in self.models_config:
            return []
        
        provider_config = self.models_config[provider]
        models = [provider_config.get('default', '')]
        
        if provider == 'ollama':
            # Handle Ollama alternatives with potential commented models
            alternatives = provider_config.get('alternatives', [])
            for alt in alternatives:
                if isinstance(alt, str):
                    # Split by spaces and filter out commented models
                    clean_models = [m.strip() for m in alt.split() if m.strip() and not m.startswith('//')]
                    models.extend(clean_models)
        else:
            # Handle regular alternatives list
            models.extend(provider_config.get('alternatives', []))
        
        # Remove duplicates while preserving order
        seen = set()
        return [x for x in models if x and not (x in seen or seen.add(x))]
    
    def _display_model_selector(self, providers: list) -> tuple:
        """Display provider and model selection for both agents."""
        st.subheader("🤖 AI Model Configuration")
        st.markdown("Select one provider and model to use for both SQL and Python agents.")
        
        # Provider selection
        current_provider = st.session_state.get('global_provider', providers[0] if providers else None)
        
        selected_provider = st.selectbox(
            "Select AI Provider:",
            providers,
            index=providers.index(current_provider) if current_provider in providers else 0,
            key="provider_select",
            help="Choose the AI provider for all operations"
        )
        
        # Model selection
        available_models = self._get_provider_models(selected_provider)
        if not available_models:
            st.warning(f"⚠️ No models configured for {selected_provider}")
            return selected_provider, None, None
        
        current_model = st.session_state.get('global_model')
        
        # Find current model index or use default
        model_index = 0
        if current_model in available_models:
            model_index = available_models.index(current_model)
        
        selected_model = st.selectbox(
            "Select AI Model:",
            available_models,
            index=model_index,
            key="model_select",
            help="Choose the specific model for all operations"
        )
        
        # Show default model info
        default_model = self.models_config.get(selected_provider, {}).get('default', 'N/A')
        if selected_model == default_model:
            st.info(f"📌 Using default model for {selected_provider}")
        
        # Temperature selection
        st.markdown("---")
        st.subheader("🌡️ Temperature Setting")
        
        # Get current temperature from session state or config
        current_temperature = st.session_state.get('global_temperature', self.config.get('temperature', 0.0))
        
        selected_temperature = st.slider(
            "Temperature:",
            min_value=0.0,
            max_value=2.0,
            value=float(current_temperature),
            step=0.1,
            key="temperature_select",
            help="Controls randomness: 0.0 = deterministic, 2.0 = very creative"
        )
        
        # Show temperature info
        if selected_temperature == 0.0:
            st.info("🎯 Deterministic mode - most consistent results")
        elif selected_temperature <= 0.3:
            st.info("📐 Low creativity - focused and precise")
        elif selected_temperature <= 0.7:
            st.info("⚖️ Balanced - good mix of consistency and creativity")
        elif selected_temperature <= 1.0:
            st.info("🎨 Creative - more varied responses")
        else:
            st.info("🌟 Very creative - highly varied and experimental")
        
        return selected_provider, selected_model, selected_temperature
    
    def _test_database_connection(self, db_url: str) -> None:
        """Test database connection and display results."""
        try:
            import sqlalchemy
            from sqlalchemy import create_engine, text
            
            with st.spinner("🔄 Testing database connection..."):
                # Create engine and test connection
                engine = create_engine(db_url)
                with engine.connect() as conn:
                    # Simple test query
                    result = conn.execute(text("SELECT 1 as test"))
                    test_value = result.fetchone()[0]
                    
                    if test_value == 1:
                        # Get database version for the success message
                        try:
                            version_result = conn.execute(text("SELECT version()"))
                            version = version_result.fetchone()[0].split(',')[0]
                            st.success(f"✅ Database connection successful!\n{version}")
                        except:
                            st.success("✅ Database connection successful!")
                    else:
                        st.error("❌ Connection test failed: Unexpected result")
                        
        except Exception as e:
            st.error(f"❌ Database connection failed: {str(e)}")
            
            # Provide helpful error messages
            error_str = str(e).lower()
            if "connection refused" in error_str:
                st.info("💡 Make sure the database server is running and accessible")
            elif "authentication failed" in error_str or "password" in error_str:
                st.info("💡 Check your database credentials in the .env file")
            elif "database" in error_str and "does not exist" in error_str:
                st.info("💡 Make sure the specified database exists")
            elif "timeout" in error_str:
                st.info("💡 Check network connectivity and firewall settings")
    
    def display(self, db_url: str = None) -> Dict[str, Any]:
        """Display the configuration interface and return current settings."""
        st.header("⚙️ Configuration")
        st.markdown("Configure your AI provider and model. The same configuration will be used for both SQL and Python agents.")
        
        # Get available providers
        available_providers = list(self.models_config.keys())
        if not available_providers:
            st.error("❌ No providers configured. Please check your config.json file.")
            return {}
        
        # Single model selector for both agents
        selected_provider, selected_model, selected_temperature = self._display_model_selector(available_providers)
        
        # Auto-save configuration when selections are made
        current_provider = st.session_state.get('global_provider')
        current_model = st.session_state.get('global_model')
        current_temperature = st.session_state.get('global_temperature')
        
        # Check if configuration has changed
        config_changed = (selected_provider != current_provider or 
                         selected_model != current_model or 
                         selected_temperature != current_temperature)
        
        if config_changed and selected_provider and selected_model and selected_temperature is not None:
            # Update session state automatically
            st.session_state.global_provider = selected_provider
            st.session_state.global_model = selected_model
            st.session_state.global_temperature = selected_temperature
            
            # Clear existing agents to force recreation with new config
            if 'sql_agent' in st.session_state:
                st.session_state.sql_agent = None
            if 'python_agent' in st.session_state:
                st.session_state.python_agent = None
            
            # Show confirmation
            st.success(f"✅ Configuration updated! Using {selected_provider.upper()} - {selected_model} (Temperature: {selected_temperature})")
        
        # Database connection information (after AI configuration)
        if db_url:
            st.markdown("---")
            st.subheader("🗄️ Database Connection")
            # Parse database URL to show connection details safely
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(db_url)
                
                # Create a safe display URL (without password)
                safe_url = f"{parsed_url.scheme}://{parsed_url.username}:***@{parsed_url.hostname}:{parsed_url.port}{parsed_url.path}"
                
                # Show only connection string
                st.code(safe_url, language="text")
                
                # Test connection button
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("🔍 Test Connection", use_container_width=True):
                        self._test_database_connection(db_url)
                    
            except Exception as e:
                st.warning(f"⚠️ Could not parse database URL: {str(e)}")
        
        # Display current configuration summary
        st.markdown("---")
        with st.expander("📋 Current Configuration Summary"):
            config_summary = {
                "AI Configuration": {
                    "Provider": st.session_state.get('global_provider', 'Not set'),
                    "Model": st.session_state.get('global_model', 'Not set'),
                    "Temperature": st.session_state.get('global_temperature', 'Not set'),
                    "Usage": "Both SQL and Python agents will use this configuration"
                }
            }
            
            # Add database info if available
            if db_url:
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(db_url)
                    config_summary["Database Configuration"] = {
                        "Host": parsed_url.hostname,
                        "Port": parsed_url.port,
                        "Database": parsed_url.path.lstrip('/'),
                        "User": parsed_url.username,
                        "Status": "Connected" if db_url else "Not configured"
                    }
                except:
                    config_summary["Database Configuration"] = {
                        "Status": "Configuration error"
                    }
            else:
                config_summary["Database Configuration"] = {
                    "Status": "Not configured"
                }
                
            st.json(config_summary)
        
        # Return current configuration
        return {
            'provider': st.session_state.get('global_provider'),
            'model': st.session_state.get('global_model')
        }
    
    def create_sql_agent(self, db_url: str, semantic_model: Dict[str, Any]) -> Optional[SQLAgent]:
        """Create SQL agent using global configuration."""
        model = st.session_state.get('global_model')
        temperature = st.session_state.get('global_temperature')
        if not model or temperature is None:
            return None
        return SQLAgent(db_url, model, semantic_model, temperature)
    
    def create_python_agent(self) -> Optional[PythonAgent]:
        """Create Python agent using global configuration."""
        model = st.session_state.get('global_model')
        temperature = st.session_state.get('global_temperature')
        if not model or temperature is None:
            return None
        return PythonAgent(model, temperature)
    
    def has_configuration(self) -> bool:
        """Check if global configuration is set."""
        return (st.session_state.get('global_model') is not None and 
                st.session_state.get('global_provider') is not None and
                st.session_state.get('global_temperature') is not None)


class LLMProviderSelector:
    """Simple class for LLM provider and model selection for both Python and SQL agents."""
    
    def __init__(self, agent_type: str = "python", db_url: str = None, semantic_model: Dict[str, Any] = None):
        self.agent_type = agent_type
        self.db_url = db_url
        self.semantic_model = semantic_model
        self.config = load_config()
        self.models_config = self.config.get('models', {})
        
        # Get OpenAI models
        openai_config = self.models_config.get('openai', {})
        self.openai_models = [openai_config.get('default', 'gpt-4o-mini')]
        self.openai_models.extend(openai_config.get('alternatives', []))
        
        # Get Ollama models
        ollama_config = self.models_config.get('ollama', {})
        self.ollama_models = [ollama_config.get('default', 'qwen2.5-coder:32b')]
        
        # Process alternatives (filter out commented models and clean up)
        alternatives = ollama_config.get('alternatives', [])
        for alt in alternatives:
            if isinstance(alt, str) and not alt.strip().startswith('//'):
                # Split by spaces and filter out commented models
                models = [m.strip() for m in alt.split() if m.strip() and not m.startswith('//')]
                self.ollama_models.extend(models)
        
        # Remove duplicates while preserving order
        seen = set()
        self.ollama_models = [x for x in self.ollama_models if not (x in seen or seen.add(x))]
    
    def display_model_selection(self):
        """Display model selection UI and return configured agent."""
        agent_name = "Python" if self.agent_type == "python" else "SQL"
        st.subheader(f"🤖 Select AI Model for {agent_name} Agent")
        
        # Show configuration info
        if not self.openai_models and not self.ollama_models:
            st.error("❌ No models configured. Please check your config.json file.")
            return None
        
        # Provider tabs
        openai_tab, ollama_tab = st.tabs(["🌐 OpenAI", "🏠 Ollama"])
        
        with openai_tab:
            if self.openai_models:
                # Show default model info
                default_model = self.models_config.get('openai', {}).get('default', 'N/A')
                st.info(f"📌 Default: {default_model}")
                
                selected_openai = st.selectbox(
                    "Choose OpenAI Model:",
                    self.openai_models,
                    key=f"{self.agent_type}_openai_model"
                )
                if st.button("Use OpenAI Model", key=f"use_{self.agent_type}_openai"):
                    return self._create_agent(selected_openai)
            else:
                st.warning("⚠️ No OpenAI models configured")
        
        with ollama_tab:
            if self.ollama_models:
                # Show default model info
                default_model = self.models_config.get('ollama', {}).get('default', 'N/A')
                st.info(f"📌 Default: {default_model}")
                
                selected_ollama = st.selectbox(
                    "Choose Ollama Model:",
                    self.ollama_models,
                    key=f"{self.agent_type}_ollama_model"
                )
                if st.button("Use Ollama Model", key=f"use_{self.agent_type}_ollama"):
                    return self._create_agent(selected_ollama)
            else:
                st.warning("⚠️ No Ollama models configured")
        
        return None
    
    def _create_agent(self, model_name: str):
        """Create the appropriate agent based on agent_type."""
        if self.agent_type == "python":
            return PythonAgent(model_name)
        elif self.agent_type == "sql":
            if not self.db_url or not self.semantic_model:
                st.error("❌ Database URL and semantic model required for SQL agent")
                return None
            return SQLAgent(self.db_url, model_name, self.semantic_model)
        else:
            st.error(f"❌ Unknown agent type: {self.agent_type}")
            return None



class SQLAgentUI:
    """Common Streamlit UI for SQL Agents."""
    
    def __init__(self, agent: BaseSQLAgent, provider_name: str = "", output_manager=None):
        self.agent = agent
        self.provider_name = provider_name
        self.output_manager = output_manager
    
    def _display_metadata(self, metadata: Dict[str, Any]) -> None:
        """Display process information in a clean format."""
        st.subheader("📊 Process Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Duration", f"{metadata['duration_seconds']}s")
            st.text(f"🕐 Start: {metadata['start_time']}")
            st.text(f"🕑 End: {metadata['end_time']}")
            
        with col2:
            st.metric("Input Tokens", f"~{metadata['input_tokens']}")
            st.metric("Output Tokens", f"~{metadata['output_tokens']}")
    
    def _display_results(self, df: pd.DataFrame) -> None:
        """Display query results with summary."""
        if not df.empty:
            st.subheader("📋 Query Results")
            st.dataframe(df, use_container_width=True)
            st.success(f"✅ Retrieved {len(df)} rows")
        else:
            st.warning("⚠️ Query executed but returned no results.")
    
    def display(self) -> None:
        """Display the complete SQL agent interface."""
        header_text = "🔍 SQL Generator"
        if self.provider_name:
            header_text += f" ({self.provider_name})"
        st.header(header_text)
        
        # Initialize session state for SQL generation control
        if 'sql_generating' not in st.session_state:
            st.session_state.sql_generating = False
        if 'sql_stop_requested' not in st.session_state:
            st.session_state.sql_stop_requested = False
        
        # Query input
        query = st.text_area(
            "What data would you like to retrieve?",
            placeholder="E.g., Show me all articles liked between April 13-15, 2023",
            help="Describe your data needs in natural language"
        )
        
        if not query:
            return
        
        # Generation controls
        col1, col2 = st.columns([1, 1])
        with col1:
            generate_btn = st.button("🤖 Generate SQL", type="primary", disabled=st.session_state.sql_generating)
        with col2:
            if st.session_state.sql_generating:
                if st.button("🛑 Stop Generation", type="secondary"):
                    st.session_state.sql_stop_requested = True
                    st.rerun()
        
        # Generate SQL
        if generate_btn:
            st.session_state.sql_generating = True
            st.session_state.sql_stop_requested = False
            st.rerun()
        
        # Handle SQL generation process
        if st.session_state.sql_generating and not st.session_state.sql_stop_requested:
            with st.spinner("🤖 Generating SQL..."):
                try:
                    sql, metadata = self.agent.generate_sql(query)
                    st.session_state.sql_generating = False
                    
                    if not sql:
                        st.error("Failed to generate SQL")
                        return
                    
                    # Store generated SQL in session state
                    st.session_state.generated_sql = sql
                    st.session_state.sql_metadata = metadata
                    st.rerun()
                    
                except Exception as e:
                    st.session_state.sql_generating = False
                    st.error(f"Error generating SQL: {str(e)}")
                    return
        
        # Handle stop request
        if st.session_state.sql_stop_requested:
            st.session_state.sql_generating = False
            st.session_state.sql_stop_requested = False
            st.warning("🛑 SQL generation stopped by user")
            st.rerun()
        
        # Display generated SQL if available
        if hasattr(st.session_state, 'generated_sql') and st.session_state.generated_sql:
            sql = st.session_state.generated_sql
            metadata = getattr(st.session_state, 'sql_metadata', {})
            
            # Display and edit SQL
            st.subheader("📝 Generated SQL")
            sql_edit = st.text_area(
                "Review and edit if needed:",
                value=sql,
                height=200,
                help="You can modify the SQL before execution"
            )
            
            # Execute controls
            col1, col2 = st.columns([1, 3])
            with col1:
                execute_btn = st.button("🚀 Execute SQL", type="primary")
            with col2:
                st.info("💡 Review the SQL above before executing")
            
            # Execute and display results
            if execute_btn:
                with st.spinner("⚡ Executing query..."):
                    results_df = self.agent.execute_sql(sql_edit)
                    st.session_state.sql_results = results_df
                    self._display_results(results_df)
                    
                    # Save results if output manager is available
                    if self.output_manager:
                        result_summary = f"Retrieved {len(results_df)} rows" if not results_df.empty else "Query executed but returned no results"
                        
                        # Add temperature to metadata if available
                        if hasattr(self.agent, 'temperature') and self.agent.temperature is not None:
                            metadata['temperature'] = self.agent.temperature
                        
                        saved_file = self.output_manager.save_sql_result(
                            user_query=query,
                            generated_sql=sql,
                            metadata=metadata,
                            executed_sql=sql_edit if sql_edit != sql else None,
                            result_summary=result_summary
                        )
                        st.success(f"💾 Results saved to: `{saved_file}`")
            
            # Display metadata
            self._display_metadata(metadata)


class PythonAgentUI:
    """Common Streamlit UI for Python Agents."""
    
    def __init__(self, agent: BasePythonAgent, provider_name: str = "", output_manager=None):
        self.agent = agent
        self.provider_name = provider_name
        self.output_manager = output_manager
    
    def _display_dataframe_info(self, df: pd.DataFrame) -> None:
        """Display DataFrame information in an expandable section."""
        with st.expander("📊 Data Preview & Info"):
            st.subheader("Sample Data")
            st.dataframe(df.head(5), use_container_width=True)
    
    def _display_metadata(self, metadata: Dict[str, Any]) -> None:
        """Display process information."""
        st.subheader("⚡ Process Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Duration", f"{metadata['duration_seconds']}s")
            
        with col2:
            st.metric("Input Tokens", f"~{metadata['input_tokens']}")
            
        with col3:
            st.metric("Output Tokens", f"~{metadata['output_tokens']}")
        
        # Timing and model details
        st.text(f"🕐 {metadata['start_time']} → 🕑 {metadata['end_time']}")
        st.text(f"🤖 Model: {metadata.get('model_name', 'Unknown')}")
    
    def _display_code_section(self, code: str) -> Tuple[str, bool]:
        """Display code with editing capabilities."""
        st.subheader("🐍 Generated Python Code")
        
        # Code editor
        edited_code = st.text_area(
            "Review and edit the code:",
            value=code,
            height=300,
            help="Modify the code as needed before execution"
        )
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            execute_btn = st.button("🚀 Execute Code", type="primary")
            
        with col2:
            st.download_button(
                label="📥 Download",
                data=edited_code,
                file_name="generated_analysis.py",
                mime="text/plain"
            )
            
        with col3:
            st.info("💡 Review the code before executing")
        
        return edited_code, execute_btn
    
    def _display_execution_results(self, success: bool, output: str) -> None:
        """Display code execution results."""
        if success:
            if output.strip():
                with st.expander("📄 Console Output"):
                    st.text(output)
            st.success("✅ Code executed successfully!")
        else:
            st.error(f"❌ {output}")
    
    def display(self, df: Optional[pd.DataFrame]) -> None:
        """Display the complete Python agent interface."""
        header_text = f"🐍 Python Data Agent"
        if self.provider_name:
            header_text += f" ({self.provider_name})"
        st.header(header_text)
        
        # Check for data
        if df is None or df.empty:
            st.warning("⚠️ No data available. Run a SQL query first.")
            return
        
        # Initialize session state for Python generation control
        if 'python_generating' not in st.session_state:
            st.session_state.python_generating = False
        if 'python_stop_requested' not in st.session_state:
            st.session_state.python_stop_requested = False
        
        # Display data info
        self._display_dataframe_info(df)
        
        # Query input
        user_query = st.text_area(
            "What analysis would you like to perform?",
            placeholder="E.g., 'Create a bar chart showing article interactions by type', 'Find correlations in the data', 'Predict future trends'",
            help="Describe your analysis needs in natural language"
        )
        
        if not user_query:
            return
        
        # Generation controls
        col1, col2 = st.columns([1, 1])
        with col1:
            generate_btn = st.button("🤖 Generate Code", type="primary", disabled=st.session_state.python_generating)
        with col2:
            if st.session_state.python_generating:
                if st.button("🛑 Stop Generation", type="secondary"):
                    st.session_state.python_stop_requested = True
                    st.rerun()
        
        # Generate code
        if generate_btn:
            st.session_state.python_generating = True
            st.session_state.python_stop_requested = False
            st.rerun()
        
        # Handle Python code generation process
        if st.session_state.python_generating and not st.session_state.python_stop_requested:
            with st.spinner("🤖 Generating Python code..."):
                try:
                    code, metadata = self.agent.generate_code(df, user_query)
                    st.session_state.python_generating = False
                    
                    if not code:
                        st.error("Failed to generate code")
                        return
                    
                    # Store generated code in session state
                    st.session_state.generated_code = code
                    st.session_state.python_metadata = metadata
                    st.session_state.current_user_query = user_query
                    st.rerun()
                    
                except Exception as e:
                    st.session_state.python_generating = False
                    st.error(f"Error generating code: {str(e)}")
                    return
        
        # Handle stop request
        if st.session_state.python_stop_requested:
            st.session_state.python_generating = False
            st.session_state.python_stop_requested = False
            st.warning("🛑 Code generation stopped by user")
            st.rerun()
        
        # Display generated code if available
        if hasattr(st.session_state, 'generated_code') and st.session_state.generated_code:
            code = st.session_state.generated_code
            metadata = getattr(st.session_state, 'python_metadata', {})
            stored_user_query = getattr(st.session_state, 'current_user_query', user_query)
            
            # Display and edit code
            edited_code, execute_btn = self._display_code_section(code)
            
            # Execute code
            if execute_btn:
                with st.spinner("⚡ Executing code..."):
                    success, output = self.agent.execute_code(edited_code, df)
                    self._display_execution_results(success, output)
                    
                    # Save results if output manager is available
                    if self.output_manager:
                        # Add temperature to metadata if available
                        if hasattr(self.agent, 'temperature') and self.agent.temperature is not None:
                            metadata['temperature'] = self.agent.temperature
                        
                        saved_file = self.output_manager.save_python_result(
                            user_query=stored_user_query,
                            generated_code=code,
                            metadata=metadata,
                            executed_code=edited_code if edited_code != code else None,
                            execution_output=output,
                            execution_success=success
                        )
                        st.success(f"💾 Results saved to: `{saved_file}`")
            
            # Display metadata
            self._display_metadata(metadata) 