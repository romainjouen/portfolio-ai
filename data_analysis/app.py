import json
import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import from our unified structure
from src import PythonAgent, SQLAgent, LLMProviderSelector, PythonAgentUI, SQLAgentUI, ConfigurationUI, load_config, OutputManager

# ===== Streamlit UI =====
def main():
    """Main application function"""
    st.title("🚀 Text to SQL and Python Analyzer")
    
    # Load configs
    try:
        config = load_config()
        with open('conf/data_analysis/semantic_model.json', 'r') as f:
            semantic_model = json.load(f)
        
        # Database connection string
        db_config = config['postgres']
        db_password = os.getenv('DB_PASSWORD', '')
        if not db_password:
            st.error("❌ Database password not found. Please set DB_PASSWORD in your .env file.")
            st.info("📁 Create a .env file in the project root with: DB_PASSWORD=your_password")
            return
        
        db_url = f"postgresql://{db_config['user']}:{db_password}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
        
        # Display configuration info in sidebar
        st.sidebar.header("📊 Configuration")
        st.sidebar.text(f"Database: {db_config['dbname']} @ {db_config['host']}")
        
        # Initialize session state for storing results and agents
        if 'sql_results' not in st.session_state:
            st.session_state.sql_results = None
        if 'last_executed_sql' not in st.session_state:
            st.session_state.last_executed_sql = None
        if 'sql_agent' not in st.session_state:
            st.session_state.sql_agent = None
        if 'python_agent' not in st.session_state:
            st.session_state.python_agent = None
        
        # Initialize configuration UI and output manager
        config_ui = ConfigurationUI()
        output_manager = OutputManager()
        
        # Create tabs for Configuration, SQL and Python
        tab1, tab2, tab3 = st.tabs([
            "⚙️ Configuration",
            "🔍 SQL Generator", 
            "🐍 Python Data Agent"
        ])
        
        with tab1:
            # Configuration Tab
            current_config = config_ui.display(db_url)
        
        with tab2:
            st.header("🔍 SQL Generator")
            
            # Check if global configuration is set
            if not config_ui.has_configuration():
                st.warning("⚠️ Please configure your AI models in the Configuration tab first.")
                return
            
            # Create or get SQL agent using global configuration
            if st.session_state.sql_agent is None:
                st.session_state.sql_agent = config_ui.create_sql_agent(db_url, semantic_model)
                
            if st.session_state.sql_agent:
                # Display current agent info
                agent = st.session_state.sql_agent
                st.success(f"✅ Using {agent.provider.upper()} - {agent.model_name}")
                
                # SQL Agent interface
                sql_ui = SQLAgentUI(st.session_state.sql_agent, st.session_state.sql_agent.provider.upper(), output_manager)
                sql_ui.display()
            else:
                st.error("❌ Failed to create SQL agent. Please check your configuration.")
        
        with tab3:
            st.header("🐍 Python Data Agent")
            
            # Check if global configuration is set
            if not config_ui.has_configuration():
                st.warning("⚠️ Please configure your AI models in the Configuration tab first.")
                return
            
            # Check if we have SQL results
            if st.session_state.sql_results is None or st.session_state.sql_results.empty:
                st.warning("⚠️ No data available. Please run a SQL query first in the SQL Generator tab.")
                return
            
            # Create or get Python agent using global configuration
            if st.session_state.python_agent is None:
                st.session_state.python_agent = config_ui.create_python_agent()
                
            if st.session_state.python_agent:
                # Display current agent info
                agent = st.session_state.python_agent
                st.success(f"✅ Using {agent.provider.upper()} - {agent.model_name}")
                
                # Python Agent interface
                python_ui = PythonAgentUI(st.session_state.python_agent, st.session_state.python_agent.provider.upper(), output_manager)
                python_ui.display(st.session_state.sql_results)
            else:
                st.error("❌ Failed to create Python agent. Please check your configuration.")
                
    except Exception as e:
        st.error(f"❌ Configuration error: {str(e)}")
        st.info("📁 Please ensure conf/config.json and conf/semantic_model.json exist and are properly formatted.")
        
        # Show debug info in case of error
        with st.expander("🔧 Debug Information"):
            st.code(str(e))
            if 'config' in locals():
                st.json(config)

if __name__ == "__main__":
    main()



