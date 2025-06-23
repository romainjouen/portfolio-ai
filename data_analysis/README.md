# 🚀 Text to SQL and Python Analyzer

A comprehensive data analysis platform that combines AI-powered SQL generation and Python data analysis. The application connects to your PostgreSQL database, understands your schema, and provides intelligent query generation and data analysis capabilities using multiple AI providers.

## ✨ Features

- **🔍 SQL Generator**: Convert natural language to SQL queries with semantic understanding
- **🐍 Python Data Agent**: Generate and execute Python code for data analysis and visualization
- **⚙️ Multi-Provider Support**: Choose between OpenAI (GPT models) and Ollama (local models)
- **🛑 Stop Controls**: Interrupt AI generation processes with stop buttons
- **📊 Interactive UI**: Modern Streamlit interface with tabbed navigation
- **💾 Output Management**: Automatic saving of results and analysis
- **🔧 Configuration Management**: Flexible model and database configuration
- **📈 Metadata Tracking**: Detailed performance and usage metrics

## 🔧 Pre-requisites

### For Ollama (Local Models)
Install Ollama on your local machine from the [official website](https://ollama.com/) and pull the necessary models:

```bash
ollama pull qwen2.5-coder:32b
# Or other models you prefer
```

### For OpenAI (Cloud Models)
Get an OpenAI API key from [OpenAI Platform](https://platform.openai.com/) and set it as an environment variable.

## 🐍 Setup Python Environment

Create and activate a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS and Linux
source venv/bin/activate
# On Windows
# venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

### Environment Variables
Create a `.env` file in the project root:

```bash
# Database password (required)
DB_PASSWORD=your_database_password

# OpenAI API key (optional, only if using OpenAI models)
OPENAI_API_KEY=your_openai_api_key
```

### Configuration Files
The application uses two main configuration files in the `conf` directory:

#### 1. `conf/config.json` - Main Configuration
```json
{
    "postgres": {
        "host": "localhost",
        "dbname": "your_database",
        "user": "your_username",
        "port": "5432"
    },
    "models": {
        "openai": {
            "default": "gpt-4o-mini",
            "alternatives": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        },
        "ollama": {
            "default": "qwen2.5-coder:32b",
            "alternatives": ["llama3.1", "mistral", "codellama"]
        }
    },
    "temperature": 0.0
}
```

#### 2. `conf/data_analysis/semantic_model.json` - Database Schema Info
```json
{
    "tables": {
        "your_table": {
            "description": "Description of what this table contains",
            "columns": {
                "column_name": "Description of this column"
            }
        }
    },
    "relationships": [
        {
            "from_table": "table1",
            "to_table": "table2",
            "relationship": "foreign_key",
            "description": "How these tables are related"
        }
    ]
}
```

## 🚀 Usage

1. **Start the application**:
```bash
streamlit run app.py
```

2. **Configure your AI models** in the Configuration tab:
   - Select your preferred AI provider (OpenAI or Ollama)
   - Choose a specific model
   - Adjust temperature settings for creativity vs consistency

3. **Generate SQL queries** in the SQL Generator tab:
   - Describe what data you want in natural language
   - Review and edit the generated SQL
   - Execute queries and view results
   - Use the stop button to interrupt long generations

4. **Analyze data** in the Python Data Agent tab:
   - Work with data from your SQL queries
   - Request analysis, visualizations, or insights
   - Review and execute generated Python code
   - Stop generation if needed

## 📁 Project Structure

```
data_analysis/
├── app.py                          # Main Streamlit application
├── .env                           # Environment variables (create this)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── conf/                          # Configuration files
│   ├── config.json               # Database & model configuration
│   └── data_analysis/
│       └── semantic_model.json   # Database schema information
├── src/                          # Source code package
│   ├── __init__.py              # Package initialization
│   ├── data_analysis/           # Core analysis agents
│   │   ├── __init__.py         
│   │   ├── python_agent.py     # Python code generation agent
│   │   ├── python_agent_utils.py
│   │   ├── sql_agent.py        # SQL generation agent
│   │   └── sql_agent_utils.py  
│   ├── ui_components/           # User interface components
│   │   └── ui.py               # Streamlit UI classes
│   └── utils/                   # Utility modules
│       ├── __init__.py         
│       └── output_manager.py   # Result saving and management
├── output/                      # Generated outputs and results
└── venv/                       # Virtual environment (created by you)
```

## 🎯 Key Features Explained

### Multi-Provider AI Support
- **OpenAI**: Cloud-based models like GPT-4o, GPT-4-turbo for high-quality results
- **Ollama**: Local models for privacy and cost control

### Stop Button Functionality
- Interrupt SQL generation or Python code generation at any time
- Prevents long-running or unwanted AI processes
- Maintains UI responsiveness during generation

### Intelligent SQL Generation
- Uses semantic model understanding of your database
- Considers table relationships and column meanings
- Generates optimized queries based on natural language input

### Advanced Python Analysis
- Generates context-aware Python code for data analysis
- Supports visualization, statistical analysis, and data manipulation
- Executes code safely with error handling

### Output Management
- Automatically saves all queries, generated code, and results
- Tracks metadata like execution time, token usage, and model performance
- Organizes outputs for easy retrieval and analysis

## 🔍 Troubleshooting

### Database Connection Issues
- Ensure PostgreSQL is running and accessible
- Verify database credentials in `config.json` and `.env`
- Check firewall and network connectivity

### AI Model Issues
- For Ollama: Ensure Ollama is running (`ollama serve`)
- For OpenAI: Verify your API key is set correctly
- Check model availability and spelling in configuration

### Generation Issues
- Use stop buttons if generation takes too long
- Try different temperature settings for varied results
- Ensure semantic model accurately describes your database


## 🎥 Demo

You can watch a demo video of an earlier version on [YouTube](https://youtu.be/kem-v9MXuG4).

## 🤝 Contributing

Feel free to contribute by:
- Adding support for more AI providers
- Improving the semantic model understanding
- Enhancing the UI/UX
- Adding new analysis capabilities

## 📄 License

This project is open source. Please check the license file for details.