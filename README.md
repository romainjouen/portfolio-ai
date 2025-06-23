# 🚀 Portfolio AI - Multi-Purpose AI Development Suite

A comprehensive collection of AI-powered applications and tools for data analysis and conversational AI. This portfolio showcases various AI implementations using modern Python frameworks and multiple AI providers.

## 📋 Project Overview

This portfolio contains two main AI applications:

### 🤖 [AI Chat Interface](./ai_chat/)
A sophisticated Streamlit-based chat interface supporting multiple AI providers and models.

**Key Features:**
- Multi-provider support (OpenAI, Anthropic Claude, Google Gemini, Mistral, DeepSeek)
- Real-time token usage and cost tracking
- Adjustable temperature settings for response creativity
- Conversation management and history
- Modern UI with model selection and configuration

**Models Supported:**
- OpenAI GPT (3.5-turbo, 4, 4-turbo, 4o)
- Anthropic Claude 3 (Sonnet & Haiku)
- Google Gemini Pro
- Mistral Medium & Mixtral 8x7B
- DeepSeek Chat & Reasoner

### 📊 [Data Analysis Platform](./data_analysis/)
An intelligent data analysis suite that combines AI-powered SQL generation with Python data analysis capabilities.

**Key Features:**
- Natural language to SQL query conversion
- AI-powered Python code generation for data analysis
- Multi-provider AI support (OpenAI & Ollama)
- PostgreSQL database integration
- Interactive Streamlit interface with stop controls
- Automatic output management and metadata tracking
- Semantic database schema understanding

**Capabilities:**
- SQL query generation from natural language
- Data visualization and statistical analysis
- Database schema comprehension
- Automated report generation
- Code execution with error handling

## 🏗️ Project Structure

```
portfolio-ai/
├── README.md                    # This overview file
├── .gitignore                   # Git ignore rules
├── github_commands.md           # Git reference commands
│
├── ai_chat/                     # Chat Interface Application
│   ├── app.py                   # Main Streamlit chat app
│   ├── requirements.txt         # Python dependencies
│   ├── environment.yml          # Conda environment
│   ├── README.md                # Detailed chat app documentation
│   ├── src/
│   │   ├── config/              # Configuration management
│   │   └── utils/               # Chat utilities and providers
│   └── venv/                    # Virtual environment
│
└── data_analysis/               # Data Analysis Platform
    ├── app.py                   # Main Streamlit analysis app
    ├── requirements.txt         # Python dependencies
    ├── README.md                # Detailed analysis app documentation
    ├── conf/                    # Configuration files
    │   ├── config.json          # Database and model settings
    │   └── data_analysis/       # Schema definitions
    ├── src/
    │   ├── data_analysis/       # Core analysis agents
    │   ├── ui_components/       # UI components
    │   └── utils/               # Utility modules
    ├── output/                  # Generated results
    └── venv/                    # Virtual environment
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL (for data analysis)
- API keys for desired AI providers
- Ollama (optional, for local models)

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/romainjouen/portfolio-ai.git
cd portfolio-ai
```

2. **Choose your application:**

**For AI Chat Interface:**
```bash
cd ai_chat
# Using conda (recommended)
conda env create -f environment.yml
conda activate ai_chat
# OR using pip
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**For Data Analysis Platform:**
```bash
cd data_analysis
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Configure environment variables:**
   - Copy `.env.example` to `.env`
   - Add your API keys and database credentials
   - Adjust configuration files as needed

4. **Run the applications:**
```bash
# For chat interface
streamlit run ai_chat/app.py

# For data analysis
streamlit run data_analysis/app.py
```

## 🛠️ Technologies Used

- **Frontend:** Streamlit for interactive web interfaces
- **AI Providers:** OpenAI, Anthropic, Google, Mistral, DeepSeek, Ollama
- **Database:** PostgreSQL with intelligent schema understanding
- **Languages:** Python 3.8+
- **Key Libraries:** pandas, matplotlib, seaborn, sqlalchemy, streamlit
- **Environment Management:** Conda, pip, virtual environments

## 🔧 Configuration

Both applications support flexible configuration:

- **AI Model Selection:** Choose between cloud and local models
- **Database Integration:** PostgreSQL connection management
- **API Management:** Secure API key handling
- **Output Control:** Customizable result saving and formatting
- **UI Preferences:** Adjustable interface settings

## 📈 Use Cases

### AI Chat Interface
- Interactive AI conversations and assistance
- Model comparison and testing
- Cost monitoring for AI API usage
- Research and development conversations
- Educational AI interactions

### Data Analysis Platform
- Business intelligence and reporting
- Database exploration and querying
- Automated data analysis workflows
- SQL learning and assistance
- Data visualization and insights

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see individual application directories for specific details.

## 🔗 Links

- [AI Chat Interface Documentation](./ai_chat/README.md)
- [Data Analysis Platform Documentation](./data_analysis/README.md)
- [Git Commands Reference](./github_commands.md)

---

**Built with ❤️ by Romain Jouen**
