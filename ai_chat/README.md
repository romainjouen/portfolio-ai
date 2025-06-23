# AI Chat Interface

A Streamlit-based chat interface that supports multiple AI models including OpenAI GPT, Anthropic Claude, Google Gemini, Mistral, and more.

## Project Structure
```
ai_chat/
├── app.py
├── src/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── chat_interface.py
│   │   └── llm_provider.py
│   └── config/
│       ├── __init__.py
│       └── config.py
├── .env
├── .env.example
├── requirements.txt
├── environment.yml
└── README.md
```

## Features

- Support for multiple AI providers:
  - OpenAI
  - Anthropic Claude 3 (Sonnet & Haiku)
  - Google Gemini Pro
  - Mistral Medium
  - Mixtral 8x7B
  - DeepSeek Chat
  - DeepSeek Reasoner
- Adjustable temperature settings
- Real-time token usage tracking
- Cost monitoring
- Conversation management

# Setup

Choose one of these setup methods:

### Method 1: Using Conda (Recommended)
```bash
# Create and activate environment
conda env create -f environment.yml
conda activate ai_chat

# Update environment if needed
conda env update -f environment.yml
```

### Method 2: Using pip
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
# or 
source env/bin/activate

# Install dependencies and local package
pip install -r requirements.txt
pip install -e .
```

### Configuration

# Add your API keys to `.env`:
   - OpenAI API key
   - Anthropic API key
   - Google API key
   - Mistral API key
   - HuggingFace API key
   - DeepSeek API key

2. Ensure your project structure matches the one shown above


## Usage

1. Ensure your environment is activated:
```bash
conda activate ai_chat
# or
source venv/bin/activate
```

2. Run the application:
```bash
streamlit run app.py
```

## Features in Detail

### Model Selection
Choose from various AI models in the sidebar. Each model has its own strengths:
- Claude 3 Sonnet/Haiku: Advanced reasoning and analysis
- GPT models: General purpose and coding
- Gemini Pro: Competitive performance at lower cost
- Mistral/Mixtral: Open source alternatives
- DeepSeek: Specialized reasoning capabilities

### Temperature Control
Adjust the temperature setting (0.0 - 1.0) to control response randomness:
- Lower values (0.0 - 0.3): More focused, deterministic responses
- Medium values (0.4 - 0.7): Balanced creativity
- Higher values (0.8 - 1.0): More creative, varied responses

### Usage Tracking
Monitor your usage in real-time:
- Message count
- Token usage
- Cost estimation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.