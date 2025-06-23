Goal :
- Build a streamlit interface to let users with a LLM model chosen by the user to ask questions and get answers.

Interface : 
- A sidebar to let the user chooses for :
    - model :
        - OpenAI models (gpt-3.5, gpt-4o)
        - deepseek models (deepseek-chat, deepseek-coder)
        - Google models (gemini-1.5-flash-latest, gemini-1.5-pro-latest, gemini-2.0-flash)
        - Anthropic models (claude-3-5-sonnet-20240620, claude-3-5-sonnet-20240229)
    - temperature


System approach :
- The system is a chat interface based on a LLM model and must be able to answer to the user's question from basic to advanced.


Frameworks :
- Python
- Pydantic

Key principles :
- Use Python
- Use docstrings and comments to explain the code and the functions
- Throughout all the process always use the selected model and keep the token count and costs related to the model used

Features :
- Tokens and related costs counter all along the project and the user can see the number of tokens used 
- The user can choose the language he wants to use

Project structure :
AI_CHAT/
├── src/
│   ├── config/
│   │   ├── config.py
│   │   └── config.yaml
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py
│       ├── llm_provider_agent.py
│       ├── llm_provider.py
│       ├── prompts.py
│       ├── search_api.py
│       ├── token_counter.py
│       └── token_tracker.py
├── .env
├── .gitignore
└── instructions.md


==================
Addings :
- put all the required elements in a config.yaml file located in /config folder
- create the related READ.ME, environment.yml, .env.example and requirements.txt files
- put all the prompts of the project in config.yaml