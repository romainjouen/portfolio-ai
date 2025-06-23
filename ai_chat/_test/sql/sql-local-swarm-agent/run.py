from dotenv import load_dotenv
from swarm import Swarm
import json
from sql_agents import sql_router_agent
from llm_client import LLMClientManager, SwarmCompatibleClient
from config import LLMProvider, MODELS

# Initialize the LLM client manager and wrap it
llm_manager = LLMClientManager()
swarm_client = SwarmCompatibleClient(llm_manager)

def get_model_config_from_name(model_name: str):
    """Get the full model configuration from a model name."""
    for model_key, config in MODELS.items():
        if config.model_name == model_name:
            return config
    return None

def process_and_print_streaming_response(response):
    content = ""
    last_sender = ""
    current_message = ""

    try:
        for chunk in response:
            # Handle string chunks (raw text)
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
                content += chunk
                continue

            # Handle dictionary chunks
            if isinstance(chunk, dict):
                # Handle content
                if "content" in chunk and chunk["content"] is not None:
                    current_content = chunk["content"]
                    if not content and "sender" in chunk:
                        print(f"\033[94m{chunk['sender']}:\033[0m", end=" ", flush=True)
                    print(current_content, end="", flush=True)
                    current_message += current_content

                # Handle tool calls
                if "tool_calls" in chunk and chunk["tool_calls"] is not None:
                    for tool_call in chunk["tool_calls"]:
                        f = tool_call["function"]
                        name = f["name"]
                        if not name:
                            continue
                        sender = chunk.get("sender", "Assistant")
                        print(f"\033[94m{sender}: \033[95m{name}\033[0m()")

                # Handle finish
                if chunk.get("finish_reason"):
                    content = current_message or content
                    current_message = ""
                    print()  # New line at end

                # Handle Swarm response
                if "response" in chunk:
                    return chunk["response"]

    except Exception as e:
        print(f"\nError processing stream: {e}")
        if isinstance(response, (str, dict)):
            if isinstance(response, dict):
                if "content" in response:
                    return response["content"]
                elif hasattr(response, 'text'):
                    return response.text
            return str(response)
        return None

    final_content = content if content else current_message
    if final_content:
        return final_content
    return None

def pretty_print_messages(messages) -> None:
    for message in messages:
        if message["role"] != "assistant":
            continue

        # print agent name in blue
        print(f"\033[94m{message.get('sender', 'Assistant')}\033[0m:", end=" ")

        # print response, if any
        if message.get("content"):
            print(message["content"])

        # print tool calls in purple, if any
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) > 1:
            print()
        for tool_call in tool_calls:
            f = tool_call["function"]
            name, args = f["name"], f.get("arguments", "{}")
            try:
                arg_str = json.dumps(json.loads(args)).replace(":", "=")
                print(f"\033[95m{name}\033[0m({arg_str[1:-1]})")
            except json.JSONDecodeError:
                print(f"\033[95m{name}\033[0m()")

def run_demo_loop(
    starting_agent, context_variables=None, stream=False, debug=False
) -> None:
    client = Swarm(client=swarm_client)  # Use the wrapped client
    print("Starting Multi-LLM Swarm CLI 🐝")
    
    # Get the full model config for the starting agent
    model_config = get_model_config_from_name(starting_agent.model)
    if model_config:
        print(f"Router Agent using: {model_config.model_name} ({model_config.provider.value})")
    else:
        print(f"Router Agent using: {starting_agent.model}")

    messages = []
    agent = starting_agent

    while True:
        try:
            user_input = input("\033[90mUser\033[0m: ")
            if not user_input.strip():
                continue
                
            messages.append({"role": "user", "content": user_input})

            response = client.run(
                agent=agent,
                messages=messages,
                context_variables=context_variables or {},
                stream=stream,
                debug=debug,
            )

            if stream:
                response = process_and_print_streaming_response(response)
            else:
                pretty_print_messages(response.messages)

            if response:
                if hasattr(response, 'messages'):
                    messages.extend(response.messages)
                elif isinstance(response, str):
                    messages.append({"role": "assistant", "content": response})
                agent = response.agent if hasattr(response, 'agent') else agent
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            if debug:
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    load_dotenv()
    run_demo_loop(sql_router_agent, stream=True, debug=True)