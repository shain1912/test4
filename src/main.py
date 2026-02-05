"""
CLI version of the AI Interview Platform.
UI strings are loaded from configuration files.
"""

import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bot import ConfigurableInterviewGraph, InterviewInfo
from src.db import init_db, insert_interview
from src.config_loader import get_config_loader


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    # Load configuration
    loader = get_config_loader()
    ui = loader.load_ui_strings()

    if not api_key:
        print(ui['cli']['api_key_error'])
        return

    # Initialize DB
    init_db()

    # Initialize Bot Graph
    bot_graph = ConfigurableInterviewGraph(api_key)

    # Get topic name for display
    topic_name = bot_graph.get_topic_name()

    # Initial State
    state = {
        "messages": [],
        "info": InterviewInfo(),
        "topics_covered": [],
        "turn_index": 0
    }

    print(ui['cli']['header'].format(topic_name=topic_name))

    # Initial Greeting from config
    greeting = bot_graph.get_greeting()
    print(f"{ui['cli']['bot_name']}: {greeting}")

    while True:
        try:
            user_input = input(f"\n{ui['cli']['user_prompt']}")
            if user_input.lower() in ['종료', 'q', 'exit', 'quit']:
                # Save Data on Exit
                final_info = state["info"].model_dump()
                insert_interview(final_info)
                print(f"\n{ui['cli']['data_saved'].format(data=final_info)}")
                break

            if not user_input.strip():
                continue

            # Update State with User Input
            state["messages"].append(HumanMessage(content=user_input))

            # Invoke Graph
            # The graph returns the updated state dictionary
            result_state = bot_graph.graph.invoke(state)

            # Update our local state tracker
            state = result_state

            # Get Bot Response (Last message should be AIMessage)
            last_message = state["messages"][-1]
            print(f"\n{ui['cli']['bot_name']}: {last_message.content}")

            # Show suggested replies if available
            if state.get("suggested_replies"):
                print(f"\n[{', '.join(state['suggested_replies'])}]")

        except KeyboardInterrupt:
            print(f"\n{ui['cli']['interrupted']}")
            break
        except Exception as e:
            print(f"\n{ui['cli']['error'].format(error=e)}")
            break


if __name__ == "__main__":
    main()
