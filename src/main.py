import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bot import BusanDesignGraph, InterviewInfo
from src.db import init_db, insert_interview

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file.")
        return

    # Initialize DB
    init_db()

    # Initialize Bot Graph
    bot_graph = BusanDesignGraph(api_key)
    
    # Initial State
    state = {
        "messages": [],
        "info": InterviewInfo(),
        "topics_covered": []
    }

    print("=========================================")
    print("부산 공공디자인 인터뷰 챗봇 (LangGraph Version)")
    print("종료하려면 '종료', 'q', 'exit'를 입력하세요.")
    print("=========================================")
    
    # Initial Greeting
    print("부산디자인봇: 안녕하세요! 부산의 걷기 좋은 도시 만들기에 참여해주셔서 감사합니다. 지금 계신 곳은 어디인가요?")
    
    # We might want to inject this greeting into the state messages?
    # For now, let's just wait for user input.

    while True:
        try:
            user_input = input("\n나: ")
            if user_input.lower() in ['종료', 'q', 'exit', 'quit']:
                # Save Data on Exit
                final_info = state["info"].dict()
                insert_interview(final_info)
                print(f"\n인터뷰 데이터가 저장되었습니다: {final_info}")
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
            print(f"\n부산디자인봇: {last_message.content}")
            
            # Optional: Debug info dump (hidden from normal user)
            # print(f"[DEBUG] Topics: {state['topics_covered']}")
            # print(f"[DEBUG] Info: {state['info']}")

        except KeyboardInterrupt:
            print("\n중단됨.")
            break
        except Exception as e:
            print(f"\n오류 발생: {e}")
            break

if __name__ == "__main__":
    main()
