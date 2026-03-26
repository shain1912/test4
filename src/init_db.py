# init_db_script.py
from db import init_db

if __name__ == "__main__":
    try:
        init_db()
        print("✅ interviews.db 데이터베이스가 성공적으로 초기화되었습니다.")
    except Exception as e:
        print(f"❌ DB 초기화 중 오류 발생: {e}")
