"""
interviews.db 샘플 데이터 생성 스크립트
- 다양한 카테고리, 위치, 심각도의 랜덤 인터뷰 데이터 삽입
"""
import sqlite3
import os
import random
import json
import uuid
from datetime import datetime, timedelta

DB_PATH = '/home/shain/dev/test4/interviews.db'


def generate_session_id():
    return str(uuid.uuid4())[:8]


def random_timestamp():
    """최근 90일 내 랜덤 타임스탬프"""
    now = datetime.now()
    delta = timedelta(
        days=random.randint(0, 90),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )
    return (now - delta).strftime("%Y-%m-%d %H:%M:%S")


# ── 템플릿 데이터 ─────────────────────────────────────────────
templates = [
    # 1. 서면 야간 안전
    {
        "weight": 25,
        "base": {
            "location_bucket": "서면 2번가 인근",
            "primary_category": "안전 (Safety)",
            "severity_score": 4,
        },
        "issues": [
            "골목이 너무 어두워서 밤에 다니기 무섭습니다.",
            "가로등이 고장난 채로 방치되어 있어요.",
            "취객이 자주 출몰해서 밤길이 위험합니다.",
            "CCTV가 없어서 범죄 사각지대 같아요.",
            "조명이 너무 어둡습니다. 빨리 개선이 필요해요.",
            "밤에 혼자 다니기 두려운 골목입니다.",
            "가로등 수가 너무 적어서 어두워요.",
        ],
        "evidence_snippets": [
            "밤에 혼자 귀가하다가 무서웠어요.",
            "가로등이 꺼져 있었습니다.",
            "취객에게 시비가 붙을 뻔했어요.",
        ],
    },
    # 2. 해운대 접근성
    {
        "weight": 30,
        "base": {
            "location_bucket": "해운대 (Haeundae)",
            "primary_category": "접근성 (Accessibility)",
            "severity_score": 3,
        },
        "issues": [
            "휠체어 경사로가 전혀 없어서 이동이 불가능합니다.",
            "보도블럭 턱이 너무 높아 휠체어로 올라갈 수 없어요.",
            "계단만 있고 엘리베이터가 없어요.",
            "유모차를 끌고 다니기 매우 힘든 구조입니다.",
            "장애인 편의시설이 전혀 갖춰져 있지 않아요.",
            "경사로 경사가 너무 급해서 위험합니다.",
            "엘리베이터가 자주 고장나서 이용이 어렵습니다.",
        ],
        "evidence_snippets": [
            "휠체어를 밀다가 턱에 걸렸어요.",
            "엘리베이터가 또 고장났습니다.",
            "유모차 때문에 계단을 혼자 들고 올라갔어요.",
        ],
    },
    # 3. 광안리 쾌적성
    {
        "weight": 30,
        "base": {
            "location_bucket": "광안리 (Gwangalli)",
            "primary_category": "쾌적성 (Comfort)",
            "severity_score": 2,
        },
        "issues": [
            "해변에 쓰레기가 나뒹굴고 있습니다.",
            "폭죽 쓰레기가 여기저기 방치되어 있어요.",
            "쓰레기통이 이미 넘쳐서 냄새가 납니다.",
            "일회용 컵과 플라스틱이 모래사장에 가득해요.",
            "주말 오후에는 쓰레기로 걷기도 힘들 정도입니다.",
            "화장실 청결 상태가 매우 불량합니다.",
            "거리 전반적으로 악취가 심합니다.",
        ],
        "evidence_snippets": [
            "발에 유리 조각이 밟혔어요.",
            "쓰레기통 근처를 지나가기 힘들었습니다.",
            "바닥에 음식물 쓰레기가 많았어요.",
        ],
    },
    # 4. 센텀시티 길찾기
    {
        "weight": 25,
        "base": {
            "location_bucket": "센텀시티 (Centum City)",
            "primary_category": "길찾기 (Wayfinding)",
            "severity_score": 1,
        },
        "issues": [
            "지하철 출구가 복잡해서 길을 잃었습니다.",
            "안내 표지판이 부족해서 방향을 잡기 어렵습니다.",
            "지도와 실제 위치가 달라서 혼란스러웠습니다.",
            "쇼핑몰 내부가 미로 같아서 출구를 못 찾겠어요.",
            "층 안내가 제대로 되어 있지 않아요.",
            "영어 표지판이 없어서 외국인 관광객이 헤맸어요.",
            "전광판 표시가 오류가 나있어서 잘못 이동했습니다.",
        ],
        "evidence_snippets": [
            "출구를 세 바퀴나 돌았습니다.",
            "직원에게 다섯 번이나 길을 물었어요.",
            "지도 앱도 여기선 엉뚱한 곳을 안내했어요.",
        ],
    },
    # 5. 남포동 혼잡
    {
        "weight": 30,
        "base": {
            "location_bucket": "남포동 (Nampo-dong)",
            "primary_category": "기타 (Other)",
            "severity_score": 2,
        },
        "issues": [
            "사람이 너무 많아서 밀려다니는 수준이에요.",
            "노점상이 인도를 거의 다 차지하고 있습니다.",
            "길이 너무 좁아서 유모차를 끌 수가 없어요.",
            "주말에는 인파 때문에 이동이 사실상 불가능합니다.",
            "관광버스와 차량이 혼잡해서 보행이 위험합니다.",
            "불법 주차 차량이 인도를 막고 있어요.",
            "휴일 오후에는 소음도 너무 심합니다.",
        ],
        "evidence_snippets": [
            "노점 때문에 인도 절반이 막혔어요.",
            "유모차가 빠져나가지 못했어요.",
            "버스가 인도 위로 올라온 적도 있었습니다.",
        ],
    },
    # 6. 동래 교통 문제
    {
        "weight": 20,
        "base": {
            "location_bucket": "동래 (Dongrae)",
            "primary_category": "교통 (Traffic)",
            "severity_score": 3,
        },
        "issues": [
            "버스 배차 간격이 너무 길어서 오래 기다립니다.",
            "버스 정류장에 의자가 없어서 노인분들이 힘들어해요.",
            "정류장 안내판이 고장나서 버스 도착 정보를 알 수 없어요.",
            "교차로 신호가 짧아서 노약자가 건너기 힘듭니다.",
            "지하철 환승 연결이 너무 불편합니다.",
            "자전거 도로가 없어서 차도에서 타야 해요.",
            "주차 공간이 너무 부족해서 불법 주차가 만연합니다.",
        ],
        "evidence_snippets": [
            "버스를 40분이나 기다렸어요.",
            "신호등 건너다가 차에 경적을 들었어요.",
            "지하철 환승 거리가 너무 멀었어요.",
        ],
    },
    # 7. 기장 환경 문제
    {
        "weight": 15,
        "base": {
            "location_bucket": "기장 (Gijang)",
            "primary_category": "환경 (Environment)",
            "severity_score": 2,
        },
        "issues": [
            "해안가 플라스틱 쓰레기 문제가 심각합니다.",
            "수산시장 주변 악취가 날씨에 따라 심해집니다.",
            "미세먼지 알림이 없어서 야외 활동 계획이 어렵습니다.",
            "녹조 현상으로 강물 색이 이상합니다.",
            "공장 매연이 주거 지역으로 유입됩니다.",
            "소음 공해가 심해서 창문을 열기 힘들어요.",
            "하수구 냄새가 비가 오면 더 심해집니다.",
        ],
        "evidence_snippets": [
            "아이들이 뛰어노는 바닷가에 쓰레기가 넘쳤어요.",
            "매연 냄새가 집 안에까지 들어와요.",
            "강물이 초록빛으로 변해 있었습니다.",
        ],
    },
    # 8. 부산역 시설 불편
    {
        "weight": 20,
        "base": {
            "location_bucket": "부산역 (Busan Station)",
            "primary_category": "시설 (Facility)",
            "severity_score": 3,
        },
        "issues": [
            "대기 의자가 충분하지 않아서 서서 기다려야 합니다.",
            "화장실 수가 적고 청결 상태도 불량합니다.",
            "냉난방이 제대로 안 되어서 불쾌합니다.",
            "짐 보관함이 부족해서 이용하기 어렵습니다.",
            "안내 방송이 잘 들리지 않아서 정보를 놓쳤어요.",
            "편의점 가격이 주변보다 훨씬 비쌉니다.",
            "우산 보관함이 없어서 우산을 들고 다녀야 해요.",
        ],
        "evidence_snippets": [
            "30분 서서 기다렸더니 다리가 아팠어요.",
            "화장실이 너무 불결해서 이용 안 했어요.",
            "에어컨이 안 켜진 대합실에서 쓰러질 뻔했어요.",
        ],
    },
]


def seed_data(num_entries=200):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 테이블이 없으면 생성
    c.execute('''
        CREATE TABLE IF NOT EXISTS interviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TEXT,
            issue_text TEXT,
            severity_score INTEGER,
            primary_category TEXT,
            location_bucket TEXT,
            evidence_span TEXT,
            raw_log TEXT
        )
    ''')

    # 기존 데이터 삭제 후 재삽입 (선택적)
    c.execute('DELETE FROM interviews')
    print("기존 데이터를 초기화했습니다.")

    total = 0
    # 가중치 기반 템플릿 리스트
    weighted_templates = []
    for t in templates:
        weighted_templates.extend([t] * t["weight"])

    # 세션별로 1~4개 이슈 묶기
    while total < num_entries:
        tmpl = random.choice(weighted_templates)
        session_id = generate_session_id()
        num_issues = random.randint(1, 3)

        for _ in range(num_issues):
            if total >= num_entries:
                break

            issue_text = random.choice(tmpl["issues"])
            evidence_span = random.choice(tmpl["evidence_snippets"])

            # 약간의 텍스트 변형
            filler = random.choice(["", " 정말", " 진짜", " 너무", " 매번", " 항상"])
            if filler and random.random() > 0.4:
                issue_text = issue_text.rstrip(".") + filler + " 불편했어요."

            severity = tmpl["base"]["severity_score"]
            # 소폭 변동 (1~5 범위 내)
            severity = max(1, min(5, severity + random.choice([-1, 0, 0, 1])))

            data = {
                "issue_text": issue_text,
                "severity_score": severity,
                "primary_category": tmpl["base"]["primary_category"],
                "location_bucket": tmpl["base"]["location_bucket"],
                "evidence_span": evidence_span,
            }
            raw_log = json.dumps(data, ensure_ascii=False)
            timestamp = random_timestamp()

            c.execute('''
                INSERT INTO interviews (
                    session_id, timestamp, issue_text, severity_score,
                    primary_category, location_bucket, evidence_span, raw_log
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id, timestamp, issue_text, severity,
                tmpl["base"]["primary_category"],
                tmpl["base"]["location_bucket"],
                evidence_span,
                raw_log,
            ))
            total += 1

    conn.commit()

    # 결과 요약 출력
    c.execute('SELECT COUNT(*) FROM interviews')
    count = c.fetchone()[0]
    c.execute('SELECT primary_category, COUNT(*) FROM interviews GROUP BY primary_category ORDER BY COUNT(*) DESC')
    cat_summary = c.fetchall()
    c.execute('SELECT location_bucket, COUNT(*) FROM interviews GROUP BY location_bucket ORDER BY COUNT(*) DESC')
    loc_summary = c.fetchall()

    conn.close()

    print(f"\n✅ 총 {count}건의 샘플 데이터 삽입 완료!\n")
    print("📊 카테고리별 분포:")
    for cat, cnt in cat_summary:
        print(f"  {cat}: {cnt}건")
    print("\n📍 위치별 분포:")
    for loc, cnt in loc_summary:
        print(f"  {loc}: {cnt}건")


if __name__ == '__main__':
    seed_data(num_entries=220)
