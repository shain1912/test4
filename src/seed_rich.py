import sqlite3
import os
import random
from src.db import init_db, insert_interview

# Reset DB
DB_PATH = os.path.join(os.path.dirname(__file__), 'interviews.db')
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
init_db()

# Define weighted templates to create realistic "Hotspots" and "Trends"
templates = [
    # Top Issue: Night Safety in Seomyeon (High Frequency)
    {
        "weight": 10,
        "base": {
            "location": "서면 2번가", 
            "urban_element": "가로등", 
            "solution_type": "Infrastructure", 
            "primary_value": "Safety"
        },
        "issues": [
            "골목이 너무 어두워서 무서워요.",
            "취객이 많은데 조명이 부족해서 불안합니다.",
            "가로등이 고장난 채로 방치되어 있어요.",
            "밤길이 너무 깜깜해서 범죄 위험이 느껴져요.",
            "CCTV랑 밝은 조명이 더 필요합니다."
        ]
    },
    # Top Issue: Broken Sidewalks in Tourist Areas (Medium Frequency)
    {
        "weight": 8,
        "base": {
            "location": "해운대 구남로", 
            "urban_element": "보도블록", 
            "solution_type": "Maintenance", 
            "primary_value": "Convenience"
        },
        "issues": [
            "보도블록이 깨져서 발이 걸려 넘어질 뻔했어요.",
            "캐리어 끌고 가는데 길이 울퉁불퉁해서 너무 불편해요.",
            "비 오면 보도블록 사이에 물이 고여서 튀어요.",
            "관광객이 많은데 길바닥 정비가 안 되어 있어서 부끄럽네요."
        ]
    },
    # Specific Issue: Accessibility at Stations (Medium Frequency)
    {
        "weight": 6,
        "base": {
            "location": "부산역 광장", 
            "urban_element": "계단/단차", 
            "solution_type": "Facility", 
            "primary_value": "Accessibility"
        },
        "issues": [
            "휠체어가 이동할 수 있는 경사로가 너무 멀리 있어요.",
            "엘리베이터 줄이 너무 길어서 계단을 이용해야 하는데 힘들어요.",
            "캐리어 짐을 들고 계단을 내려가기가 위험합니다.",
            "노약자를 위한 이동 지원 시설이 부족합니다."
        ]
    },
    # Minor Issue: Trash in Cafe Streets (Low Frequency)
    {
        "weight": 4,
        "base": {
            "location": "전포 카페거리", 
            "urban_element": "쓰레기통", 
            "solution_type": "Service", 
            "primary_value": "Aesthetics"
        },
        "issues": [
            "먹다 남은 음료 컵이 화단에 버려져 있어요.",
            "쓰레기통이 없어서 사람들이 아무데나 버리고 가요.",
            "거리가 지저분해서 미관상 좋지 않습니다."
        ]
    },
     # Random Noise (Single occurrences)
    {
        "weight": 2,
        "base": {"location": "광안리 해변", "urban_element": "킥보드", "solution_type": "Policy", "primary_value": "Safety"},
        "issues": ["전동 킥보드가 너무 빨리 달려서 위험해요."]
    },
    {
        "weight": 2,
        "base": {"location": "동래역", "urban_element": "횡단보도", "solution_type": "Infrastructure", "primary_value": "Convenience"},
        "issues": ["신호가 너무 짧아서 건너기 힘들어요."]
    }
]

# Generate 50-60 entries
total_entries = 0
for tmpl in templates:
    count = tmpl['weight'] + random.randint(-1, 2) # Add some randomness
    for _ in range(max(1, count)):
        issue_text = random.choice(tmpl['issues'])
        
        # Add slight variation to text to make embeddings distinct but clustered
        variations = [" 특히", " 정말", " 진짜", " 너무", " 매번"]
        if random.random() > 0.5:
            issue_text = issue_text.replace(".", random.choice(variations) + " 불편해요.")

        data = tmpl['base'].copy()
        data['issue'] = issue_text
        
        # Randomize 'solution_detail' slightly
        sol_prefixes = ["조속한 ", "확실한 ", "시급한 ", "노후화 개선 및 ", ""]
        data['solution_detail'] = random.choice(sol_prefixes) + "해결책 마련 필요"

        insert_interview(data)
        total_entries += 1

print(f"Inserted {total_entries} realistic entries with skewed distribution.")
