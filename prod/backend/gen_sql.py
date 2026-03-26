import json
import random
import uuid

templates = [
    {
        "weight": 20,
        "base": {
            "location_bucket": "서면 2번가 인근", 
            "primary_category": "안전 (Safety)",
            "severity_score": 4
        },
        "issues": [
            "골목이 너무 어두워서 무서워요.",
            "가로등이 없어서 밤길이 위험합니다.",
            "취객이 많아서 밤에 다니기 두려워요.",
            "CCTV가 부족해서 범죄 사각지대 같아요.",
            "조명이 너무 어둡습니다. 밝게 해주세요."
        ]
    },
    {
        "weight": 50,
        "base": {
            "location_bucket": "해운대 (Haeundae)", 
            "primary_category": "접근성 (Accessibility)",
            "severity_score": 3
        },
        "issues": [
            "휠체어가 다닐 수 있는 경사로가 없습니다.",
            "보도블럭 턱이 너무 높아서 휠체어 이동이 힘들어요.",
            "계단밖에 없어서 유모차나 휠체어는 못 지나갑니다.",
            "엘리베이터가 고장나서 이동할 수가 없어요.",
            "장애인 편의시설이 전무합니다."
        ]
    },
    {
        "weight": 50,
        "base": {
            "location_bucket": "광안리 (Gwangalli)", 
            "primary_category": "쾌적성 (Comfort)",
            "severity_score": 2
        },
        "issues": [
            "해변가에 쓰레기가 너무 많아요.",
            "폭죽 쓰레기가 여기저기 널려있어서 더러워요.",
            "쓰레기통이 넘쳐서 냄새가 심합니다.",
            "일회용 컵들이 길거리에 방치되어 있어요.",
            "거리가 너무 지저분하고 악취가 납니다."
        ]
    },
    {
        "weight": 40,
        "base": {
            "location_bucket": "센텀시티 (Centum City)", 
            "primary_category": "길찾기 (Wayfinding)",
            "severity_score": 1
        },
        "issues": [
            "지하철 출구가 너무 복잡해서 길을 잃었어요.",
            "안내 표지판이 없어서 어디로 가야할지 모르겠어요.",
            "지도가 잘못되어 있어서 엉뚱한 곳으로 갔습니다.",
            "쇼핑몰 내부가 미로 같아서 찾기가 힘들어요.",
            "길 안내가 불친절합니다."
        ]
    },
    {
        "weight": 40,
        "base": {
            "location_bucket": "남포동 (Nampo-dong)", 
            "primary_category": "기타 (Other)",
            "severity_score": 2
        },
        "issues": [
            "사람이 너무 많아서 밀려 다녔어요.",
            "길이 너무 좁아서 다니기가 힘듭니다.",
            "노점상이 인도를 차지해서 걸을 공간이 없어요.",
            "관광객이 너무 많아서 보행이 불가능해요.",
            "매우 혼잡하고 정신이 없습니다."
        ]
    }
]

sql_inserts = []
sql_inserts.append("INSERT INTO interviews (session_id, issue_text, severity_score, primary_category, location_bucket, evidence_span, raw_log) VALUES")

values = []
for tmpl in templates:
    count = tmpl['weight'] + random.randint(-5, 5) 
    for _ in range(max(1, count)):
        issue_text = random.choice(tmpl['issues'])
        variations = [" 정말", " 진짜", " 너무", " 매번", " 항상"]
        if random.random() > 0.3:
            issue_text = issue_text.replace(".", random.choice(variations) + " 불편해요.")

        b = tmpl['base']
        session_id = str(uuid.uuid4())[:8]
        evidence = issue_text[:15] + "..."
        raw_log = json.dumps({"generated_by": "seed script"})
        
        # Escape single quotes and proper format
        val = f"('{session_id}', '{issue_text}', {b['severity_score']}, '{b['primary_category']}', '{b['location_bucket']}', '{evidence}', '{raw_log}'::jsonb)"
        values.append(val)

sql_inserts.append(",\n".join(values) + ";")

with open("/home/shain/dev/test4/prod/backend/seed.sql", "w") as f:
    f.write("\n".join(sql_inserts))

print("seed.sql generated.")
