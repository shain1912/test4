#!/bin/bash
# =============================================================
#  📦 USB 오프라인 배포 준비 스크립트 (본인 PC에서 1회 실행)
# =============================================================
#  실행 후 생성된 폴더째로 USB에 복사하면 됩니다.
# =============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "======================================================"
echo "  📦 USB 오프라인 배포 패키지 준비 중..."
echo "======================================================"
echo ""

# ── 가상환경 활성화 ──────────────────────────────────────────
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ 가상환경 활성화됨"
else
    echo "⚠️  venv 없음. 먼저 start.sh 를 실행해 venv를 만드세요."
    exit 1
fi

# ── 오프라인 패키지 다운로드 ─────────────────────────────────
echo "📥 패키지를 offline_packages/ 에 다운로드합니다..."
mkdir -p offline_packages
pip download -r requirements.txt -d offline_packages/ -q
echo "✅ 오프라인 패키지 다운로드 완료"

# ── .env 예시 파일 생성 ──────────────────────────────────────
if [ ! -f ".env.example" ]; then
    echo "OPENAI_API_KEY=sk-여기에_본인_API_키를_입력하세요" > .env.example
    echo "✅ .env.example 생성됨"
fi

# ── USB 복사 대상 출력 ───────────────────────────────────────
echo ""
echo "======================================================"
echo "  ✅ 준비 완료! 아래 항목을 USB에 복사하세요:"
echo ""
echo "  📁 복사할 것:"
echo "     - configs/"
echo "     - data/"
echo "     - src/"
echo "     - offline_packages/"
echo "     - requirements.txt"
echo "     - start_offline.sh"
echo "     - .env.example"
echo ""
echo "  🚫 복사 안 해도 되는 것:"
echo "     - venv/           (용량 큼, OS마다 달라서 불필요)"
echo "     - .git/           (git 히스토리)"
echo "     - interviews.db   (초기화된 DB를 원한다면 포함 가능)"
echo "     - .env            (API 키 노출 주의! 직접 입력하게 할 것)"
echo "======================================================"
echo ""
