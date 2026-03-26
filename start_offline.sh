#!/bin/bash
# =============================================================
#  🏙️ 부산 걷기 좋은 도시 AI 평가 시스템 - 오프라인 실행 스크립트
#  (인터넷 없이 USB에서 바로 실행 가능)
# =============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "======================================================"
echo "  🏙️  부산 걷기 좋은 도시 AI 평가 시스템"
echo "======================================================"
echo ""

# ── 1. Python 확인 ──────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "❌ Python3 가 설치되어 있지 않습니다."
    echo "   https://www.python.org 에서 Python 3.10 이상을 설치하세요."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python $PYTHON_VERSION 확인됨"

# ── 2. .env 파일 확인 ────────────────────────────────────────
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  .env 파일이 없습니다."
    echo ""
    read -p "   OpenAI API Key 를 입력하세요 (sk-...): " API_KEY
    echo "OPENAI_API_KEY=$API_KEY" > .env
    echo "✅ .env 파일 생성 완료"
else
    echo "✅ .env 파일 확인됨"
fi

# ── 3. 가상환경 설정 ─────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 가상환경을 생성합니다..."
    python3 -m venv venv
fi

echo "🔄 가상환경 활성화 중..."
source venv/bin/activate

# ── 4. 패키지 설치 (오프라인 우선) ──────────────────────────
if [ -d "offline_packages" ] && [ "$(ls -A offline_packages)" ]; then
    echo "📦 오프라인 패키지로 설치합니다 (인터넷 불필요)..."
    pip install -q --no-index --find-links=offline_packages/ -r requirements.txt
else
    echo "🌐 오프라인 패키지 없음 → 인터넷에서 설치합니다..."
    pip install -q -r requirements.txt
fi

echo "✅ 패키지 설치 완료"

# ── 5. Streamlit 실행 ────────────────────────────────────────
echo ""
echo "======================================================"
echo "  🚀  앱을 시작합니다: http://localhost:8501"
echo "  종료하려면 터미널에서 Ctrl+C 를 누르세요."
echo "======================================================"
echo ""

streamlit run src/app.py --server.headless false
