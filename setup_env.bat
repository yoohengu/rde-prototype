@echo off
echo =============================================
echo 🚀 Python Virtual Environment Setup Started
echo =============================================

:: 1️⃣ 가상환경 생성
if not exist .venv (
    python -m venv .venv
    echo ✅ Virtual environment created (.venv)
) else (
    echo ⚙️ Virtual environment already exists (.venv)
)

:: 2️⃣ 가상환경 활성화
call .venv\Scripts\activate

:: 3️⃣ pip 최신화
echo 🔄 Upgrading pip...
python -m pip install --upgrade pip

:: 4️⃣ 패키지 설치
echo 📦 Installing required packages...
pip install -r requirements.txt

echo =============================================
echo ✅ Setup complete! Environment is ready.
echo =============================================

pause
