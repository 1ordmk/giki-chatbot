
#!/bin/bash

set -e

echo "🚀 Uploading GIKI Smart Assistant to GitHub..."

GITHUB_USER="1ordmk"

REPO_NAME="giki-chatbot"

cat > .gitignore << 'GITIGNORE'

.env

*.env

!.env.example

__pycache__/

*.pyc

venv/

data/raw/*.json

data/processed/*.json

data/analysis/*.npy

.DS_Store

*.log

*.db

chromedriver*

GITIGNORE

cat > .env.example << 'ENVEX'

PINECONE_API_KEY=your-key-here

PINECONE_INDEX_HOST=your-host-here

LLM_PROVIDER=codellama:13b

SECRET_KEY=change-me

ADMIN_USERNAME=admin

ADMIN_PASSWORD=admin123

WEATHER_API_KEY=your-key

EXCHANGE_RATE_API_KEY=your-key

NEWS_API_KEY=your-key

GIKI_LATITUDE=33.9407

GIKI_LONGITUDE=72.6267

ENVEX

mkdir -p data/{raw,processed,analysis}

touch data/{raw,processed,analysis}/.gitkeep

[ ! -d ".git" ] && git init

[ -z "$(git config user.name)" ] && git config user.name "1ordmk" && git config user.email "your-email@example.com"

git add .

git commit -m "Initial commit: GIKI Smart Assistant with RAG pipeline" || echo "Already committed"

git branch -M main

git remote remove origin 2>/dev/null || true

git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git

echo ""

echo "✅ Ready! Press Enter to push (you'll need your GitHub token)"

read

git push -u origin main && echo "🎉 SUCCESS! https://github.com/$GITHUB_USER/$REPO_NAME" || echo "❌ Failed - create repo first at https://github.com/new"

