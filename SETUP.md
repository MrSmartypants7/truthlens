# GitHub Setup Guide

## 1. Create the repo on GitHub

Go to https://github.com/new:
- Name: `truthlens`
- Visibility: **Public**
- Do NOT initialize with a README

## 2. Push the code

```bash
cd truthlens-repo

git init
git add .
git commit -m "feat: initial TruthLens release — FAISS + Ollama hallucination detection"

git remote add origin https://github.com/YOUR_USERNAME/truthlens.git
git branch -M main
git push -u origin main
```

## 3. Enable GitHub Pages

1. Go to your repo on GitHub
2. Click **Settings** → **Pages** (left sidebar)
3. Under **Source**, select **Deploy from a branch**
4. Set Branch: `main` / Folder: `/docs`
5. Click **Save**

Your landing page will be live at:
**https://YOUR_USERNAME.github.io/truthlens**

(Takes 1–2 minutes to deploy the first time.)

## 4. Update YOUR_USERNAME everywhere

Find and replace `YOUR_USERNAME` in:
- `README.md` — badge URL + clone URL
- `docs/index.html` — GitHub button links (search for `github.com`)

## 5. Add repo metadata on GitHub

**Description:** Real-time LLM hallucination detection using FAISS vector search and Ollama — no API keys required.

**Website:** https://YOUR_USERNAME.github.io/truthlens

**Topics:** `llm` `hallucination-detection` `faiss` `ollama` `fastapi` `rag` `python` `vector-search`

## 6. Pin the repo on your profile

GitHub profile → Edit profile → Customize pins → select `truthlens`
