# 🩺 MediLens

> **AI-Powered Medical Report Analysis & Health Intelligence Platform**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Overview

**MediLens** is a full-stack AI healthcare platform that transforms
complex laboratory reports into clear, visual, patient-friendly
insights.

It combines **OCR**, **AI**, **Retrieval-Augmented Generation (RAG)**,
**FastAPI**, and **Streamlit** to help users understand medical reports,
visualize affected organs, monitor health trends, and interact with an
AI assistant.

> ⚠️ **Disclaimer:** MediLens is an educational tool and does **not**
> replace advice from qualified healthcare professionals.

------------------------------------------------------------------------

# ✨ Features

-   📄 OCR-based report extraction
-   🤖 Google Gemini AI explanations
-   🧠 Retrieval-Augmented Generation (RAG)
-   🫀 Visual Organ Explorer
-   📊 Health Dashboard
-   📈 Trend Analysis
-   💬 AI Medical Chatbot
-   🔐 JWT Authentication
-   📤 PDF / CSV / JSON Export
-   🌐 REST API with FastAPI

------------------------------------------------------------------------

# 🏗️ Architecture

``` text
Medical Report
      │
      ▼
 OCR Engine
      │
      ▼
 Parser
      │
      ▼
 Database
      │
 ┌────┴─────┐
 ▼          ▼
AI Engine  Dashboard
 │          │
 ▼          ▼
Chatbot   Organ Explorer
      │
      ▼
 Streamlit Frontend
```

------------------------------------------------------------------------

# 🛠️ Tech Stack

  Category         Technologies
  ---------------- ---------------------
  Backend          FastAPI, SQLAlchemy
  Frontend         Streamlit
  AI               Google Gemini, RAG
  OCR              OCR Pipeline
  Database         SQLite
  Visualization    Plotly, SVG
  Authentication   JWT
  Language         Python 3.11

------------------------------------------------------------------------

# 📂 Project Structure

``` text
MediLens/
├── app.py
├── routes.py
├── streamlit_app.py
├── visual_engine.py
├── parser.py
├── ocr_engine.py
├── ai_engine.py
├── auth.py
├── database.py
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

# 🚀 Getting Started

``` bash
git clone https://github.com/ZahikAbasdar/MediLens.git
cd MediLens
python -m venv venv
```

Activate:

``` bash
# Windows
venv\Scripts\activate
```

Install:

``` bash
pip install -r requirements.txt
```

Run Backend

``` bash
python -m uvicorn app:app --reload
```

Run Frontend

``` bash
streamlit run streamlit_app.py
```

------------------------------------------------------------------------

# 🔑 Environment Variables

``` env
GEMINI_API_KEY=YOUR_API_KEY
SECRET_KEY=YOUR_SECRET_KEY
```


------------------------------------------------------------------------

# 🎯 Roadmap

-   [ ] Multi-language support
-   [ ] Cloud database
-   [ ] DICOM viewer
-   [ ] Doctor portal
-   [ ] Mobile application

------------------------------------------------------------------------

# 🤝 Contributing

Pull requests are welcome. Please open an issue first to discuss major
changes.

------------------------------------------------------------------------

# 👨‍💻 Author

**Zahik Abas dar**

GitHub: https://github.com/ZahikAbasdar

------------------------------------------------------------------------

# ⭐ Support

If this project helped you or you found it interesting, consider giving
it a **⭐ Star** on GitHub.
