Perfect 🎯 — here’s a **ready-to-upload, beautifully formatted `README.md`** for your GitHub project

> **🎓 Student Dev Suite** — the ultimate AI + ML powered student toolkit 🚀

You can copy this directly into your repo root as `README.md`.
It includes badges, features, screenshots placeholders, and proper descriptions.

---

````markdown
# 🎓 Student Dev Suite

> 🧠 An AI + ML powered productivity suite for students — featuring an AI Assistant, Resume Builder, IDE, Study Recommender, and AI Text Detector (ML).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit)
![Groq](https://img.shields.io/badge/AI-Groq%20LLaMA%203-00cc66?logo=openai)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🌟 Overview

**Student Dev Suite** is a modern, multi-module application built in **Streamlit** to assist students in coding, career building, and learning with integrated AI & ML capabilities.  
It provides interactive tools for generating resumes, running code in multiple languages, analyzing text, and improving your study experience.

---

## 🚀 Features

### 🤖 **AI Assistant (Groq LLaMA 3)**
Ask coding, AI/ML, or academic questions directly inside the app.  
Powered by Groq’s ultra-fast LLaMA-3.3 model.

### 🧾 **Resume Generator (Hybrid Layout)**
- Add your photo, education, and projects.
- Exports **DOCX** and **PDF** formats.
- Auto-formatted hybrid design (Modern + Classic).
- Smart keyword suggestions for job optimization.

### 💻 **Code IDE**
- Run **Python, Java, C++, and JavaScript**.
- Syntax highlighting (via `streamlit-ace`).
- Output console for Stdout/Stderr.
- Lightweight, sandboxed execution.

### 📘 **Document & PDF Creator**
- Create notes or formatted documents.
- Export instantly to DOCX or PDF.

### 🧠 **AI Text Detector (ML)**
Detect whether your text is **AI-generated or human-written** using heuristic and linguistic signals.

**Features:**
- Upload up to 5 text files.
- Calculates **AI Probability (%)** for each.
- Displays verdicts:
  - ✅ Likely human-written  
  - 🤔 Possibly AI-edited  
  - ⚠️ Highly likely AI-generated
- Shows overall **average AI probability**.
- Smooth Lottie animation for visualization.

### 📚 **Study Recommender**
Suggests learning resources and project ideas based on:
- Focus area (e.g., Python, ML, Web Dev)
- Skill level (Beginner → Advanced)

### 🧩 **Resume Enhancer (ML)**
Compares your resume text vs job description using **TF-IDF**.
- Suggests missing keywords.
- Measures similarity score.

### 🧮 **Progress Predictor (ML)**
Predicts your task completion based on weekly practice hours using **Linear Regression**.

### 🔍 **Code Insights (ML)**
- Static analysis for Python, JS, C++, Java.
- Checks complexity via `radon`.
- Integration with **Pylint** for detailed feedback.

---

## 🪄 Visuals

| Page | Preview |
|------|----------|
| 🏠 Home (Animated) | ![Home Animation](https://lottie.host/8e85fca2-2d19-4e48-a0a7-cc8a1e80c82b/ld7h02r8s2.json) |
| 🤖 AI Assistant | *(Example screenshot placeholder)* |
| 🧠 AI Text Detector | *(Example screenshot placeholder)* |
| 🧾 Resume Generator | *(Example screenshot placeholder)* |

> 💡 *You can replace these placeholders with screenshots of your running Streamlit app.*

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/Student-Dev-Suite.git
cd Student-Dev-Suite
````

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # (on Windows)
source venv/bin/activate  # (on Mac/Linux)
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

> If you don’t have a `requirements.txt` yet, use:
>
> ```bash
> pip install streamlit groq python-docx fpdf2 streamlit-ace scikit-learn transformers radon pylint
> ```

### 4️⃣ Set up your `.env` file

Create a file named `.env` in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🧠 Run the App

```bash
streamlit run streamlit_student_suite.py
```

---

## 🧩 Project Structure

```
📦 Student-Dev-Suite
├── streamlit_student_suite.py    # main app
├── requirements.txt              # dependencies
├── .env                          # API key file
├── studentsuite_users.db         # local user database

```

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit + Lottie Animations
* **Backend:** Python
* **AI API:** Groq LLaMA 3
* **ML Models:** scikit-learn, Transformers, Radon
* **Doc Generation:** python-docx, FPDF2

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use and modify.

---

## ✨ Author

👩‍💻 **Kanika Manwal**
💼 *Developer | AI & ML Enthusiast*
🔗 [GitHub](https://github.com/kanika-manwal) | [LinkedIn](https://linkedin.com/in/)



---

```
