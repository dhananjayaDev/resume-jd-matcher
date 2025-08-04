# 🧠 Resume & JD Matcher Dashboard

A Streamlit-based application that matches resumes to job descriptions using sentence embeddings and regression, with built-in explainability via SHAP and LIME. Designed for recruiters, job seekers, and ML enthusiasts who want transparent, interpretable matching.

---

## 🚀 Features

- 🔍 **Semantic Matching**: Uses SentenceTransformer to encode resumes and job descriptions
- 📈 **Regression Model**: Predicts a match score between 0 and 1
- 🧠 **Explainability**:
  - SHAP text plots to highlight impactful resume terms
  - LIME explanations for word-level contribution analysis
- 🎨 **Compact UI**: Responsive layout with scrollable visualizations and recruiter-friendly styling

---

## 🧰 Tech Stack

- `Streamlit` – UI and dashboard
- `SentenceTransformer` – Embedding model
- `scikit-learn` – Regression training
- `SHAP` & `LIME` – Model explainability
- `Docker` (optional) – Containerized deployment

---

## 📦 Installation

```bash
git clone https://github.com/your-username/resume-jd-matcher.git
cd resume-jd-matcher
pip install -r requirements.txt