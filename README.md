# Resume & JD Matcher Dashboard

A Streamlit-based application that matches resumes to job descriptions using sentence embeddings and regression, with built-in explainability via SHAP and LIME. Designed for recruiters, job seekers, and ML enthusiasts who want transparent, interpretable matching.

---

## Features

-  **Semantic Matching**: Uses SentenceTransformer to encode resumes and job descriptions
-  **Regression Model**: Predicts a match score between 0 and 1
-  **Explainability**:
  - SHAP text plots to highlight impactful resume terms
  - LIME explanations for word-level contribution analysis
-  **Compact UI**: Responsive layout with scrollable visualizations and recruiter-friendly styling

---

##  Tech Stack

- `Streamlit` – UI and dashboard
- `SentenceTransformer` – Embedding model
- `scikit-learn` – Regression training
- `SHAP` & `LIME` – Model explainability
- `Docker` (optional) – Containerized deployment

---

## Installation

```bash
git clone https://github.com/your-username/resume-jd-matcher.git](https://github.com/dhananjayaDev/resume-jd-matcher.git
cd resume-jd-matcher
pip install -r requirements.txt
```

---

## Usage

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

---

# Input Format

- **Resume**: Paste plain text (e.g., project descriptions, skills)
- **Job Description**: Paste plain text (e.g., responsibilities, qualifications)

---

## Output

- **Match Score**: A float between 0 and 1
- **SHAP Explanation**: Highlights which resume words influenced the score
- **LIME Explanation**: Lists top contributing words with weights

---

## Model Training

The model is trained using:
- Sentence embeddings of job descriptions
- Resume–JD pairs with labeled match scores
- A regression model (`Ridge`, `SVR`, etc.)

Training logic is modularized in `model/trainer.py`.

---

## Explainability Modules

- `explainability/shap_regressor.py`: Generates SHAP values and text plots
- `explainability/lime_explainer.py`: Generates LIME explanations using word-level perturbations

---

## Preprocessing

- `utils/preprocess.py`: Cleans and normalizes input text (lowercasing, punctuation removal, etc.)

---

## Project Structure

```
resume-jd-matcher/
├── app.py
├── model/
│   └── trainer.py
├── explainability/
│   ├── shap_regressor.py
│   └── lime_explainer.py
├── utils/
│   └── preprocess.py
├── requirements.txt
└── README.md
```

---

## Notes

- SHAP plots may appear flat if the model lacks training data or feature sensitivity
- LIME explanations are more robust for short resumes
- Consider fine-tuning the embedding model for better domain alignment

---

## License

MIT License. See `LICENSE` for details.

---

## Acknowledgments

- [SHAP](https://github.com/slundberg/shap)
- [LIME](https://github.com/marcotcr/lime)
- [SentenceTransformers](https://www.sbert.net/)
```
