# NeoPharmabert — ICSR Validity Classifier

> Fine-tuned PubMedBERT for automated pharmacovigilance literature screening

![Model](https://img.shields.io/badge/Model-PubMedBERT-blue)
![F1 Score](https://img.shields.io/badge/F1%20Macro-0.976-brightgreen)
![Domain](https://img.shields.io/badge/Domain-Pharmacovigilance-orange)
![Task](https://img.shields.io/badge/Task-Multi--Label%20Classification-purple)

---

## Overview

**NeoPharmabert** is a fine-tuned [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) model that classifies PubMed abstracts for **ICSR (Individual Case Safety Report) validity** — a critical task in drug safety and pharmacovigilance literature screening.

Given a PubMed abstract, the model predicts whether each of the four ICSR validity criteria is present:

| Criterion | Description |
|-----------|-------------|
| **Patient** | Is an identifiable patient mentioned? |
| **Reporter** | Is an identifiable reporter mentioned? |
| **Drug** | Is a suspect drug mentioned? |
| **Event** | Is an adverse event mentioned? |

An abstract is classified as a **Valid ICSR** only if all four criteria are met.

---

## Model Performance

| Metric | Score |
|--------|-------|
| F1 Macro | **0.976** |
| Dataset Size | ~5,000 labeled abstracts |
| Base Model | PubMedBERT |
| Task | Multi-label binary classification |

---

## Project Structure

```
NeoPharmabert/
├── index.html              # Frontend UI (HTML/CSS/JS)
├── backend.py              # FastAPI backend
├── NeoPharmabert.ipynb     # Training notebook (Google Colab)
└── README.md               # Project documentation
```

---

## Run Locally

### 1. Install dependencies
```bash
pip install fastapi uvicorn transformers torch
```

### 2. Run the backend
```bash
uvicorn backend:app --reload
```

### 3. Open in browser
```
http://localhost:8000
```

---

## Model on HuggingFace

The trained model is hosted on HuggingFace Hub:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("dhanu-sree-1905/NeoPharmabert")
tokenizer = AutoTokenizer.from_pretrained("dhanu-sree-1905/NeoPharmabert")
```

---

## Tech Stack

- **Model:** PubMedBERT (microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
- **Fine-tuning:** HuggingFace Transformers, PyTorch
- **Backend:** FastAPI, Uvicorn
- **Frontend:** HTML, CSS, JavaScript
- **Training:** Google Colab (T4 GPU)
- **Hosting:** HuggingFace Hub

---

## About

Built by **Dhanusree** as a portfolio project at the intersection of NLP and Pharmacovigilance.

- Hands-on experience in ICSR validity assessment using the standard four-criterion framework
- Fine-tuned on ~5,000 labeled PubMed abstracts
- Deployed with a clean web UI for real-time classification
