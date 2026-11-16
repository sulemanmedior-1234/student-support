# Student Support and Feedback Assistant

An end-to-end multi-modal AI capstone project featuring Supervised Learning, Unsupervised Clustering, NLP Sentiment Analysis, Deep Learning (CNN), and a modular Streamlit frontend with optional Groq LLM integration.

## Project Structure
- `app.py`: Main 5-tab Streamlit web application.
- `models/`: Modular scripts for supervised trees, KMeans, CNN, and VADER NLP.
- `utils/`: Data loading and LLM helper utilities.
- `data/`: Directory for placing `main.csv`, `feedback.csv`, `meta.json`, and `images.npz`.

## Local Setup & Execution
1. Clone repository and place dataset files inside the `data/` folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt