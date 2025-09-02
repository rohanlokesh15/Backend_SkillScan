# backend/setup_nltk.py
import nltk
for r in ["punkt"]:
    nltk.download(r)
print("Downloaded NLTK data.")
