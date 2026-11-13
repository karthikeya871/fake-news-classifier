import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords',quiet=True)
STOP_WORDS = set(stopwords.words('english'))
def clean_text(text: str, remove_stopwords: bool = True)-> str:
    """Cleans input text by lowercasing, stripping punctuation, numbers, and stopwords. """
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\d+', '', text)

    words = text.split()
    if remove_stopwords:
         cleaned_words = [word for word in words if word not in STOP_WORDS]
    return " ".join(cleaned_words)

