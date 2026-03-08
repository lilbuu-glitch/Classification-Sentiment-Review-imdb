import re
import string
import logging
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self, remove_stopwords: bool = True, use_lemmatization: bool = True):
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        
    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        if not text:
            return []
        
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        if not self.stop_words:
            return tokens
        
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        return filtered_tokens
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        if not self.lemmatizer:
            return tokens
        
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens
    
    def preprocess_text(self, text: str) -> str:
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)
        
        if self.remove_stopwords:
            tokens = self.remove_stopwords_from_tokens(tokens)
        
        if self.use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        
        processed_text = ' '.join(tokens)
        return processed_text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        processed_texts = []
        
        for i, text in enumerate(texts):
            processed_text = self.preprocess_text(text)
            processed_texts.append(processed_text)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1}/{len(texts)} texts")
        
        logger.info(f"Batch preprocessing completed for {len(texts)} texts")
        return processed_texts

def main():
    preprocessor = TextPreprocessor(remove_stopwords=True, use_lemmatization=True)
    
    test_text = """
    This movie was absolutely fantastic! The acting was great and the plot was amazing. 
    I would definitely recommend it to anyone. Visit http://example.com for more info.
    """
    
    print("Original text:")
    print(test_text)
    print("\nProcessed text:")
    processed = preprocessor.preprocess_text(test_text)
    print(processed)

if __name__ == "__main__":
    main()
