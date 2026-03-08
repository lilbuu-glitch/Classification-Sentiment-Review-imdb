# Sentiment Analysis ML Project

A production-ready machine learning project for sentiment analysis of IMDB movie reviews using Python, scikit-learn, and FastAPI.

## 🎯 Project Overview

This project implements a complete sentiment classification system that predicts whether a movie review is positive or negative. The system follows professional ML engineering practices with modular code, reproducible experiments, and API deployment.

### Key Features

- **Complete ML Pipeline**: Data preparation → EDA → Preprocessing → Feature engineering → Training → Evaluation → Deployment
- **Multiple Models**: Logistic Regression, Linear SVM, Random Forest with hyperparameter tuning
- **Production API**: FastAPI-based REST API with comprehensive validation and error handling
- **Reproducible**: Fixed random seeds, structured experiments, and comprehensive logging
- **Professional Structure**: Clean, modular codebase following best practices

## 📊 Dataset

The project uses the IMDB Movie Review Dataset with 50,000 movie reviews:

- **Input**: Movie review text
- **Output**: Sentiment label (positive/negative)
- **Size**: 50,000 reviews (balanced dataset)
- **Source**: `data/IMDB Dataset.csv`

## 🏗️ Project Structure

```
sentiment-analysis-ml/
├── data/
│   ├── IMDB Dataset.csv          # Original dataset
│   ├── train.csv                 # Training split (80%)
│   ├── val.csv                   # Validation split (10%)
│   └── test.csv                  # Test split (10%)
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb    # Exploratory data analysis
│   └── 02_model_training_and_evaluation.ipynb  # Model training and evaluation
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Data loading and validation
│   ├── preprocessing.py          # Text preprocessing pipeline
│   ├── feature_engineering.py    # TF-IDF feature engineering
│   ├── train.py                  # Model training and hyperparameter tuning
│   ├── evaluate.py               # Model evaluation and metrics
│   └── utils.py                  # Utility functions
├── api/
│   ├── __init__.py
│   ├── main.py                   # FastAPI application
│   ├── schema.py                 # Pydantic models for API
│   └── inference.py              # Model inference logic
├── scripts/
│   └── train_model.py            # Training script
├── models/
│   ├── best_model.joblib         # Trained model
│   └── feature_pipeline.joblib   # Feature engineering pipeline
├── evaluation_results/           # Model evaluation results
├── plots/                        # Visualization plots
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd sentiment-analysis-ml

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\\Scripts\\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 2. Train the Model

#### Option A: Using the Training Script (Recommended)

```bash
# Train the model with default settings
python scripts/train_model.py

# Train with custom parameters
python scripts/train_model.py \
    --max-features 15000 \
    --test-size 0.15 \
    --verbose
```

#### Option B: Using Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Run the notebooks in order:
# 1. 01_eda_and_preprocessing.ipynb
# 2. 02_model_training_and_evaluation.ipynb
```

### 3. Start the API Server

```bash
# Start the FastAPI server
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Or run from the root directory
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was absolutely fantastic!"}'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Great movie!", "Terrible plot."]}'
```

## 📖 API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API info |
| `/health` | GET | Health check |
| `/model/info` | GET | Model information |
| `/predict` | POST | Single text prediction |
| `/predict/batch` | POST | Batch text prediction |
| `/stats` | GET | API statistics |

### Request/Response Examples

#### Single Prediction

**Request:**
```json
{
  "text": "This movie was absolutely fantastic!"
}
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.94,
  "processing_time": 0.023
}
```

#### Batch Prediction

**Request:**
```json
{
  "texts": [
    "Great movie with excellent acting!",
    "Boring and predictable plot."
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "sentiment": "positive",
      "confidence": 0.91,
      "processing_time": 0.015
    },
    {
      "sentiment": "negative",
      "confidence": 0.87,
      "processing_time": 0.012
    }
  ],
  "total_processed": 2,
  "processing_time": 0.027
}
```

## 🧠 Model Details

### Feature Engineering

- **Text Preprocessing**: Lowercase, HTML tag removal, URL removal, punctuation removal, number removal, tokenization, stopword removal, lemmatization
- **Vectorization**: TF-IDF with n-grams (1,2)
- **Parameters**: max_features=10,000, min_df=5, max_df=0.9

### Models Trained

1. **Logistic Regression**
   - Regularization: L2
   - Solver: liblinear
   - C: [0.1, 1.0, 10.0]

2. **Linear SVM**
   - Loss: hinge
   - C: [0.1, 1.0, 10.0]

3. **Random Forest**
   - Trees: [100, 200]
   - Max depth: [10, 20, None]
   - Min samples split: [2, 5]

### Evaluation Metrics

- **Accuracy**: Overall prediction accuracy
- **Precision**: Weighted precision score
- **Recall**: Weighted recall score
- **F1-score**: Weighted F1-score (primary metric)
- **ROC-AUC**: Area under ROC curve

## 📊 Performance

Typical performance on the test set:

- **Accuracy**: ~0.89
- **F1-Score**: ~0.89
- **ROC-AUC**: ~0.95

*Note: Actual performance may vary based on random seed and data split.*

## 🔧 Configuration

### Training Parameters

```python
config = {
    'max_features': 10000,        # TF-IDF max features
    'ngram_range': (1, 2),        # N-gram range
    'min_df': 5,                  # Minimum document frequency
    'max_df': 0.9,                # Maximum document frequency
    'remove_stopwords': True,     # Remove stopwords
    'use_lemmatization': True,    # Use lemmatization
    'test_size': 0.2,             # Test set proportion
    'val_size': 0.1,              # Validation set proportion
    'random_state': 42            # Random seed
}
```

### Model Selection

Models are selected based on validation F1-score. The best model is automatically saved and used for inference.

## 🧪 Testing

```bash
# Run tests (if available)
pytest tests/

# Test API endpoints
python -c "
import requests
response = requests.post('http://localhost:8000/predict', 
                        json={'text': 'Great movie!'})
print(response.json())
"
```

## 📝 Logging

The application uses Python's logging module with:

- **Console output**: Real-time logs
- **File output**: `sentiment_analysis.log`
- **Log levels**: DEBUG, INFO, WARNING, ERROR

## 🔄 Reproducibility

- **Fixed random seeds**: All random operations use seed=42
- **Version pinned dependencies**: requirements.txt with specific versions
- **Structured experiments**: All parameters and results logged
- **Model serialization**: Complete pipeline saved for reproducible inference

## 🐳 Docker Deployment (Optional)

```dockerfile
# Dockerfile example
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

## 📚 Development Guide

### Adding New Models

1. Update `ModelTrainer.model_configs` in `src/train.py`
2. Add hyperparameter grid
3. Retrain using the training script

### Extending Preprocessing

1. Modify `TextPreprocessor` class in `src/preprocessing.py`
2. Update `TextPreprocessorTransformer` in `src/feature_engineering.py`
3. Retrain the model with new preprocessing

### API Extensions

1. Add new schemas in `api/schema.py`
2. Implement endpoints in `api/main.py`
3. Update inference logic in `api/inference.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper testing
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- IMDB Dataset providers
- scikit-learn team for excellent ML tools
- FastAPI team for the web framework
- NLTK and spaCy teams for NLP tools

## 📞 Support

For issues and questions:

1. Check the logs in `sentiment_analysis.log`
2. Review the troubleshooting section below
3. Create an issue in the repository

## 🔍 Troubleshooting

### Common Issues

**Model loading errors:**
```bash
# Ensure model files exist
ls -la models/
# Re-train if missing
python scripts/train_model.py
```

**API server errors:**
```bash
# Check logs
tail -f sentiment_analysis.log
# Verify dependencies
pip install -r requirements.txt
```

**Memory issues:**
```bash
# Reduce max_features in training
python scripts/train_model.py --max-features 5000
```

**Import errors:**
```bash
# Ensure you're in the project root
cd sentiment-analysis-ml
# Install in development mode
pip install -e .
```
