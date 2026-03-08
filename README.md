# Classification Sentiment Review IMDB

Proyek ini adalah implementasi end-to-end Machine Learning Pipeline untuk klasifikasi sentimen ulasan film menggunakan dataset IMDB. Dibangun sebagai bagian dari penilaian teknis AI Engineer.

---

## Tahap 1: Data Preparation & Exploratory Data Analysis (EDA)

Pada tahap awal, saya melakukan analisis mendalam terhadap **IMDB Dataset (50,000 reviews)** untuk memahami karakteristik data sebelum masuk ke proses modeling.

### 1.1 Data Loading & Distribution
Data dimuat menggunakan `pandas`. Distribusi sentimen menunjukkan dataset yang sangat seimbang (**25,000 Positif** dan **25,000 Negatif**), sehingga tidak diperlukan teknik resampling (Over/Under-sampling).

### 1.2 Text Preprocessing
Proses pembersihan teks dilakukan melalui pipeline modular di `src/preprocessing.py` yang mencakup:
*   **Cleaning**: Penghapusan tag HTML, URL, angka, dan karakter khusus.
*   **Tokenization**: Memecah kalimat menjadi token kata individual menggunakan `nltk`.
*   **Stopword Removal**: Menghapus kata-kata umum bahasa Inggris yang tidak membawa makna sentimen (e.g., "is", "the", "at").
*   **Lemmatization**: Mengubah kata ke bentuk dasarnya (e.g., "running" -> "run") menggunakan WordNetLemmatizer untuk mengurangi dimensionalitas fitur.

---

## Tahap 2: Model Development (Core AI)

Saya menggunakan pendekatan **Traditional Machine Learning (TF-IDF + Scikit-Learn)** karena efisiensinya yang tinggi pada dataset ulasan film yang besar tanpa memerlukan sumber daya GPU yang intensif seperti Transformer.

### 2.1 Feature Engineering
Teks dikonversi menjadi representasi numerik menggunakan **TF-IDF Vectorizer** dengan parameter:
*   `max_features`: 10,000 (untuk membatasi kompleksitas).
*   `ngram_range`: (1, 2) (menangkap kombinasi dua kata seperti "not good").

### 2.2 Model Selection & Evaluation
Saya melatih beberapa model (Logistic Regression, Linear SVM, dan Random Forest). **Logistic Regression** terpilih sebagai model terbaik berdasarkan kestabilan F1-Score pada data validasi.

**Test Metrics (Best Model):**
| Metric | Score |
| :--- | :--- |
| **Accuracy** | 0.8926 |
| **F1-Score** | 0.8920 |
| **Precision** | 0.8924 |
| **Recall** | 0.8926 |
| **ROC-AUC** | 0.9576 |

### 2.3 Visualisasi Performa
Berikut adalah bukti nyata performa model pada test set (20% data):

#### Confusion Matrix
![Confusion Matrix](Image/Output%20Confusion%20Matrix.png)

*Model menunjukkan kemampuan seimbang dalam mengklasifikasikan kelas positif dan negatif.*

#### ROC Curve
![ROC Curve](Image/Output%20ROC.png)

*Skor AUC sebesar 0.96 menunjukkan pemisahan kelas yang sangat baik.*

---

## Tahap 3: Model Serving & Deployment

Model terbaik disimpan menggunakan `joblib` ke dalam folder `models/` agar siap digunakan untuk produksi (Model Serving).

### 3.1 REST API via FastAPI
Model dibungkus menggunakan **FastAPI** untuk menyediakan layanan prediksi real-time. API dirancang untuk menerima payload JSON dan mengembalikan hasil prediksi beserta probabilitas keyakinannya.

*   **Endpoint Utama**: `POST /predict`
*   **Payload**: `{"text": "Isi ulasan di sini..."}`
*   **Output**: Hasil sentimen (Positif/Negatif) dan confidence score.

---

## Tahap 4: Dokumentasi & Instalasi

### Alasan Pemilihan Arsitektur
Saya memilih **TF-IDF + Logistic Regression** karena:
1.  **Kecepatan**: Inference time yang sangat rendah (ideal untuk API).
2.  **Explainability**: Bobot fitur mudah diinterpretasi untuk melihat kata-apa saja yang memicu sentimen positif/negatif.
3.  **Akurasi**: Performa mendekati model modern Transformer pada dataset IMDB namun jauh lebih ringan secara infrastruktur.

### Cara Menjalankan di Lokal

**1. Clone Repository & Install Dependencies:**
```bash
git clone https://github.com/lilbuu-glitch/Classification-Sentiment-Review-imdb.git
cd Classification-Sentiment-Review-imdb
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate untuk Windows
pip install -r requirements.txt
```

**2. Download NLTK Data:**
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

**3. Jalankan API Server:**
```bash
python -m uvicorn api.main:app --reload
```

### Contoh Request & Response

**Request (cURL):**
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "The movie was an absolute masterpiece. The cinematography was breathtaking and the acting was top-notch."
}'
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.9842,
  "processing_time": 0.015
}
```

---
*Dibuat oleh AI Engineer Candidate - 2024*
