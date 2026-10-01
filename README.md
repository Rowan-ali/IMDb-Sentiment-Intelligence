# 🎬 IMDb Sentiment Intelligence

<p align="center">
  <strong>End-to-End NLP Sentiment Analysis • Classical ML • Deep Learning • Transformers • Streamlit</strong>
</p>

<p align="center">
  <a href="https://imdb-sentiment-intelligence-jhwc5r5xlpzby5tpmttdwa.streamlit.app/"><strong>🚀 Live Application</strong></a>
  &nbsp;•&nbsp;
  <a href="demo/IMDb_Sentiment_Intelligence_Demo.mp4"><strong>🎥 Video Demo</strong></a>
  &nbsp;•&nbsp;
  <a href="notebooks/IMDb_Sentiment_Analysis.ipynb"><strong>📓 Project Notebook</strong></a>
</p>

<p align="center">
  <img src="assets/streamlit_demo.png"
       alt="IMDb Sentiment Intelligence Streamlit Application"
       width="900">
</p>

<p align="center">
  <em>Interactive sentiment classification powered by the optimized TF-IDF + Logistic Regression pipeline.</em>
</p>

---

## 🚀 Live Application

### 👉 [Launch IMDb Sentiment Intelligence](https://imdb-sentiment-intelligence-jhwc5r5xlpzby5tpmttdwa.streamlit.app/)

Enter or paste a movie review to receive a **Positive** or **Negative** sentiment prediction together with confidence and class probabilities.

## 🎥 Project Demo

### 👉 [Watch the Full Project Demo](demo/IMDb_Sentiment_Intelligence_Demo.mp4)

## 📓 Complete Notebook

### 👉 [Open the Complete Jupyter Notebook](notebooks/IMDb_Sentiment_Analysis.ipynb)

---

## 📌 Project Overview

**IMDb Sentiment Intelligence** is an end-to-end NLP system designed to classify movie reviews as either **Positive** or **Negative**.

The project goes beyond training a single classifier. It develops and evaluates multiple generations of NLP approaches, beginning with a strong **TF-IDF + Logistic Regression baseline**, progressing through **Word2Vec and recurrent neural networks**, and finally exploring a **pretrained Transformer**.

The workflow also includes controlled hyperparameter experiments, regularization analysis, qualitative investigation of misclassified reviews, and deployment of the strongest directly comparable full-test model through an interactive **Streamlit web application**.

---

## 🎯 Project Objectives

The main objectives of this project are to:

- Build a reproducible sentiment-analysis pipeline.
- Explore and preprocess the IMDb movie review dataset.
- Preserve meaningful linguistic information such as negation.
- Establish a strong classical machine-learning baseline.
- Train Word2Vec embeddings for semantic exploration.
- Build and compare Simple RNN, LSTM, Bidirectional LSTM, and GRU architectures.
- Investigate Dropout regularization and recurrent-model overfitting.
- Tune vocabulary size and maximum sequence length.
- Explore pretrained Transformer-based sentiment classification.
- Compare models using consistent evaluation metrics.
- Analyze real model failures qualitatively.
- Deploy a trained sentiment classifier through Streamlit.

---

## 🗂️ Dataset

The project uses the **IMDb Movie Review Dataset** available through the Keras dataset API.

### Dataset Structure

| Split | Reviews |
|---|---:|
| Official Training Set | 25,000 |
| Official Test Set | 25,000 |
| **Total** | **50,000** |

The dataset contains two balanced sentiment classes:

- `0` → Negative
- `1` → Positive

The official test set is preserved for model evaluation rather than merging all reviews and creating a new random train/test split.

For neural-model development, the official training data is further divided into:

| Development Split | Reviews |
|---|---:|
| Training | 20,000 |
| Validation | 5,000 |

---

## 🧠 Complete NLP Pipeline

```text
IMDb Reviews
      │
      ▼
Data Inspection
      │
      ▼
Text Cleaning
      │
      ├──────────────────────────────┐
      │                              │
      ▼                              ▼
TF-IDF                         Token Sequences
      │                              │
      ▼                              ▼
Logistic Regression              Embedding
                                     │
                         ┌───────────┼───────────┐
                         ▼           ▼           ▼
                        RNN         LSTM         GRU
                                     │
                                     ▼
                               Bidirectional LSTM

Additional Experiments
      │
      ├── Word2Vec
      ├── Dropout Regularization
      ├── Vocabulary / Sequence-Length Tuning
      ├── Pretrained Transformer
      └── Error Analysis

Final Deployment
      │
      ▼
Optimized TF-IDF + Logistic Regression
      │
      ▼
Streamlit Application
```

---

## 🧹 Text Preprocessing

The preprocessing strategy was intentionally conservative to avoid destroying useful sentiment information.

The pipeline includes:

- Lowercasing
- Removal of dataset-specific special markers
- HTML break cleanup
- URL removal
- Removal of unnecessary numeric and non-alphabetic characters
- Apostrophe preservation
- Whitespace normalization

Aggressive linguistic transformations were intentionally avoided.

In particular, **negation terms such as `not`, `no`, and `never` are preserved**, because they can directly reverse sentiment.

Stopword removal, stemming, and lemmatization were not automatically applied because removing or altering words without considering sentiment context can damage useful information.

---

# 🔬 Modeling Experiments

## 1. TF-IDF + Logistic Regression

A classical machine-learning baseline was developed using TF-IDF features and Logistic Regression.

The baseline was subsequently optimized through TF-IDF configuration experiments performed using development training and validation data.

### Final TF-IDF Configuration

The selected representation uses:

- Maximum features: **30,000**
- N-gram range: **(1, 2)**
- Classifier: **Logistic Regression**

### Official Test Performance

| Metric | Score |
|---|---:|
| Accuracy | **89.58%** |
| Precision | **89.31%** |
| Recall | **89.94%** |
| F1-score | **89.62%** |

The optimized TF-IDF + Logistic Regression pipeline produced the strongest directly comparable performance among the models evaluated on the complete official 25,000-review test set.

---

## 2. Word2Vec

A Word2Vec model was trained on the training reviews to investigate distributed semantic representations.

### Configuration

```text
Vector Size : 100
Window      : 5
Min Count   : 5
Architecture: Skip-gram
Epochs      : 10
```

Word2Vec was used to explore semantic similarity and demonstrate how words can be represented in a continuous embedding space rather than as sparse frequency-based features.

---

## 3. Simple RNN

A Simple RNN was developed as the first recurrent neural architecture.

### Architecture

```text
Integer Sequence
      ↓
Embedding (128)
      ↓
SimpleRNN (64)
      ↓
Dense (32, ReLU)
      ↓
Sigmoid Output
```

Dropout experiments were subsequently performed to investigate and control overfitting.

### Final Test Performance

| Metric | Score |
|---|---:|
| Accuracy | 83.88% |
| Precision | 85.75% |
| Recall | 81.27% |
| F1-score | 83.45% |

The experiment demonstrated that a basic recurrent architecture can learn sentiment patterns, but it was less effective than the classical TF-IDF baseline.

---

## 4. Dropout Regularization Experiment

Multiple dropout values were evaluated for the Simple RNN:

```text
0.0
0.2
0.4
0.5
```

A dropout rate of **0.2** produced the strongest selected validation behavior according to the predefined criterion.

The experiment showed that moderate regularization can reduce the severity of overfitting, while stronger dropout does not automatically produce better generalization.

---

## 5. LSTM

An LSTM network was developed to improve the modeling of longer-term dependencies compared with the Simple RNN.

### Architecture

```text
Integer Sequence
      ↓
Embedding (128)
      ↓
LSTM (64)
      ↓
Dense (32, ReLU)
      ↓
Sigmoid Output
```

### Final Test Performance

| Metric | Score |
|---|---:|
| Accuracy | 85.36% |
| Precision | **88.72%** |
| Recall | 81.03% |
| F1-score | 84.70% |

The LSTM improved overall accuracy and F1-score relative to the Simple RNN and achieved particularly strong precision.

---

## 6. Bidirectional LSTM

A Bidirectional LSTM was implemented as an additional experiment to allow sequence information to be processed in both directions.

### Final Test Performance

| Metric | Score |
|---|---:|
| Accuracy | 85.55% |
| Precision | 82.59% |
| Recall | **90.10%** |
| F1-score | 86.18% |

Compared with the standard LSTM, the Bidirectional LSTM produced substantially higher recall and a stronger F1-score, while precision decreased.

---

## 7. GRU

A GRU was implemented as a more compact gated recurrent architecture.

### Architecture

```text
Integer Sequence
      ↓
Embedding (128)
      ↓
GRU (64)
      ↓
Dense (32, ReLU)
      ↓
Sigmoid Output
```

### Final Test Performance

| Metric | Score |
|---|---:|
| Accuracy | **86.92%** |
| Precision | 85.38% |
| Recall | **89.10%** |
| F1-score | **87.20%** |

Among the primary recurrent architectures, the GRU achieved the strongest overall combination of accuracy, recall, and F1-score.

---

# 📊 Recurrent Model Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Simple RNN | 83.88% | 85.75% | 81.27% | 83.45% |
| LSTM | 85.36% | **88.72%** | 81.03% | 84.70% |
| GRU | **86.92%** | 85.38% | **89.10%** | **87.20%** |

The results demonstrate that increasing architectural complexity alone does not guarantee better performance. The GRU provided the strongest overall recurrent-model performance for this task.

---

# ⚙️ Vocabulary Size & Sequence Length Tuning

The GRU was used to investigate the interaction between vocabulary size and maximum sequence length.

### Search Space

**Vocabulary sizes**

```text
10,000
15,000
20,000
```

**Sequence lengths**

```text
300
500
700
```

This produced **nine experimental configurations**.

The configuration selected according to the predefined validation criterion was:

```text
Vocabulary Size : 10,000
Sequence Length : 700
```

### Original vs Final Tuned GRU

| Model | Vocabulary | Sequence | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Original GRU | 19,679 | 500 | **86.92%** | **85.38%** | 89.10% | **87.20%** |
| Tuned GRU | 10,000 | 700 | 86.09% | 83.22% | **90.41%** | 86.67% |

The selected tuned configuration increased positive-class recall by **1.31 percentage points**, but decreased accuracy by **0.83 percentage points** and F1-score by **0.53 percentage points**.

This experiment illustrates an important machine-learning principle:

> Hyperparameter tuning can change a model's error trade-off, but validation improvements do not guarantee superior final test performance.

The original GRU therefore remains the preferred GRU configuration for overall classification performance.

---

# 🤗 Pretrained Transformer

A pretrained **DistilBERT sentiment classifier** was also explored using Hugging Face Transformers.

Model:

```text
distilbert-base-uncased-finetuned-sst-2-english
```

The Transformer was evaluated on a **50-review sample** from the official test set.

### Sample Performance

| Metric | Score |
|---|---:|
| Accuracy | 90.00% |
| Precision | 95.83% |
| Recall | 85.19% |
| F1-score | 90.20% |

> **Important:** These Transformer metrics are based on only 50 reviews and therefore should not be directly ranked against models evaluated on all 25,000 official test reviews.

The experiment demonstrates the power of contextual pretrained language representations while maintaining a methodologically fair interpretation of the results.

---

# 🏆 Overall Model Comparison

| Model | Representation | Accuracy | Precision | Recall | F1 | Evaluation Size |
|---|---|---:|---:|---:|---:|---:|
| **TF-IDF + Logistic Regression** | TF-IDF | **89.58%** | 89.31% | **89.94%** | **89.62%** | 25,000 |
| Simple RNN | Learned Embedding | 83.88% | 85.75% | 81.27% | 83.45% | 25,000 |
| LSTM | Learned Embedding | 85.36% | 88.72% | 81.03% | 84.70% | 25,000 |
| GRU | Learned Embedding | 86.92% | 85.38% | 89.10% | 87.20% | 25,000 |
| Pretrained DistilBERT | Contextual Embeddings | 90.00% | **95.83%** | 85.19% | 90.20% | 50 |

### Best Directly Comparable Model

🏆 **Optimized TF-IDF + Logistic Regression**

The Transformer produced promising sample-level results, but its evaluation size is not directly comparable with the complete-test evaluations.

This distinction is important to avoid drawing conclusions from unequal experimental conditions.

---

# 🔎 Qualitative Error Analysis

Five high-confidence misclassified reviews from the final tuned GRU were investigated manually.

The analysis revealed several recurring challenges.

### 1. Mixed Sentiment

Some reviews simultaneously praise and criticize different aspects of a movie, making the final sentiment difficult to infer from local sentiment cues.

### 2. Negation

Expressions such as:

```text
not as bad
not bored
not the worst
```

contain negative words but communicate neutral or positive sentiment when interpreted compositionally.

### 3. Sentiment-Target Ambiguity

A review may praise a director or DVD collection while criticizing the actual movie. A sequence classifier must determine which entity the sentiment refers to.

### 4. Long-Range Context

Long reviews can contain extensive criticism before ending with an overall positive recommendation or rating. The model may over-weight the dominant local sentiment rather than the final judgment.

### 5. Possible Label Noise

One high-confidence false positive contained overwhelmingly positive language, an explicit recommendation, and a **7/10 rating**, despite having a negative dataset label.

This suggests that some apparent model errors may reflect annotation ambiguity or label noise rather than purely model failure.

---

# 🌐 Streamlit Application

A professional Streamlit interface was developed to make the trained sentiment classifier interactive.

The deployed application uses:

```text
Optimized TF-IDF
        +
Logistic Regression
```

The user enters a movie review and receives:

- Predicted sentiment
- Positive / Negative classification
- Prediction confidence
- Positive probability
- Negative probability
- Processed-text preview
- Model information
- Dark / Light interface support

### Why TF-IDF + Logistic Regression for Deployment?

The optimized TF-IDF + Logistic Regression pipeline was selected because it:

- Achieved **89.58% accuracy** on all 25,000 official test reviews.
- Produced the strongest directly comparable full-test performance.
- Provides very fast inference.
- Requires relatively few computational resources.
- Does not require GPU acceleration.
- Does not require downloading a large pretrained model at application startup.
- Is highly suitable for lightweight real-time deployment.

---

# 🖥️ Streamlit Application Structure

```text
app/
│
├── app.py
├── sentiment_model.pkl
└── tfidf_vectorizer.pkl
```

The serialized files contain the already-trained model artifacts. The Streamlit application performs **inference only** and does not retrain the classifier when the application starts.

---

# 🚀 Running the Application Locally

## 1. Clone the Repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

If a `requirements.txt` file is not being used, the Streamlit application itself requires:

```bash
python -m pip install streamlit scikit-learn joblib
```

## 4. Run Streamlit

Navigate to the folder containing `app.py`:

```bash
cd SentimentApp
```

Then run:

```bash
python -m streamlit run app.py
```

Streamlit will normally open the application at:

```text
http://localhost:8501
```

---

# 📁 Recommended Repository Structure

```text
IMDb-Sentiment-Intelligence/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── IMDb_Sentiment_Analysis.ipynb
│
├── app/
│   ├── app.py
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
└── assets/
    └── streamlit_preview.png
```

Keeping notebooks, deployment files, and visual assets separated makes the repository easier to navigate and more professional for GitHub presentation.

---

# 🛠️ Technology Stack

### Programming & Data Analysis

- Python
- NumPy
- Pandas

### Visualization

- Matplotlib

### Machine Learning

- scikit-learn
- TF-IDF
- Logistic Regression

### NLP & Embeddings

- Keras Tokenizer
- Gensim
- Word2Vec

### Deep Learning

- TensorFlow
- Keras
- Simple RNN
- LSTM
- Bidirectional LSTM
- GRU

### Transformers

- Hugging Face Transformers
- DistilBERT

### Deployment

- Streamlit
- Joblib

---

# 📈 Evaluation Metrics

Models were evaluated using:

- **Accuracy** — overall proportion of correct predictions.
- **Precision** — reliability of positive predictions.
- **Recall** — proportion of actual positive reviews correctly identified.
- **F1-score** — harmonic balance between precision and recall.
- **Classification Report** — class-level evaluation.
- **Confusion Matrix** — distribution of correct and incorrect predictions.
- **Validation Loss** — used for neural-model checkpoint selection and overfitting analysis.

Using multiple metrics prevents model quality from being judged using accuracy alone.

---

# 🧪 Experimental Principles

Several methodological principles were maintained throughout the project:

### No Test-Based Hyperparameter Selection

Hyperparameter decisions were based on development training and validation data rather than selecting configurations according to test performance.

### Controlled Model Comparisons

Where possible, recurrent models used consistent training splits and comparable architectural settings.

### Early Stopping

Neural models restored the checkpoint corresponding to the best validation loss to reduce overfitting.

### Test-Set Transparency

Models evaluated on different numbers of examples are explicitly identified rather than being treated as directly equivalent.

### Conservative Text Cleaning

Preprocessing preserves linguistically meaningful information instead of applying aggressive transformations automatically.

---

# 💡 Key Findings

1. **Classical NLP remains extremely competitive.**  
   Optimized TF-IDF + Logistic Regression outperformed all recurrent models on the complete official test set.

2. **GRU was the strongest primary recurrent model.**  
   It achieved **86.92% accuracy** and **87.20% F1-score**.

3. **Larger models or input spaces are not automatically better.**  
   Vocabulary and sequence-length experiments showed non-monotonic validation behavior.

4. **Regularization matters.**  
   Dropout and early stopping helped control recurrent-model overfitting.

5. **Validation improvement does not guarantee test improvement.**  
   The tuned GRU improved validation behavior during selection but did not surpass the original GRU on final overall test performance.

6. **Error analysis reveals limitations hidden by aggregate metrics.**  
   Negation, mixed sentiment, long-range context, target ambiguity, and possible annotation noise all contributed to difficult predictions.

7. **Transformers provide powerful contextual modeling.**  
   DistilBERT produced strong results on the evaluated sample, although a full-test evaluation would be required for a fair direct comparison.

---

# 🔮 Future Improvements

Potential extensions include:

- Full 25,000-review Transformer evaluation
- Fine-tuning BERT or DistilBERT directly on IMDb
- Attention-based recurrent architectures
- Aspect-Based Sentiment Analysis
- Explicit negation modeling
- Explainable AI using SHAP or LIME
- Model calibration analysis
- Automated experiment tracking
- Cloud deployment of the Streamlit application
- REST API deployment
- Docker containerization
- CI/CD integration

---

# ⚠️ Limitations

The project has several important limitations:

- The task is binary sentiment classification and does not model neutral sentiment.
- Some reviews contain mixed or aspect-specific opinions that cannot be represented perfectly by one global label.
- Dataset labels may contain occasional ambiguity or noise.
- The pretrained Transformer was evaluated on a smaller sample rather than the complete official test set.
- Confidence scores should not be interpreted as guaranteed probabilities of correctness.
- Results are specific to the IMDb movie-review domain and may not generalize directly to other types of text.

---

# 📚 Project Workflow Summary

```text
Data Loading
     ↓
Exploratory Inspection
     ↓
Text Cleaning
     ↓
Train / Validation Strategy
     ↓
TF-IDF Baseline
     ↓
TF-IDF Optimization
     ↓
Word2Vec
     ↓
Simple RNN
     ↓
Dropout Experiment
     ↓
LSTM
     ↓
Bidirectional LSTM
     ↓
GRU
     ↓
Vocabulary & Sequence-Length Tuning
     ↓
Pretrained Transformer
     ↓
Model Comparison
     ↓
Qualitative Error Analysis
     ↓
Streamlit Deployment
```

---

# 👩‍💻 Author

**Rowan Ali**  
Data Science Student — Alexandria University

Areas of interest:

`Data Science` · `Machine Learning` · `Natural Language Processing` · `Deep Learning` · `Statistical Analysis`

---

## ⭐ Project Summary

**IMDb Sentiment Intelligence** demonstrates a complete NLP lifecycle—from raw text and classical feature engineering to recurrent deep learning, contextual Transformers, controlled experimentation, qualitative error analysis, and real-time deployment.

A key result of the project is that model complexity does not automatically translate into better generalization. The optimized **TF-IDF + Logistic Regression** pipeline achieved **89.58% accuracy and 89.62% F1-score** on the complete 25,000-review official test set, outperforming the recurrent architectures while remaining efficient enough for lightweight real-time deployment.

The project therefore combines not only model development, but also **experimental discipline, critical evaluation, error interpretation, and production-oriented deployment**.

---

<p align="center">
  <b>🎬 IMDb Sentiment Intelligence</b><br>
  From text preprocessing to real-time NLP inference.
</p>

---

## 🔗 Quick Links

- **Live Application:** [IMDb Sentiment Intelligence](https://imdb-sentiment-intelligence-jhwc5r5xlpzby5tpmttdwa.streamlit.app/)
- **Video Demo:** [Full Project Demonstration](demo/IMDb_Sentiment_Intelligence_Demo.mp4)
- **Notebook:** [Complete NLP Analysis](notebooks/IMDb_Sentiment_Analysis.ipynb)

<p align="center">
  <strong>🎬 IMDb Sentiment Intelligence</strong><br>
  From raw movie reviews to real-time sentiment predictions.
</p>
