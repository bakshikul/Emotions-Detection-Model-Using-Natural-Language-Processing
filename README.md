# 😊 Emotions-Detection-Model-Using-Natural-Language-Processing 

An **NLP-based Emotion Detection System** that analyzes text and predicts the underlying emotion expressed by the user. The project uses **Natural Language Processing (NLP)** techniques and a trained machine learning model to classify text into different emotional categories.

The project also includes an interactive **Streamlit web application** where users can enter text and receive the predicted emotion through a simple and user-friendly interface.

---

## 📌 Project Overview

Understanding human emotions from text is an important application of **Natural Language Processing** and **Machine Learning**.

This project takes a textual input such as:

> *"I am extremely happy with my results!"*

and processes it using NLP techniques before predicting the corresponding emotion.

The complete pipeline includes:

**Text Input → Text Preprocessing → Feature Extraction → Trained ML Model → Emotion Prediction → Streamlit Interface**

---

## 🎯 Objectives

* Build an NLP-based emotion classification system.
* Preprocess and clean textual data.
* Convert text into numerical features suitable for machine learning.
* Train a machine learning model for emotion classification.
* Save and reuse the trained model.
* Create an interactive Streamlit application.
* Provide real-time emotion predictions from user input.

---

## ✨ Features

* 📝 Text-based emotion detection
* 🧹 NLP text preprocessing
* 🤖 Machine Learning classification
* 📊 Text feature extraction
* ⚡ Real-time prediction
* 🖥️ Interactive Streamlit interface
* 🔄 Reusable trained model
* 📱 Simple and user-friendly UI

---

## 🧠 Emotions Detected

The model can classify text into multiple emotion categories depending on the classes present in the training dataset.

Typical examples include:

* 😊 Joy
* 😢 Sadness
* 😡 Anger
* 😨 Fear
* 😲 Surprise
* ❤️ Love

> The exact emotion classes depend on the dataset used for training the model.

---

## 🏗️ Project Architecture

```text
                    ┌───────────────────┐
                    │    User Input     │
                    │      (Text)       │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Text Preprocessing│
                    │ Cleaning / NLP    │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Feature Extraction│
                    │  Text → Features  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Trained ML Model  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Emotion Prediction│
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Streamlit Web App │
                    └───────────────────┘
```

---

## 📂 Project Structure

```text
Emotion-Detection-NLP/
│
├── app.py
├── emotionsNLP.ipynb
├── train.txt
├── NLP_Emotions_model/
│
├── requirements.txt
├── README.md
└── screenshots/
```

### File Description

| File / Folder         | Description                                       |
| --------------------- | ------------------------------------------------- |
| `app.py`              | Streamlit application for emotion prediction      |
| `emotionsNLP.ipynb`   | Data preprocessing, model training and evaluation |
| `train.txt`           | Training dataset                                  |
| `NLP_Emotions_model/` | Saved trained NLP/ML model and related artifacts  |
| `requirements.txt`    | Required Python libraries                         |
| `README.md`           | Project documentation                             |

---

## 🔄 Machine Learning Workflow

### 1. Data Collection

The model is trained using a text dataset containing sentences associated with different emotion labels.

Example:

```text
I am feeling amazing today → Joy
I am really disappointed → Sadness
This makes me extremely angry → Anger
I am scared of what might happen → Fear
```

### 2. Text Preprocessing

The input text is cleaned before being passed to the model.

Typical preprocessing operations include:

* Converting text to lowercase
* Removing unnecessary characters
* Removing unwanted spaces
* Tokenization
* Removing unnecessary words
* Preparing text for feature extraction

### 3. Feature Extraction

Machine learning algorithms cannot directly understand raw text.

Therefore, the text is transformed into numerical features using the feature extraction technique implemented in the notebook.

```text
Raw Text
   ↓
Preprocessing
   ↓
Feature Extraction
   ↓
Numerical Representation
```

### 4. Model Training

The processed dataset is used to train a machine learning classification model.

The model learns relationships between textual patterns and their corresponding emotion labels.

### 5. Prediction

When the user enters new text:

```text
User Text
   ↓
Preprocessing
   ↓
Feature Transformation
   ↓
Trained Model
   ↓
Predicted Emotion
```

---

## 🛠️ Technologies Used

### Programming Language

* **Python**

### Libraries

* **Pandas** – Data manipulation and analysis
* **NumPy** – Numerical computations
* **Scikit-learn** – Machine learning and feature processing
* **NLTK** – Natural Language Processing
* **Joblib** – Saving and loading trained models
* **Streamlit** – Web application development

> The exact dependencies may vary depending on the implementation in the notebook.

---

## 💻 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Emotion-Detection-NLP.git
```

### 2. Navigate to the Project Directory

```bash
cd Emotion-Detection-NLP
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

## ▶️ Running the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

After running the command, Streamlit will provide a local URL where you can access the application.

Usually:

```text
http://localhost:8501
```

---

## 🧪 Example Predictions

### Example 1 — Joy

```text
I am extremely happy today because I achieved my goal!
```

**Predicted Emotion:** 😊 Joy

### Example 2 — Sadness

```text
I feel very lonely and disappointed after losing someone close to me.
```

**Predicted Emotion:** 😢 Sadness

### Example 3 — Anger

```text
I am extremely angry because they treated me unfairly.
```

**Predicted Emotion:** 😡 Anger

---

## 📊 Model Pipeline

The overall prediction pipeline can be represented as:

```text
                Input Sentence
                       │
                       ▼
              Text Preprocessing
                       │
                       ▼
              Feature Extraction
                       │
                       ▼
                 ML Classifier
                       │
                       ▼
               Emotion Label
                       │
                       ▼
             Streamlit Interface
```

---

## 🖥️ Application

The Streamlit application provides an interactive interface where users can:

1. Enter a sentence or paragraph.
2. Submit the text for analysis.
3. Process the text through the trained NLP pipeline.
4. Receive the predicted emotion.

This makes the trained machine learning model accessible without requiring users to interact directly with the Python notebook.

---

## 📈 Possible Applications

Emotion detection from text can be used in various real-world applications, including:

* 💬 Customer feedback analysis
* 📱 Social media sentiment monitoring
* 🤖 Chatbots and virtual assistants
* 📧 Email analysis
* 🎧 Customer support systems
* 📊 Product review analysis
* 🧠 Human-computer interaction
* 📢 Social media monitoring

---

## 🔮 Future Improvements

The project can be further improved by:

* Using larger and more diverse datasets.
* Improving text preprocessing.
* Comparing multiple machine learning algorithms.
* Using TF-IDF, Word2Vec, GloVe or transformer-based embeddings.
* Implementing deep learning models such as LSTM and GRU.
* Experimenting with BERT or other transformer architectures.
---

## 📚 Learning Outcomes

Through this project, I worked with:

* Natural Language Processing
* Text preprocessing
* Feature extraction
* Machine learning classification
* Model serialization using Joblib
* Model inference
* Streamlit application development
* End-to-end ML project deployment workflow

---

## 🚀 Future Vision

The goal is to evolve this project from a basic text classification system into a more advanced **Emotion Intelligence System** capable of understanding complex emotional patterns in real-world conversations.

Future versions could combine NLP, deep learning and transformer-based architectures to provide more accurate and context-aware emotion predictions.

---

## 👨‍💻 Author

**Kul Bakshi**

Student | AI/ML Enthusiast

---

## ⭐ Support

If you found this project useful or interesting, consider giving the repository a ⭐ on GitHub!

Feel free to fork the project, experiment with the model, and improve the implementation.
