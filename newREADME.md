
## 🧐 Deep Learning Assignment 2

*Course:* Deep Learning - 22CAC04  
*Institution:* Chaitanya Bharathi Institute of Technology  
*Department:* Information Technology  
*Due Date:* 20-04-25

### 🔍 Overview

This repository contains the implementation of *Question 1* and *Question 2* of the Deep Learning Assignment 2.

---

## 📌 Question 1: Latin-to-Devanagari Transliteration

### 🚀 Objective

To build a flexible RNN-based seq2seq architecture for transliterating Latin script inputs to their corresponding Devanagari script representations. The model supports multiple cell types: *SimpleRNN*, *GRU*, and *LSTM*, with tunable hyperparameters.

---

### 🗂 Dataset

Dataset used: [Dakshina Dataset (Google)](https://github.com/google-research-datasets/dakshina)  
Files used:

- `hi.translit.sampled.train.tsv`
- `hi.translit.sampled.dev.tsv`
- `hi.translit.sampled.test.tsv`

Each file contains the following columns:

- Devanagari script
- Latin transliteration
- Frequency count

---

### 🧱 Model Architecture

1. Embedding layers for both encoder and decoder
2. Encoder RNN (LSTM / GRU / SimpleRNN) to process Latin inputs
3. Decoder RNN (LSTM / GRU / SimpleRNN) to generate Devanagari characters
4. Dense layer with softmax for prediction

**Tunable Parameters:**

- Embedding dimension
- Hidden size
- RNN cell type (`rnn`, `gru`, `lstm`)
- Number of RNN layers

---

### 🧮 Theoretical Analysis

#### a) Total Number of Computations

Let:  
- *m* = embedding dimension  
- *k* = hidden size  
- *T* = sequence length  
- *V* = vocabulary size  

**Encoder Computation:** O(T × (m×k + k²))  
**Decoder Computation:** O(T × (m×k + k² + k×V))

#### b) Total Number of Parameters

- Encoder LSTM: 4 × (k × (k + m + 1))  
- Decoder LSTM: 4 × (k × (k + m + 1))  
- Output Dense: k × V  
- Embedding Layers: V × m (for both encoder and decoder)

---

### 📊 Training Details

- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Batch Size:** 64  
- **Epochs:** 30  
- **Validation Accuracy:** ~94.6%  
- **Test Accuracy:** *0.9457*

---

### 📈 Sample Predictions

Below are examples demonstrating the model’s transliteration capability from Latin input to Devanagari output:

- **Input:** `a n k`  
  **Target:** अं क  
  **Predicted:** ऐंक

- **Input:** `a n k a`  
  **Target:** अं क अ  
  **Predicted:** अंका

- **Input:** `a n k i t`  
  **Target:** अं कि त  
  **Predicted:** अंकित

- **Input:** `a n a k o n`  
  **Target:** अ न क ो न  
  **Predicted:** अनकों

- **Input:** `a n k h o n`  
  **Target:** अं ख ो ं  
  **Predicted:** अंखों

---

### 🧲 Evaluation

```bash
Test Accuracy: 0.9457
```

---

### 🛠 How to Run

#### 🔧 Install Requirements

```bash
pip install tensorflow==2.12.0 pandas gdown
```

#### ▶ Run Training

Ensure the .tsv files from the Dakshina dataset are placed in your working directory.

```bash
python main_seq2seq_transliteration.py
```

---

### 📂 File Structure

```
.
├── README.md
├── main_seq2seq_transliteration.py
├── hi.translit.sampled.train.tsv
├── hi.translit.sampled.dev.tsv
└── hi.translit.sampled.test.tsv
```

---

📘 **References**

- [Keras LSTM Seq2Seq Example](https://keras.io/examples/nlp/lstm_seq2seq/)
- [Machine Learning Mastery - Seq2Seq](https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/)
- [Dakshina Dataset](https://github.com/google-research-datasets/dakshina)

---

# 🎤 Fine-Tuned GPT-2 on Poetry Dataset

This project fine-tunes the GPT-2 language model on a poetry dataset to generate creative and lyrical text. It uses TensorFlow and Hugging Face Transformers to train the model and generate diverse lyrics from a given prompt.

## 📂 Dataset

Dataset used: [Poetry Dataset by Paul Mooney](https://www.kaggle.com/datasets/paultimothymooney/poetry)  
For this project, only the `bieber.txt` file was utilized.

## 🧠 Model

The model is based on Hugging Face's `gpt2` and was trained using TensorFlow.

**Configuration:**

- Base: GPT-2 (`TFGPT2LMHeadModel`)
- Tokenizer: GPT-2 tokenizer with EOS token padding
- Max input length: 128 tokens
- Training epochs: 3
- Batch size: 4

## 🚀 Setup Instructions

1. Clone this repo and open the notebook in Colab.
2. Install the necessary libraries:

```bash
pip install kagglehub datasets transformers
```

3. Authenticate with Kaggle to access the dataset.
4. Run the notebook to preprocess, train, and generate outputs.

## 🏋️‍♀️ Training Log

```
Epoch 1/3 - loss: 0.2775  
Epoch 2/3 - loss: 0.1929  
Epoch 3/3 - loss: 0.1515  
```

## 🎶 Sample Generations

Given this prompt:

```text
In the moonlight, I feel  
My heart begins to heal  
The stars are shining bright  
I'm floating in the night  
```

The model outputs:

- 🎶 *Version 1:*  
  In the moonlight, I feel  
  My heart begins to heal  
  The stars are shining bright  
  I'm floating in the night  
  It's all about the magic  

- 🎶 *Version 2:*  
  In the moonlight, I feel  
  My heart begins to heal  
  The stars are shining bright  
  I'm floating in the night  
  So take my hand

## 📁 Output

Model saved at:

```
./fine_tuned_gpt2_poetry/
```

## 🔮 Future Enhancements

- Train on a larger or more stylistically diverse poetry corpus
- Add rhyme/meter-based constraints for improved poetic structure
- Deploy as an interactive app or web-based lyric assistant

---

## 👤 Author

**Namani Pravardhan**
