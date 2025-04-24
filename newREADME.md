
# 🧠 Deep Learning Assignment - CBIT IT Dept

**Course**: DL (22CAC04)  
**Institution**: CBIT, Hyderabad  
**Branch**: Information Technology  
**Assignment Due**: April 20, 2025

---

## 📘 Contents

This repository showcases two key components of Assignment 2:

- 🔤 **Q1**: Transliteration using Sequence-to-Sequence models  
- ✍️ **Q2**: Text generation by fine-tuning GPT-2 on poetry

---

## 🔡 Question 1 - Transliteration from Latin to Devanagari

### 🎯 Goal

Design and implement a configurable sequence-to-sequence model that transliterates Latin script inputs into their corresponding Devanagari outputs. The architecture should be adaptable to various RNN cells and configurations.

### 📁 Dataset - Dakshina by Google

Source: [Google Research - Dakshina](https://github.com/google-research-datasets/dakshina)  
Files used:

- `hi.translit.sampled.train.tsv`
- `hi.translit.sampled.dev.tsv`
- `hi.translit.sampled.test.tsv`

Each line contains:
- Hindi word in Devanagari
- Corresponding Latin script
- Frequency count

---

### 🏗️ Model Blueprint

- 🔠 Embedding layer (encoder & decoder)
- 🧠 Recurrent Cell (RNN / LSTM / GRU)
- 🔄 Encoder processes Latin characters
- 🗣️ Decoder generates Devanagari characters
- 🔚 Dense layer with softmax for output prediction

> 🛠️ The design supports dynamic changes for:
> - RNN type
> - Embedding dimensions
> - Hidden layer size
> - Vocabulary size

---

### 🔬 Computation & Parameters

#### (a) Total Computation

For:
- `m =` embedding size
- `k =` hidden dimension
- `T =` sequence length
- `V =` vocabulary size

Computation per time step:  
- Encoder ≈ T × (m×k + k²)  
- Decoder ≈ T × (m×k + k² + k×V)

#### (b) Total Parameters

- Encoder LSTM: 4 × (k × (k + m + 1))  
- Decoder LSTM: 4 × (k × (k + m + 1))  
- Output layer: k × V  
- Embeddings: V × m (encoder + decoder)

---

### 🧪 Performance

- **Optimizer**: Adam  
- **Loss**: Categorical Crossentropy  
- **Epochs**: 30  
- **Batch Size**: 64  
- **Test Accuracy**: `94.57%`

---

### ✨ Sample Output

| Latin Input | Expected Output | Model Prediction |
|-------------|-----------------|------------------|
| a n k       | अंक              | ऐंक              |
| a n k i t   | अंकित            | अंकित             |
| a n k a     | अंका             | अंका              |
| a n a k o n | अनकों            | अनकों             |
| a n k h o n | अंखों            | अंखों             |

---

### ▶️ Run Instructions

Install required libraries:

```bash
pip install tensorflow==2.12.0 pandas gdown
```

Run training script (with dataset in root):

```bash
python main_seq2seq_transliteration.py
```

---

## ✨ Question 2 - GPT-2 Fine-Tuning for Poetry Generation

### 📝 Summary

This task focuses on creatively generating poetic text by fine-tuning OpenAI's GPT-2 model on a curated poetry dataset using the Transformers library and TensorFlow backend.

### 📚 Dataset

Used: [`poetry` dataset by Paul Mooney (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/poetry)

- Only the `bieber.txt` file was selected for fine-tuning.

---

### 🧱 GPT-2 Configuration

- Model: `gpt2` via Hugging Face
- Tokenizer: GPT-2 with padding token
- Sequence length: 128 tokens
- Epochs: 3
- Batch Size: 4

---

### 🔄 Training Logs

```
Epoch 1 - Loss: 0.2775
Epoch 2 - Loss: 0.1929
Epoch 3 - Loss: 0.1515
```

---

### 🎤 Sample Generation

Prompt:

```text
In the moonlight, I feel
My heart begins to heal
The stars are shining bright
I'm floating in the night
```

Generated:

```
🌟 Version 1:
In the moonlight, I feel
My heart begins to heal
The stars are shining bright
I'm floating in the night
Whispers in the dark

🌟 Version 2:
In the moonlight, I feel
My heart begins to heal
The stars are shining bright
I'm floating in the night
Calling out your name
```

---

### 📦 Output Directory

Trained artifacts are saved in:

```
./fine_tuned_gpt2_poetry/
```

---

## 🔧 Setup Guide

Install requirements:

```bash
pip install kagglehub datasets transformers
```

Follow notebook to:
- Load dataset
- Preprocess
- Fine-tune GPT-2
- Generate poetic output

---

## 👨‍💻 Author

**Namani Pravardhan**  
CBIT | Dept. of IT  


---
