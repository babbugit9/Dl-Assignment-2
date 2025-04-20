## 🧐 Deep Learning Assignment 2

*Course:* Deep Learning - 22CAC04\
*Institution:* Chaitanya Bharathi Institute of Technology\
*Department:* Information Technology\
*Due Date:* 20-04-25

### 🔍 Overview

This repository contains the implementation of *Question 1* and *Question 2* of the Deep Learning Assignment 2.

---

## 📌 Question 1: Latin-to-Devanagari Transliteration

### 🚀 Objective

To build a flexible RNN-based seq2seq architecture for transliterating Latin script inputs to their corresponding Devanagari script representations. The model supports multiple cell types: *SimpleRNN, **GRU, and **LSTM*, with tunable hyperparameters.

---

### 🗂 Dataset

Dataset used: [Dakshina Dataset (Google)](https://github.com/google-research-datasets/dakshina)\
Files used:

- hi.translit.sampled.train.tsv
- hi.translit.sampled.dev.tsv
- hi.translit.sampled.test.tsv

Each file contains columns:

- Devanagari script
- Latin transliteration
- Frequency count

---

### 🧱 Model Architecture

1. *Embedding Layer* for both encoder and decoder
2. *Encoder RNN (LSTM / GRU / SimpleRNN)* - processes the Latin script input
3. *Decoder RNN (LSTM / GRU / SimpleRNN)* - generates the Devanagari script character-by-character using the final encoder state
4. *Dense Layer* with softmax activation for character prediction

*Flexibility:*

- Embedding Dimension
- Hidden Units
- RNN Cell Type ('lstm', 'gru', 'rnn')
- Number of Layers (extendable in the function)

---

### 🧮 Theoretical Analysis

#### a) Total Number of Computations

Let:

- m = embedding dimension
- k = hidden size
- T = sequence length
- V = vocabulary size

Total computations (approx):\
Encoder: O(T × (m×k + k²))\
Decoder: O(T × (m×k + k² + k×V))

#### b) Total Number of Parameters

Encoder LSTM: 4 × (k×(k + m + 1))\
Decoder LSTM: 4 × (k×(k + m + 1))\
Dense Output: k × V\
Embedding Layers: V × m (each for encoder and decoder)

---

### 📊 Training Details

- *Optimizer:* Adam
- *Loss:* Categorical Crossentropy
- *Batch Size:* 64
- *Epochs:* 30
- *Validation Accuracy:* \~94.6%
- *Test Accuracy:* *0.9457*

---

### 📈 Sample Predictions

| Input (Latin) | Target (Devanagari) | Predicted |
| ------------- | ------------------- | --------- |
| a n k         | अ ं क               | ऐंक       |
| a n k a       | अ ं क               | अंका      |
| a n k i t     | अ ं क ि त           | अंकित     |
| a n a k o n   | अ ं क ो ं           | अनकों     |
| a n k h o n   | अ ं क ो ं           | अंखों     |

---

### 🧲 Evaluation

bash
Test Accuracy: 0.9457


---

### 🛠 How to Run

#### 🔧 Install Requirements

bash
pip install tensorflow==2.12.0 pandas gdown


#### ▶ Run Training

Ensure the .tsv files from Dakshina dataset are in your working directory.

python
python main_seq2seq_transliteration.py


---

### 📂 File Structure


.
├── README.md
├── main_seq2seq_transliteration.py  # All code for Q1
├── hi.translit.sampled.train.tsv
├── hi.translit.sampled.dev.tsv
└── hi.translit.sampled.test.tsv


---

📘 References

- [Keras LSTM Seq2Seq Example](https://keras.io/examples/nlp/lstm_seq2seq/)
- [Machine Learning Mastery - Seq2Seq](https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/)
- [Dakshina Dataset](https://github.com/google-research-datasets/dakshina)

---

# 🎤 Fine-Tuned GPT-2 on Poetry Dataset

This project fine-tunes the GPT-2 language model on a poetry dataset to generate creative and lyrical text. It uses TensorFlow and Hugging Face Transformers to train the model and generate diverse versions of lyrics based on a given prompt.

## 📂 Dataset

The dataset used is the [Poetry Dataset by Paul Mooney](https://www.kaggle.com/datasets/paultimothymooney/poetry), accessed via KaggleHub. This dataset contains collections of poems from various artists.

Only the `bieber.txt` file was used for this fine-tuning demo.

## 🧠 Model

We used the `gpt2` model from Hugging Face's `transformers` library and fine-tuned it using TensorFlow on the selected poetry dataset.

**Model Details:**

- Base model: GPT-2 (`TFGPT2LMHeadModel`)
- Tokenizer: GPT-2 tokenizer with EOS token used as padding
- Input length: 128 tokens
- Training epochs: 3
- Batch size: 4

## 🚀 Setup Instructions

1. Clone this repository and open the notebook in Google Colab.

2. Install required libraries:

   ```bash
   pip install kagglehub datasets transformers
   ```

3. Authenticate with Kaggle to access the dataset.

4. Run the notebook cells to:

   - Download the dataset
   - Preprocess and tokenize the text
   - Fine-tune the model
   - Save and generate from the model

## 🏋️‍♀️ Training

The model was trained over 3 epochs:

```
Epoch 1/3 - loss: 0.2775  
Epoch 2/3 - loss: 0.1929  
Epoch 3/3 - loss: 0.1515  
```

The model converges quickly due to the relatively small dataset.

## 🎶 Poetry Generation

After training, the model can generate creative lyrical continuations based on a prompt. For example, given:

```text
In the moonlight, I feel
My heart begins to heal
The stars are shining bright
I'm floating in the night
```

It generates multiple versions like:

```
🎶 Version 1:
In the moonlight, I feel
My heart begins to heal
The stars are shining bright
I'm floating in the night
It's all about the magic

🎶 Version 5:
In the moonlight, I feel
My heart begins to heal
The stars are shining bright
I'm floating in the night
So take my hand
```

## 📁 Output

The fine-tuned model and tokenizer are saved in the directory:

```
./fine_tuned_gpt2_poetry/
```

You can reload the model for inference or further training.

## 🛠️ Future Work

- Train on larger or more diverse poetry datasets
- Improve output structure using rhyme detection or meter control
- Convert this into an interactive web app or chatbot

---

## 👤 Author

**Namani Pravardhan**

