## Project Overview

This workspace contains two main notebooks demonstrating modern NLP techniques:

- [SastaGPT.ipynb](SastaGPT.ipynb): Implements a Transformer-based language model from scratch, inspired by the "Attention Is All You Need" paper. It covers model architecture, training, and text generation.
- [Embeddings.ipynb](Embeddings.ipynb): Explores word and sentence embeddings, and applies them to a real-world sentence classification task using disaster-related tweets.

## Directory Structure

```
disaster_tweets.csv         # Dataset of tweets labeled for disaster relevance
Embeddings.ipynb           # Notebook for embeddings and sentence classification
embeddings.jpeg            # Illustration for embeddings concepts
SastaGPT.ipynb             # Notebook for transformer language modeling
transformers.png           # Illustration for transformer architecture
```

## SastaGPT Notebook

- **Data Preparation:** Loads and cleans a movie script for training.
- **Tokenizer:** Uses GPT-2 tokenization via `tiktoken`.
- **Model Architecture:** Defines a configurable Transformer with multi-head self-attention, feedforward layers, and positional encoding.
- **Training:** Trains the model on the dataset using PyTorch, with progress tracking and loss reporting.
- **Text Generation:** Demonstrates text generation using the trained model.
- **Parameter Analysis:** Visualizes how hyperparameters affect model size.

### Usage

1. Open [SastaGPT.ipynb](SastaGPT.ipynb) in VS Code or Jupyter.
2. Run the cells sequentially to build, train, and test the model.
3. Adjust hyperparameters in the `Config` dataclass as needed.

## Embeddings Notebook

- **Word Embeddings:** Loads pretrained Word2Vec embeddings, demonstrates vector operations and similarity.
- **Sentence Embeddings:** Uses `gpt4all.Embed4All` to generate sentence embeddings.
- **Data Cleaning:** Cleans tweet text for classification.
- **Classification:** Trains an SVM classifier to predict whether tweets are about real disasters.
- **Evaluation:** Prints validation accuracy and allows prediction on custom sentences.

### Usage

1. Open [Embeddings.ipynb](Embeddings.ipynb).
2. Run the cells to explore embeddings and train the classifier.
3. Use the `predict` function to test sentences of your choice.

## Dataset

- [disaster_tweets.csv](disaster_tweets.csv): Contains tweets with a `text` column and a `target` column (1 for disaster-related, 0 for not).

## Requirements

- Python 3.10+
- PyTorch
- scikit-learn
- gensim
- nltk
- tiktoken
- gpt4all

Install dependencies using:

```sh
pip install torch scikit-learn gensim nltk tiktoken gpt4all
```

## Notes

- For GPU training, use Google Colab or Kaggle Notebooks.
- The notebooks are designed for educational purposes and can be extended for more advanced NLP tasks.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Sentence-BERT](https://arxiv.org/abs/1908.10084)
- [Word2Vec](https://arxiv.org/abs/1301.3781)
- [gpt4all Python Embedding Docs](https://docs.gpt4all.io/gpt4all_python_embedding.html#gpt4all.gpt4all.Embed4All.embed)

---
