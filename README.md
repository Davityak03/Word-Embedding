# Word-Embedding

## Overview

This Jupyter Notebook, `Word_Embedding.ipynb`, demonstrates the use of word embeddings in natural language processing (NLP). Word embeddings are vector representations of words that capture their meanings, semantic relationships, and syntactic patterns. This notebook walks through how to create and use word embeddings using popular libraries and techniques.

## Features

- **Introduction to Word Embeddings**: A brief explanation of what word embeddings are and why they are important in NLP tasks.
- **Preprocessing Text Data**: Demonstrates how to clean and prepare text data for word embedding models.
- **Word2Vec**: Implements the Word2Vec algorithm using `gensim` to create word embeddings from a text corpus.
- **Visualizing Word Embeddings**: Uses dimensionality reduction techniques (like PCA or t-SNE) to visualize word vectors.
- **Evaluating Word Embeddings**: Discusses methods to evaluate the quality of generated embeddings.
  
## Technologies Used

- **Python**: The core programming language used in this notebook.
- **gensim**: A Python library used for unsupervised topic modeling and natural language processing, specifically for training Word2Vec models.
- **NLTK**: For text preprocessing and tokenization tasks.
- **Matplotlib & Seaborn**: Visualization libraries used to plot word embeddings.
- **sklearn**: For dimensionality reduction techniques like PCA and t-SNE.

## Setup Instructions

### Requirements
Before running this notebook, ensure you have the following libraries installed:

- Python 3.x
- `gensim`
- `nltk`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install the required packages using the following command:

```bash
pip install gensim nltk matplotlib seaborn scikit-learn
```

### Running the Notebook

1. **Launch Jupyter Notebook**:
   Open a terminal and navigate to the directory where the `Word_Embedding.ipynb` file is located. Launch the Jupyter Notebook by typing:

   ```bash
   jupyter notebook
   ```

2. **Load the Notebook**: In the Jupyter interface, open the `Word_Embedding.ipynb` file.

3. **Execute the Cells**: Run the cells sequentially to preprocess the text data, train the word embedding model, and visualize the embeddings.

## Usage

- **Training Word Embeddings**: The notebook trains a Word2Vec model on a text corpus and outputs word vectors for each word.
- **Visualizing Embeddings**: A section of the notebook uses PCA or t-SNE to reduce the dimensionality of the word vectors and visualize them in 2D space.
- **Similarity Queries**: The trained embeddings can be used to find similar words, explore word relationships, and perform analogies (e.g., "king" - "man" + "woman" = "queen").

## Example

Here is a simple example of using word embeddings trained in this notebook:

```python
# Find the most similar words to "king"
model.wv.most_similar('king')
```

### Sample Output:

```
[('queen', 0.85), ('prince', 0.82), ('monarch', 0.81), ...]
```

