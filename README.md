# Image-Caption-Generator

An image captioning system that generates natural-language descriptions for images. Built using a VGG16 CNN encoder, an LSTM decoder, and an attention mechanism, trained on the Flickr8k dataset.

<img width="600" height="500" alt="sample-image" src="https://github.com/user-attachments/assets/5539dcbe-df19-48b8-947e-f2289b4de4a5" />


## Overview

- Attention mechanism implemented from scratch, so the model learns where to look in the image while generating each word.
- Attention heatmaps showing which region of the image the model focused on for each word.
- Trained on 8,091 images from Flickr8k, evaluated using BLEU-1 to BLEU-4.
- Trained in two stages, 20 epochs total.

## Dataset

[Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) — 8,000+ images, each with 5 human-written captions, downloaded via Kaggle.

## Project Workflow

### Data Loading
Load `captions.txt` into a DataFrame and group all 5 captions per image into a dictionary: `{image_id: [captions]}`.

### Caption Cleaning
Each caption is:
- Lowercased
- Stripped of punctuation and numbers
- Extra whitespace removed
- Wrapped with `startseq` and `endseq` tokens, so the model learns when a caption starts and ends

### Tokenization
A Keras `Tokenizer` is fit on all cleaned captions to build the vocabulary. The vocabulary size and max caption length are computed and the tokenizer is saved (`tokenizer.pkl`) for later use during inference.

### Train/Test Split
Images are split into 85% training and 15% test sets.

### Feature Extraction (CNN Encoder)
A pretrained VGG16 (ImageNet weights) is used, but instead of using the final classification layer, features are taken from the `block5_conv3` layer. This gives a 14x14 spatial grid (196 regions x 512 channels) instead of a single flattened vector — this spatial detail is required for the attention mechanism. Every image is passed through VGG16 once and the resulting feature map is cached as a `.npy` file, so the CNN doesn't need to be re-run on every training epoch.

### Model Architecture (Attention + LSTM Decoder)
- **Image branch**: the cached (196, 512) feature map is projected to a lower dimension.
- **Caption branch**: the partial caption so far is embedded and passed through an LSTM to get the current decoder state.
- **Attention**: combines the image features and current LSTM state to compute a weight over the 196 image regions, then produces a weighted context vector — this is what lets the model "look" at relevant parts of the image.
- **Decoder head**: the context vector and LSTM state are combined and passed through a dense layer to predict the next word.

 ### Training Data Generator
A generator streams `(image_features, partial_caption) -> next_word` training pairs on the fly, instead of loading the entire dataset into memory.

### Training (Two Stages)
The model is trained for 10 epochs, then for 10 more epochs (20 total), with checkpoints saved after every epoch.

### Caption Generation (Inference)
Captions are generated word-by-word using greedy decoding: starting from `startseq`, the model repeatedly predicts the most likely next word until `endseq` or the max length is reached.

### Evaluation (BLEU Score)
Captions are generated for the full test set and compared against reference captions using BLEU-1 to BLEU-4.

### Attention Visualization
For a sample image, the attention weights at each decoding step are extracted and plotted as a heatmap over the image, showing which region the model focused on for each generated word.

## Results

Dataset: 8,091 images, vocabulary size 8,781 words, max caption length 37 tokens.

| Metric | Score |
|--------|-------|
| BLEU-1 | 0.5197 |
| BLEU-2 | 0.3214 |
| BLEU-3 | 0.1865 |
| BLEU-4 | 0.1051 |

### Training curves

<img width="1200" height="750" alt="training_curves" src="https://github.com/user-attachments/assets/57645194-d176-4901-bec0-87c6bab01c1c" />


### Attention visualization

<img width="1800" height="1200" alt="attention_visualization" src="https://github.com/user-attachments/assets/9a2578cb-ac47-4c6e-8081-84bc0cf245bf" />


### Model architecture

<img width="500" height="773" alt="model_architecture" src="https://github.com/user-attachments/assets/fe6bea68-f41c-4e67-9389-0666ec6a2c58" />


## Repository Structure

```
.
├── notebooks/
│   └── Image-Captioning-final.ipynb
├── assets/
│   ├── training_curves.png
│   ├── attention_visualization.png
│   ├── model_architecture.png
│   └── sample_output.png
├── caption_model.h5
├── tokenizer.pkl
└── README.md
```

## Tech Stack

Python, TensorFlow/Keras, NumPy, Pandas, NLTK, Matplotlib, scikit-learn


By [Abhishek Sharma](https://github.com/abhisheksharma28703)
