# SignLLM: LLMs are Good Sign Language Translators

Implementation of the CVPR 2024 paper "LLMs are Good Sign Language Translators" for Kaggle.

## Features
- VQ-Sign module for character-level tokenization
- CRA module for word-level reconstruction
- Frozen LLM integration
- Phoenix-2014T dataset support
- Kaggle-compatible lightweight implementation

## Setup on Kaggle

```python
# Clone repository
!git clone https://github.com/yourusername/signllm.git
%cd signllm

# Install dependencies
!pip install -r requirements.txt

# Download NLTK for evaluation
import nltk
nltk.download('punkt')