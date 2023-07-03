# A program for sentiment analysis of Japanese documents, and their English translations.
This is E2J_sent, a work-in-progress program which uses a DistilBERT model to analyze the sentiment valence of Japanese documents against English translations.
This is my first in-depth progress working with PyTorch and Huggingface models, for the purpose of comparing sentiment analysis between English and Japanese pieces of text.
<br><br>
Currently, this project is a work-in-progress. I've completed the backend and included some files which you can test out, either in Jupyter or through command line. At the moment, I am working on a frontend for this project in JavaScript. 

## Tools used:
- Huggingface
- PyTorch
- Python
- Pandas

## Dependencies: 
- pdfminer.six
- Stanza
- PyTorch and dependencies
- Pandas

## Usage: 
```python3 e2j_sent.py [-h] in_JP in_EN```
<br>
### Args:
in_JP: Japanese document's path 
<br>
in_EN: English document's path
### Outputs:
out.csv: Output of the model's sentiment analysis, broken up line-by-line
<br>
[in_JP].txt: UTF-8 encoded version of the input document's text (in Japanese)
<br>
[in_EN].txt: UTF-8 encoded version of the input document's text (in English)
<br>
[in_JP]_tokenized: UTF_8 encoded version of the input document's text after tokenization (in Japanese)
<br>
[in_EN]_tokenized: UTF_8 encoded version of the input document's text after tokenization (in English)
