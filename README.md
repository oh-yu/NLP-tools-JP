# Requirements
pwd 
=> hoge/NLP-tools-JP
wget http://public.shiroyagi.s3.amazonaws.com/latest-ja-word2vec-gensim-model.zip
unzip ./latest-ja-word2vec-gensim-model.zip

# Similarity Between Two Sentences
## 1. Gzip Compression Dist
https://gist.github.com/kyo-takano/fa2b42fb4df20e2566c29c31f20f87ed  
https://aclanthology.org/2023.findings-acl.426/
## 2. Word Mover's Dist
from utils import word_movers_dist  
word_movers_dist(sentence1="hoge", sentence2="fuga")  
https://proceedings.mlr.press/v37/kusnerb15.html
## 3. Word Rotator's Dist
from utils import word_rotators_dist  
word_rotators_dist(sentence1="hoge", sentence2="fuga")  
https://arxiv.org/abs/2004.15003
