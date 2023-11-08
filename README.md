# Similarity Between Two Sentences
## 0. Requirements
```
pwd  
=> hoge/NLP-tools-JP/sentences-similarity  
wget http://public.shiroyagi.s3.amazonaws.com/latest-ja-word2vec-gensim-model.zip  
unzip ./latest-ja-word2vec-gensim-model.zip
```
https://github.com/shiroyagicorp/japanese-word2vec-model-builder/blob/master/LICENSE
## [Gzip Compression Dist](https://aclanthology.org/2023.findings-acl.426/ )
https://gist.github.com/kyo-takano/fa2b42fb4df20e2566c29c31f20f87ed  

## [Word Mover's Dist](https://proceedings.mlr.press/v37/kusnerb15.html )
```
from utils import word_movers_dist  
word_movers_dist(sentence1="私の名前はhogeです。好きな食べ物はりんごで、嫌いな食べ物は野菜です。", sentence2="私の名前はfugaです。嫌いな食べ物はキャベツで、好きな食べ物は果物です。")
=> 0.9481324553489685
```

## [Word Rotator's Dist](https://arxiv.org/abs/2004.15003 )
```
from utils import word_rotators_dist  
word_rotators_dist(sentence1="私の名前はhogeです。好きな食べ物はりんごで、嫌いな食べ物は野菜です。", sentence2="私の名前はfugaです。嫌いな食べ物はキャベツで、好きな食べ物は果物です。")
=> 0.44947749376296997
```

