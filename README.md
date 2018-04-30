# Introduction

This project implements a CRF-based named entity recognizer. It uses the CRFSuite library as its basis. It’s able to handle 13 kinds of labels for PER, LOC and ORG. The implementation achieved the best F1 score of 0.81 on test set (trained on 2,000 sentences). 

# Requirements
1. python 3.x
2. CRFsuite 0.12 (https://github.com/chokkan/crfsuite)
3. LibLBFGS (must have for crfsuite)
4. scikit-learn (for the word2vec clustering, optional)
5. gensim (for the LDA, optional)
6. nltk (used for stopword list)

# Used features

The CRF named-entity recognizer uses the following features:
 
Unigram Features - window: wi-2, wi-1, wi, wi+1
1. The word itself.
2. The Part of Speech tag of the word.
3. Whether the word starts with an uppercase letter.
4. The brown cluster of the word.
5. The brown cluster prefix of the word representing in 0s and 1s.  (4, 8, 10, 12 bits).
6. Whether the word exists in a Gazetteer of person names.
7. Whether the word exists in a Gazetteer of places names.
8. The pattern of the word, i.e: John -> UL+ (stands for uppercase-lowercase-lowercase-lowercase)
9. The 1-2-3-character-prefix of the word, i.e. John -> J, Jo, Joh
10. The 1-2-3-character-suffix of the word, i.e. John -> n, hn, ohn
11. Whether the word contains the following combination of characters: digit + “.”, digit + “,”
12. Whether the word exists in the top 1000 most frequent word list
13. kmeans word2vec cluster

Bigram Features - window: [wi-2, wi-1], [wi-1, wi], [wi, wi+1] 
1. The word itself.
2. The Part of Speech tag of the word.
3. Whether the word starts with an uppercase letter.

# Usage

Use the following command to extract features from the data file: 
    
    $ python feature_extractor.py [INPUT] [OUTPUT]

For example, process a original data file train.gold and output it to train_feature.txt
    
    $ python feature_extractor.py train.gold train_feature.txt

Once you've obtained the feature files, you can just follow the training and tagging steps in CRFsuite.
