#!/usr/bin/env python

import re
import string
import json
import argparse
from nltk.corpus import stopwords

# list of features
#word,pos,summarized pattern, digit+'.', digits+',', is_in_freq_list
#startswith upper, word2vec cluster, 1-2-3 char prefix, suffix
#if in name_place names, brown cluster
unigram_feats = ['w', 'pos', 'pat', 'd&,', 'd&.', 'freq', 'su', 'wv',
                 'p2', 'p3', 'p1', 's2', 's3', 's1', 
                 'is_n', 'is_c',
                 'br_no', 'br4', 'br8', 'br10', 'br12', 
                 ]

bigram_feats = ['w', 'pos', 'su']
stopwords = stopwords.words('english')


# generate feature template
templates = []
for feat_name in unigram_feats:
    templates += [((feat_name, idx),) for idx in range(-2, 2)]

for feat_name in bigram_feats:
    templates += [((feat_name, i), (feat_name, i+1)) for i in range(-2, 2)]


# Read pretrained brown cluster
def get_brown():
    
    brown = {}
    brown_id = {}
    with open("./data/cmu_brown.txt", 'r') as input_file:
        corpus = input_file.readlines()
        
    for line in corpus:
        line = line.strip()
        if not line:
            continue
        fields = line.split('\t')
        brown[fields[1].lower()] = fields[0]
        brown_id[fields[1].lower()] = fields[-1]
        
    return brown, brown_id


# helper function for determining whether a word has the pattern "digit + symbol"
def is_digit_and_sym(token, p):
    bd = False
    bdd = False
    for c in token:
        if c.isdigit():
            bd = True
        elif c == p:
            bdd = True
        else:
            return False
    return bd and bdd


#load the pretrained word2vec cluster
def get_word2vec():
    
    word2vec = json.load(open('./data/word_to_cluster.json'))
        
    return word2vec


#map the word to a fixed set of character
def get_pattern(token):
    
    r = ''
    for c in token:
        if c.isupper():
            r += 'U'
        elif c.islower():
            r += 'L'
        elif c.isdigit():
            r += 'D'
        elif not c.isalnum():
            r += 'S'
            
    result = re.sub(r'([ULDS])\1+', r'\1+', r)
    
    return result

#whether the word is a title
def has_title(token):
    return token[0].isupper()

#get the top 1000 word list
def get_freq(corpus):
    x = {}
    for line in corpus:
        line = line.strip('\n') 
        if line:
            word = line.split('\t')[1:][0].lower()
            if word not in stopwords and word not in string.punctuation:
                if word not in x:
                    x[word] = 1
                else:
                    x[word] += 1
        else:
            continue  
    
    freq = sorted(x, key=x.get, reverse=True)[:1000]
    return freq
    
    
def readiter(corpus, brown_cluster, brown_no, name_list, region_list, freq, wv):
            
    output = []
    for line in corpus:
        line = line.strip('\n')
                
        if line:
            fields = line.split('\t')   
            word = fields[1]
                
            #feat_items: store the features
            feat_items = {'F': []} # 'F' for fields.
                            
            # the original word
            feat_items['w'] = word
             
            # the word2vec cluster id
            feat_items['wv'] = str(wv[word.lower()]) if word.lower() in wv else 'N/A'
            
            # the word has a symbol
            feat_items['pat'] = get_pattern(word)
            
            # the part-of-speech
            feat_items['pos'] = fields[2]
                
            # where in the sentence the word occurred (1=end, 10=tenth word)
            feat_items['freq'] = 'yes' if word.lower() in freq else 'no'
            
            # the tag
            feat_items['y'] = fields[3]
                    
            # start with a uppercase letter
            feat_items['su'] = 'yes' if has_title(word) else 'no' 
                    
            # The prefix and suffix of the word
            feat_items['p1'] = word[:1] if len(word) >= 1 else 'N/A'
            feat_items['p2'] = word[:2] if len(word) >= 2 else 'N/A'
            feat_items['p3'] = word[:3] if len(word) >= 3 else 'N/A'
           
            feat_items['s1'] = word[-1:] if len(word) >= 1 else 'N/A'
            feat_items['s2'] = word[-2:] if len(word) >= 2 else 'N/A'
            feat_items['s3'] = word[-3:] if len(word) >= 3 else 'N/A'
         
            # Brown cluster representation
            feat_items['br4'] = brown_cluster[word.lower()][:4] if word.lower() in brown_cluster and len(brown_cluster[word.lower()]) >= 4 else 'N/A'  
            feat_items['br8'] = brown_cluster[word.lower()][:8] if word.lower() in brown_cluster and len(brown_cluster[word.lower()]) >= 8 else 'N/A'  
            feat_items['br10'] = brown_cluster[word.lower()][:10] if word.lower() in brown_cluster and len(brown_cluster[word.lower()]) >= 10 else 'N/A'  
            feat_items['br12'] = brown_cluster[word.lower()][:12] if word.lower() in brown_cluster and len(brown_cluster[word.lower()]) >= 12 else 'N/A'  
            
            # Brown cluster id
            feat_items['br_no'] = brown_no[word.lower()] if word.lower() in brown_no else 'N/A'
            
            # Whether the word is contained in a Gazetteer
            feat_items['is_n'] = 'yes' if word.lower() in name_list else 'no'
            feat_items['is_c'] = 'yes' if word.lower() in region_list else 'no'
            
            # Digits and ','.
            feat_items['d&,'] = 'yes' if is_digit_and_sym(word, ',') else 'no'
            # Digits and '.'.
            feat_items['d&.'] = 'yes' if is_digit_and_sym(word, '.') else 'no'
 
            output.append(feat_items)
        else:
            yield output
            output = []
                
#write feature file
def output_features(out_path, X):
   
    for template in templates:
        name = '|'.join(['%s[%d]' % (f, o) for f, o in template])
        X_len = len(X)
        for t in range(X_len):
            values = []
            for field, offset in template:
                p = t + offset
                if p < 0 or p >= X_len: 
                    values = []
                    break
                values.append(X[p][field])
            if values:
                X[t]['F'].append('%s=%s' % (name, '|'.join(values)))
                
    if X is not None:
        X[0]['F'].append('__BOS__') 
        X[-1]['F'].append('__EOS__') 
        
    with open(out_path, 'a') as fo:
        for t in range(len(X)):
            fo.write('%s' % X[t]['y'])
            for a in X[t]['F']:
                if isinstance(a, str):
                    fo.write('\t%s' % a.replace(':', '__COLON__'))
                else:
                    fo.write('\t%s:%f' % (a[0].replace(':', '__COLON__'), a[1]))
            fo.write('\n')
        fo.write('\n')
 
 
# get person and place names   
name_files = ["./data/name1.csv", "./data/name2.csv"]
city_files = ["./data/country_city.csv"]

def get_name_list(name_files):
    names = set()
    for name_file in name_files:
        with open(name_file) as f:
            next(f)
            next(f)
            for line in f:
                name = line.split(",")[0]
                if name.lower() not in stopwords:
                    names.add(name.lower())
    return names


def get_country_city(city_files):
    country_city = set()
    for city_file in city_files:
        with open(city_file) as f:
            next(f)
            for line in f:
                parts = line.split(",")
                country_city.add(parts[2].lower())
                country_city.add(parts[4].lower())
                country_city.add(parts[7].lower())
    return country_city

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file', help='input file')
    parser.add_argument('output_file', help='output file')

    args = parser.parse_args()
    
    corpus = []
    brown_cluster, brown_id = get_brown()
    name_list = get_name_list(name_files)
    region_list = get_country_city(city_files)
  
    with open(args.input_file, 'r') as input_file:
        corpus = input_file.readlines()
        
    freq = get_freq(corpus)
    
    #word2vec cluster
    wv = get_word2vec()
    
    iter = readiter(corpus, brown_cluster, brown_id, name_list, region_list, freq, wv)
    for sample in iter:
        output_features(args.output_file, sample)

if __name__ == "__main__":
    main()
