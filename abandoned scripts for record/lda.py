#!/usr/bin/env
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

def generate_lda(input_f, output_o):
 
    en_stop = set(stopwords.words('english'))
    morewords = ['the', 'a', 'an', '\'s', 'cannot', 'almost', '\'\'', '``',
                 'can\'t']
    en_stop.update(morewords)

    p_stemmer = PorterStemmer()
        
    doc_set = []
    with open(input_f, 'r') as in_file:
        corpus = in_file.readlines()
        for line in corpus:
            line = line.strip('\n')
            if not line:
                continue
            doc_set.append(line)

    texts = []

    for i in doc_set:
        
        raw = i.lower()
        tokens = raw.split(" ")
        
        stopped_tokens = [i for i in tokens if not i in en_stop]
        stopped_tokens = [i for i in stopped_tokens if not i in string.punctuation]
        
        #stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        texts.append(stopped_tokens)

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=100, id2word = dictionary, passes=20)

    topics = ldamodel.show_topics(num_topics=100, num_words=10, log=False,
                                   formatted=True)

    print("List of topics:")
    with open('output_o', 'a') as file:
        for i, topic in enumerate(topics):
            # not adding topic to the tuple here prevents unicode errors
            file.write(topic + '\n')
    
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file', help='input file')
    parser.add_argument('output_file', help='output file')

    args = parser.parse_args()
    generate_lda(args.input_file, args.output_file)

if __name__ == "__main__":
    main()

