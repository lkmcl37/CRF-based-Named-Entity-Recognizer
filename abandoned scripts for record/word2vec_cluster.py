from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
import nltk
import codecs
 
from sklearn import cluster
from sklearn import metrics
 
# training data

def word2vec_cluster(in_file, out_file):
    sentences = []
    with codecs.open(in_file, 'r',encoding='utf-8', errors='ignore') as in_file:
        corpus = in_file.readlines()
        for line in corpus:
            line = line.strip('\n')
            if not line:
                continue
            line = line.lower()
            line = line.split(" ")
            sentences.append(line)

    print("training model...")
    model = Word2Vec(sentences, min_count=2)
     
    print("get vector data...")
    X = model[model.wv.vocab]

    NUM_CLUSTERS=50
    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance,avoid_empty_clusters=True, repeats=30)

    print("assigning cluster..")
    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
     
    words = list(model.wv.vocab)

    with open(out_file, 'a') as out_file: 
        for i, word in enumerate(words):  
            out_file.write(word + ":" + str(assigned_clusters[i]) + '\n')
     
     
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file', help='input file')
    parser.add_argument('output_file', help='output file')

    args = parser.parse_args()
    word2vec_cluster(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
