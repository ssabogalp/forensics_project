import subprocess
from nltk.corpus import wordnet
import nltk
from gensim.utils import lemmatize,toptexts
from gensim.corpora.dictionary import Dictionary
from gensim import similarities
import gensim
from gensim.models import TfidfModel
import sys


def create_bow(text):
    """ Returns an list of preporcessed words in bytes, according to:
    Gensim stop list, 2) has a length > 3 characters, 3) is lemmatized
    using WordNet, and 4) is stemmed using the Porter stemmer.

    For using wordnet, I installed nltk using
    sudo pip3 install nltk
    then I downloaded word net running
    nltk.download()
    and storing wordnet in home directory. I also downloaded
    punkt from packages, to be able to use word tokenizer,
    and stopwords from package.

    Input: str type consisting of a sentence or paragraph

    Ouput: a list of words that have been pre-processed using stopwords,
     stemming, and lemmatization
    """
    # In dictionary, VB verb base, NN common noun, etc, see
    # https://sites.google.com/site/partofspeechhelp/home/nn_vb

    #TOKENIZE
    word_list=nltk.word_tokenize(text)
    #STOPWORDS REMOVE
    text_no_stop = [word for word in word_list
            if word not in nltk.corpus.stopwords.words('english')]

    #LEMMATIZE
    wn_lem=[]
    for word in text_no_stop:
        wn_lem.append(nltk.stem.WordNetLemmatizer().lemmatize(word))
    #STEMMING
    porter_stemmer = nltk.stem.porter.PorterStemmer()
    stem_lem_text=[]
    for word in wn_lem:
        stem_lem_text.append(porter_stemmer.stem(word))
    return stem_lem_text

def get_tfidf_model(strings_in_lang):
    """ Obtains tfidf elements to conduct a search
    
    Arguments:
    file_path -- path from the binary to get strings from

    Returns:
    model, index, dictio -- elements necessary to use toptexts search
    """
    sentences=[]
    for text in strings_in_lang:
        words=[]
        for word in create_bow(text):
            # lematization from create bow does not return a string,
            # but bytes, so it is decode as utf-8 string
            words.append(word)
        sentences.append(words)
    dictio=Dictionary(sentences)
    corpus_list=[]
    for idx, sentence in enumerate(sentences):
        # doc2bow converts a list of stems into pairs of
        # (index_corpus,count_sentence)
        corpus_list.append(dictio.doc2bow(sentence))
    # fits a model from a list of bows (from doc2bow each item)
    model = TfidfModel(corpus_list)
    # applies the model to all the elements in the corpus, to convert pairs (int, int),
    # so the second element in the pair, which in the corpus was the term frequency, now
    # becomes a float that is the result of a function of the term frequency
    corp_tfidf=[]
    for c in corpus_list:
        corp_tfidf.append(model[c])
    # gets matrix similarity, which is an index that can be queried
    index=similarities.MatrixSimilarity(corp_tfidf)
    return model, index, dictio

def print_results(results, strings_in_lang_with_offset):
    """ Obtains strings from a fiel
    
    Arguments:
    file_path -- path from the binary to get strings from

    Returns:
    strings_with_offset -- strings found. The first word of each string is the offset
        on the file
    """
    print("\nTFIDF MATCHES:**************************")
    for r in results:
        if r[1] <= 0:
            break
        print(strings_in_lang_with_offset[r[0]] )