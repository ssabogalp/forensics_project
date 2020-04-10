import argparse
from langdetect import detect
import subprocess
from nltk.corpus import wordnet
import nltk
from gensim.utils import lemmatize,toptexts
from gensim.corpora.dictionary import Dictionary 
from gensim import similarities
import gensim
from gensim.models import TfidfModel
from nltk.stem.porter import *

#remember to download wordnet
#nltk.download()

syns = wordnet.synsets("big")

synonyms = []
antonyms = []

for syn in wordnet.synsets("rape"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

print(set(synonyms))



def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('file_path',  help='Path of the file that will be search for strings in language')
    parser.add_argument("-L",'--lang',  help='sentence to input')
    parser.add_argument("-Q",'--query',  help='sentence to input')
    parser.add_argument("-V", "--version", help="show program version", action="store_true")
    return parser.parse_args()

args = parse_arguments()
print(args.query)

#EXECUTE STRINGS COMMAND
process = subprocess.Popen(["strings","--radix=x",args.file_path], stdout=subprocess.PIPE)
stdout = process.communicate()[0]
 

#DETERMINE LANGUAGE OF EACH SENTENCE.
texts=[]
strings = str(stdout).replace("      ","").replace("     ","").split("\\n") 
#print(strings)
strings_in_lang=[]
for s in strings:
    if len(s.split(" "))>1:
        #print(s.split(" ",1)[1])
        if args.lang:
            lang=args.lang
        else:
            lang='en'
        if lang==detect(s.split(" ",1)[1]):
            strings_in_lang.append( (  s.split(" ",1)[0] , s.split(" ",1)[1] ) )
            texts.append
print(strings_in_lang)

#TFIDF THING

texts= [
    'My love for you have dissappeared',
    'Window resize changes the default settings',
    'Clicking the button yields an exception',
    'Button clicks cause exceptions',
    'When I click the button, an error occurs',
    'Button listener not found for on click event',
    'File save option greyed out after edits',
    'Font size cannot be changed after save',
    'Dialog window won\'t close after update',
    'Cursor disappears when selecting multiple options',
    'Cannot regain focus after clicking outside window',
    'I do not love you any more',
]

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

    IMPORTANT: NO stemming and NO lemma, remove stemming and lemma from the
    configuration with the best performance, that is GENSIM_NLTK_STOP,
    which has nltk stopword, gensim lemmatization, and porter stemmer.

    Input: str type consisting of a sentence or paragraph
    
    Ouput: a list of words that have been pre-processed using stopwords,
     stemming, and lemmatization
    """
    # In dictionary, VB verb base, NN common noun, etc, see
    # https://sites.google.com/site/partofspeechhelp/home/nn_vb
    
    #TOKENIZE
    word_list=nltk.word_tokenize(text)
    #STOPWORDS
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


sentences=[]
for text in texts:
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
    print(corpus_list[idx])

# fits a model from a list of bows (from doc2bow each item)
model = TfidfModel(corpus_list)



# applies the model to all the elements in the corpus, to 
# convert pairs (int, int), so the second element in
# the pair, which in the corpus was the term frequency, now
# becomes a float that is the result of a function of the term
# frequency
corp_tfidf=[]

for c in corpus_list:
    corp_tfidf.append(model[c])
# gets matrix similarity, which is an index that can be queried
index=similarities.MatrixSimilarity(corp_tfidf)

top_k=3
results=toptexts(model[corpus_list[0]], range(len(texts)),index, top_k)

print(results)