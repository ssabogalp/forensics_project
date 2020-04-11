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
import sys

#remember to download wordnet
#nltk.download()

#MAX_WIDTH_OF_100_IN_LINE_OF_CODE_12345678901234567890123456789012345678901234567890123456789012345

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('file_path',  help='Path of the file that will be search for strings \
        in language')
    parser.add_argument("-L",'--lang',  help='language to filter, if a language different from \
        english is picked, it only prints strings in that language, because search  or synonym \
        techniques are only supported in english')
    parser.add_argument("-O",'--out',  help='Output file for strings obtained in specific a \
        if this is not chosen, the default file name is "out_lang_strings.txt"')
    parser.add_argument("-Q",'--query',  help='search for word or similar phrase')
    parser.add_argument("-M",'--max',  help='max results returned')
    parser.add_argument('-s', action='store_true',help="Search using exact match for synonyms \
        of a word  in --query.")
    parser.add_argument('-lsy', action='store_true',help="list synonyms of each word in --query")
    parser.add_argument("-V", "--version", help="show program version", action="store_true")
    return parser.parse_args()

def print_synonyms(lang):
    #we eliminate duplicated spaces with split() and join(), then we split by space
    words_query= ' '.join(args.query.strip().split()).split(" ")
    if lang=='en':
        print("SYNONYMS:")
        for word in words_query:
            print("for word '"+word+"':")
            synonyms = []
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    synonyms.append(l.name())
                    
            print(",".join(synonyms))
            print("")#line break
    else:
        print("WARNING: synonyms can only be found in english language")

def get_strings_with_offset(file_path):
    #EXECUTE STRINGS COMMAND
    process = subprocess.Popen(["strings","--radix=x",file_path],  stdout=subprocess.PIPE)
    strings_with_offset=[]
    while True:
        line =process.stdout.readline()
        if not line:
            break
        strings_with_offset.append(str(line.strip()))
    return strings_with_offset

def get_language_strings(lang, strings_with_offset):
    strings_in_lang=[]
    strings_in_lang_with_offset=[]
    #FILTER ENGLISH STRINGS
    for string_offset in strings_with_offset:
        if len(string_offset.split(" "))>1:
            #get the string withouth offset at the beginning and make it lower case
            string_no_offset= string_offset.split(" ",1)[1].lower()
            #only take letters and spaces from string
            string_no_offset= ''.join(x for x in string_no_offset if x.isalpha() or x==' ') 
            if len(string_no_offset.split(" "))<3:
                continue
            try:
                print(string_no_offset)
                if lang==detect(string_no_offset):
                    print(string_no_offset)
                    strings_in_lang_with_offset.append(string_offset) 
                    out_lang_file.write(string_offset+"\n")
                    strings_in_lang.append(string_no_offset)
            except Exception:
                None #Lang detect exception because the string contained no language. Just continue...
    return strings_in_lang, strings_in_lang_with_offset

def get_string_matches(strings_with_offset, word):
    matches=[]
    found=False
    for string in strings_with_offset:
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                if l.name() in string:
                    matches.append(string)
                    print(string)
                found=True
                break
            if found:
                found=False
                break
    return matches



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

def get_tfidf_model(strings_in_lang):
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
        #print(corpus_list[idx])
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




args = parse_arguments()
#Assign defaults (with string this could be set in add_argument, but since we have different object
# I did it all here)
lang=None
if args.lang:
    lang=args.lang
else:
    lang='en'
out_lang_file=None
if args.out:
    out_lang_file=open(args.out, 'w')
else:
    out_lang_file=open('out_lang_strings.txt', 'w')
max_results=None
if args.max:
    max_results=args.max
else:
    max_results=30 

#PRINT SYNONYMS
if args.lsy:
    print_synonyms(lang)



if not lang=='en':
    print("WARNING: Since a language different from english was chosen, the strings on that lanague \
will be output and the program will terminate")

#DETERMINE LANGUAGE OF EACH SENTENCE.

if not lang=='en':
    exit()

strings_with_offset=get_strings_with_offset(args.file_path)

if args.s:
    get_string_matches(strings_with_offset, args.query)

strings_in_lang, strings_in_lang_with_offset = get_language_strings(lang, strings_with_offset)

print(strings_in_lang)
print(strings_in_lang_with_offset)
#TFIDF THING
model, index, dictio=get_tfidf_model(strings_in_lang)
#top_k=3
#results=toptexts(model[corpus_list[0]], range(len(texts)),index, top_k)


results=toptexts(model[dictio.doc2bow(create_bow(args.query))], range(len(strings_in_lang)),index, max_results)

print(create_bow(args.query))
for r in results:
    print(r)
    # if r[1] <= 0:
    #     break
    print(strings_in_lang_with_offset[r[0]])
