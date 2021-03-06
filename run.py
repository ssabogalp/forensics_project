import argparse
import sys
from strings_layer.strings_util import *
from lang_layer.lang import *
from tfidf_layer.tfid import *
from deep_learning_layer.evidence_predictor import EvidencePredictor
from deep_learning_layer.utils import *

#IMPORTANT: remember to download wordnet using nltk.download() after installing the dependencies

#MAX_WIDTH_OF_100_IN_LINE_OF_CODE_12345678901234567890123456789012345678901234567890123456789012345
 

SIMILARITY_THRESHOLD=0.3
MAX_WORDS=15

def parse_arguments():
    """ Parses arguments that the used can give the program for different results

    Returns:
    parser.parse_args() -- The filter strings without and with offset

    """

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
    parser.add_argument('--lsy', action='store_true',help="list synonyms of each word in --query")
    parser.add_argument("-V", "--version", help="show program version", action="store_true")
    parser.add_argument("-P", "--predict", help="make predictions according previously trained dataset", action="store_true")
    parser.add_argument("-T", "--train", help="train dataset passed in this argument")
    return parser.parse_args()

if __name__ == "__main__":
    """
        On this main method will be called the different APIs to receive the arguments from
        the user, and output the pertinent result.
    """

    # PARSE ARGUMENTS
    args = parse_arguments()
    # ASSIGN DEFAULTS. Note: defaults (with string this could be set in add_argument,
    # but since we have different object I did it all here)
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

    #TRAIN
    if args.train:
        pre=EvidencePredictor()
        #"data/abuse.csv"
        pre.train(2,MAX_WORDS,args.train)
    


    #OBTAIN STRINGS WITH THE OFFSET IN THE FILE
    strings_with_offset=get_strings_with_offset(args.file_path)

    #DETERMINE LANGUAGE OF EACH SENTENCE.
    strings_in_lang, strings_in_lang_with_offset = get_language_strings(
        lang, strings_with_offset, out_lang_file)
    print("STRINGS IN THE DEFAULT/CHOSEN LANGUAGE OUTPUT TO THE FILE SUCCESSFULY")

    #IF NOT IN ENGLISH UNTIL HERE, TERMINATE PROGRAM
    if not lang=='en':
        print("WARNING: Since a language different from english was chosen, the strings on that lanague \
    will be output and the program will terminate")
        exit()

    #PRINT SYNONYMS
    if args.lsy:
        print_synonyms(lang, args.query)
        if not args.s:
            print("\nSince argument '-s' is missing, only the synonym list will be \
        printed and the program will terminate")
            exit()

    #MATCH BY SYNONYM
    if args.s:
        get_string_matches_syn(strings_with_offset, args.query)
        exit()

    #DEEP LEARNING TRAINING
    if args.predict:
        pre=EvidencePredictor()
        print("\nDEEP LEARNING***************************")
        #print(pre.predict("we can attack when she gets off in the bus stop",MAX_WORDS))
        for string in strings_in_lang:
            vector=pre.predict(string,MAX_WORDS)
            if vector[0][1]>SIMILARITY_THRESHOLD:
                print(vector)
                print(string)

    #TFIDF
    model, index, dictio = get_tfidf_model(strings_in_lang)

    #OBTAIN RESULTS
    if args.query:
        results=toptexts(
            model[dictio.doc2bow(create_bow(args.query))], range(len(strings_in_lang)),index, max_results)

        #PRINT RESULTS
        print_results(results, strings_in_lang_with_offset)
