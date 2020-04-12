from langdetect import detect
from nltk.corpus import wordnet
def print_synonyms(lang, query):
    """ Obtains strings from a fiel
    
    Arguments:
    lang -- The language (currently only supported english)
    query -- the keywords for which the synonyms will be printed
    """

    #we eliminate duplicated spaces with split() and join(), then we split by space
    words_query= ' '.join(query.strip().split()).split(" ")
    if lang=='en':
        print("\nSYNONYMS:*******************************")
        for word in words_query:
            print("for word '"+word+"':")
            synonyms = []
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    synonyms.append(l.name())

            print(",".join(synonyms) )
    else:
        print("WARNING: synonyms can only be found in english language")

def get_language_strings(lang, strings_with_offset, out_lang_file):
    """ Filter strings in a specific language from a list of strings
    
    Arguments:
    lang -- language to filter
    strings_with_offset -- strings with offset
    out_lang_file -- file path to write all the strings

    Returns:
    strings_in_lang, strings_in_lang_with_offset -- The filter strings without and with offset
    """

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
                #print(string_no_offset)
                if lang==detect(string_no_offset):
                    #print(string_no_offset)
                    strings_in_lang_with_offset.append(string_offset)
                    out_lang_file.write(string_offset+"\n")
                    strings_in_lang.append(string_no_offset)
            except Exception:
                None #Lang detect exception because the string contained no language. Just continue...
    out_lang_file.close()
    return strings_in_lang, strings_in_lang_with_offset

def get_string_matches(strings_with_offset, word):
    """ Filter strings  by an exact match with a list of synonyms obtained from a word
    
    Arguments:
    strings_with_offset -- strings with offset
    word -- the word that will be used to get synonyms

    Returns:
    matches -- The strings that matches one or more synonyms
    """

    matches=[]
    found=False
    print("\nSYNONYM EXACT MATCHES:******************")
    for string in strings_with_offset:
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                if " ".join(l.name().split("_")) in string:
                    matches.append(string)
                    print(string )
                    found=True
                    break
            if found:
                found=False
                break
    return matches