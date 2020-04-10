import csv
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import re

def get_words(text):
    """ Convert ith training example text to lower case and split it into 
    words (list of words)  and inserts a space before punctiation symbols, 
    so they are interpreted as a single word that has an embedding. 
    For instance, "hello," does not have an embedding because
    it would be taken as a single word with comma as last character

    Arguments:
    sentence -- the text that will be splited into words

    Returns:
    sentence_words -- A list of words
    """
    text=re.sub(r'([a-zA-Z])(!)', r'\1 \2', text)
    text=re.sub(r'([a-zA-Z])(\.)', r'\1 \2', text)
    text=re.sub(r'([a-zA-Z])(,)', r'\1 \2', text)
    text=re.sub(r'([a-zA-Z])(\?)', r'\1 \2', text)
    text=re.sub(r'([a-zA-Z])(:)', r'\1 \2', text)
    text=re.sub(r'([a-zA-Z])(;)', r'\1 \2', text)
    text_words =text.lower().split() 
    return text_words

def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices 
    corresponding to words in the sentences.The output shape should be such 
    that it can be given  to `Embedding()` 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every 
    sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences 
    from X, of shape (m, max_len)

    Credit: This function is from the RNN course from Andrew NG at 
    deeplearning.ai, with some modifications
    """
    m = X.shape[0]  # number of training examples
    # Initialize X_indices as a numpy matrix of zeros and the correct 
    # shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        text_words=get_words(X[i])
        j = 0
        # Loop over the words of text_words
        for w in text_words:
            # Set the (i,j)th entry of X_indices to the index of the 
            # correct word.
            X_indices[i, j] = word_to_index[w]
            j = j + 1
    return X_indices

def read_glove_vecs(glove_file):
    """ Reads the glove vectors and returns mappings from word to index, 
    index to word, and word to vector.
    
    Arguments:
    filename -- the path of glove file

    Returns:
    words_to_index -- dictionary mapping a word to an index
    index_to_words -- dictionary mapping a index to a word
    word_to_vec_map -- dictionary mapping a word to the vector representation

    Credit: This function is from the RNN course from Andrew NG at 
    deeplearning.ai, with some modifications
    """
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def softmax(x):
    """
    Compute softmax values for each set of scores in x.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# def delete_dataset(dataset_id):
#     """
#     Deletes all info related to a dataset.
    
#     Arguments:
#     filename -- the path of glove file
#     """
#     try:
#         TrainingExample.objects.filter(dataset_id=dataset_id).delete()
#         Dataset.objects.filter(id=dataset_id).delete()
#     #An exception should not happen here, but this method is called
#     #inside except in views, so need to make sure preventing an exception
#     #is shown to the user.
#     except Exception as exception:
#         logging.exception("Unexpected exception has occurred")

def read_csv_multitask(filename):
    """ Reads a file that contains in each line a sentence with its 
    respective labels.
    For instance:
    The movie was terrible, 1_2_3 
    Would be a sentence with three labels 
    
    Arguments:
    filename -- the path of the csv containing the file with the data

    Returns:
    X -- a list of texts 
    labels -- a list of labels  
    """
    texts = []
    labels = []
    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            texts.append(row[0])
            labels.append(row[1].split("_"))
    X = np.asarray(texts)
    return X, labels 

def read_csv(filename = 'data/emojify_data.csv'):
    phrase = []
    label = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            label.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(label, dtype=int)

    return X, Y

def convert_to_one_hot(Y, C):
    """ obtains the one hot vector representation of each element  of an 
    array of integers.

    Arguments:
    Y -- array of numbers, for each of this number it will be obtianed 
        the one hot vector representation
    C -- the size of the vector for the one hot vector representtion
    
    Returns:
    an array of one hot vectors

    Credit: This function is from the RNN course from Andrew NG 
    at deeplearning.ai, with some modifications
    """
    return np.eye(C)[Y.reshape(-1)]


def convert_to_one_hot_multitask(Y, C):
    """ obtains the multi hot encode vector representation of each element  
    of an array of arrays of integers.
    Arguments:
    Y -- is a matrix, and each row are the classes of a training example, 
        for each row it will be obtianed the multi hot vector representation
    C -- the size of the vector for the one hot vector representtion
    
    Returns:
    an array of multi hot vectors
    """
    result=[]
    for y in Y:#Y 
        added_Y=np.zeros(C) 
        for clase in y:#y is a row of Y, and clase is a class of y
            y = np.eye(C)[int(clase)]
            added_Y=added_Y+y
        result.append(added_Y)
    return np.asarray(result)


def write_uploaded_file(f,name):
    """ Writes a potentially big file by chunks
    
    Arguments:
    f -- File to write 
    """
    with open(uploaded_datasets_path+name+'.txt', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)



data_path=os.path.dirname(os.path.realpath(__file__))+"/data/"
uploaded_datasets_path=data_path+"/uploaded_datasets/"
trained_datasets_path=data_path+"/trained_models/"
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(
    data_path+'glove.6B.50d.txt')

