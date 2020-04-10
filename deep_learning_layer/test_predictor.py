import unittest
import numpy as np
from control_predictor import ControlPredictor
from utils import *

#Relative imports can only be performed in a package. So, run the code as a package.
#IMPORTANT! so you have to run your code as python3 -m deep_learning_layer.test_predictor
cp=ControlPredictor()
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
def test_sentences_to_indices(self):
    
    X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
    X1_indices = cp.sentences_to_indices(X1, word_to_index, max_len = 9)
    print("X1 =", X1)
    print("X1_indices =", X1_indices)

def test_pretrained_embedding_layer(self):
    embedding_layer = cp.pretrained_embedding_layer(word_to_vec_map, word_to_index)
    print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])

c=ControlPredictor()
c.train(5,10)
print(c.predict("Good job",10))