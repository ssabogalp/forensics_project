#
#TODOS: show graph in training
#

from utils import *
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.models import load_model


trained_datasets_path = 'data/saved_model'

class ControlPredictor(object):
    """ Recurrent Neural Network that can be trained with a small dataset, 
    because it uses a previously trained word embeddings
    """

    def pretrained_embedding_layer(self, word_to_vec_map, word_to_index):
        """
        Creates a Keras Embedding() layer and loads in pre-trained GloVe 
        50-dimensionalvectors.
        
        Arguments:
        word_to_vec_map -- dictionary mapping words to their GloVe vector 
            representation. 
        word_to_index -- dictionary mapping from words to
            their indices in the vocabulary (400,001 words)

        Returns:
        embedding_layer -- pretrained layer Keras instance
        
        Credit: This function is from the RNN course from Andrew NG at
        deeplearning.ai,with some modifications
        """
        # adding 1 to fit Keras embedding (requirement)
        vocab_len = len(word_to_index) + 1  
        # define dimensionality of your GloVe word vectors (= 50)
        emb_dim = word_to_vec_map["cucumber"].shape[0]   
        # Initialize the embedding matrix as a numpy array of zeros of shape
        # (vocab_len, dimensions of word vectors = emb_dim)
        emb_matrix = np.zeros((vocab_len,emb_dim))
        # Here we will take word_to_index and word_to_vec_map, to build a 
        # matrix of shape (400.001 , 50). Set each row "index" of the 
        # embedding matrix to be the word vector representation of the 
        # "index"th word of the vocabulary
        for word, index in word_to_index.items():
            emb_matrix[index, :] = word_to_vec_map[word]
        # Define Keras embedding layer with the correct output/input sizes,
        # make it trainable. Use Embedding(...). Trianable is set to false
        # because the datasets are too small to retrain the emeddings
        embedding_layer = Embedding( vocab_len, emb_dim, trainable=False)
        # Build the embedding layer, it is required before setting the
        # weights of the embedding layer. Do not modify the "None".
        embedding_layer.build((None,))
        # Set the weights of the embedding layer to the embedding matrix.
        # Your layer is now pretrained.
        embedding_layer.set_weights([emb_matrix])
        return embedding_layer


    def create_model(self, input_shape, word_to_vec_map, word_to_index):
        """
        Creates the keras model
        
        Arguments:
        input_shape -- shape of the input, usually (max_len,)
        word_to_vec_map -- dictionary mapping every word in a vocabulary 
            into its 50-dimensional vector representation
        word_to_index -- dictionary mapping from words to their indices in
            the vocabulary(400,001 words)

        Returns:
        model -- a model instance in Keras

        Credit: This function is from the RNN course from Andrew NG at 
        deeplearning.ai, with some modifications
        """
        # Define sentence_indices as the input of the graph, it should be of
        # shape input_shape and dtype 'int32' (as it contains indices).
        sentence_indices = Input(shape = input_shape, dtype = 'int32')
        # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
        embedding_layer = self.pretrained_embedding_layer(word_to_vec_map,
            word_to_index)
        # Propagate sentence_indices through  embedding layer, you get back 
        # the embeddings. Note: Params=20000050  comes from embeding 
        # dimensions,  400.001 times 50
        embeddings = embedding_layer(sentence_indices)   
        # Propagate the embeddings through an LSTM layer with 128-dimensional
        # hidden state. 
        X = LSTM(units =128 , return_sequences= True )(embeddings)
        # Add dropout with a probability of 0.5
        X = Dropout(0.5, noise_shape=None, seed=None)(X)
        # Propagate X trough another LSTM layer with 128-dimensional hidden
        # state
        X = LSTM(units =128 , return_sequences= False )(X)
        # Add dropout with a probability of 0.5
        X = Dropout(0.5, noise_shape=None, seed=None)(X)
        # softmax is replaced by sigmoid, because two elements can have 0.9
        # probabilit, for instance, it is not needed normalization
        X = Dense(units =5, activation="softmax")(X)
        # Add a softmax activation
        X = Activation('softmax')(X)
        # Create Model instance which converts sentence_indices into X.
        model = Model(inputs = sentence_indices, outputs=X,
            name='ControlPredictor') 
        return model

    def train(self, number_outputs, max_num_words):
        """ Trains a model based on a dataset  

        Arguments:
            id_dataset -- the id of the dataset that will be trained
        
        Returns:
            accuracy -- 
        """
        X_train, Y_train = read_csv()
        X_train_indices = sentences_to_indices(X_train, word_to_index,
            max_num_words)
        Y_train_oh = convert_to_one_hot(Y_train, C = number_outputs)
        model = self.create_model((max_num_words,), word_to_vec_map,
            word_to_index)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32,
            shuffle=True)
        model.save(trained_datasets_path+'.h5')
        self.model=model
        return model

    def predict(self, text_sample, max_num_words):
        """
        Makes a prediction for a sample text, using a model previously 
        trained with a dataset.
        
        Arguments:
        dataset_id -- the id of the data set that was used to train a model
        text_sample --the text used to predict the labels

        Returns:
        model -- a vector of individual probabilities
        """
        if self.model:
            model=self.model
        else:
            model = load_model(trained_datasets_path+'.h5')
            self.model=model
        indices_text=sentences_to_indices(np.array([text_sample]),
            word_to_index, max_len = max_num_words)
        return model.predict( indices_text )
    


#**************************
#Additional comments
#**************************
# To add new classes, modfiy C=5  in five parts. remover commas and quotes form text.
# Change  numver in X1_indices = sentences_to_indices(X1,word_to_index, max_len = 5    
# Change X = Dense(units =5, activation="softmax")(X)  
#
# If a requirement outputs more than one control, you could get the elemtns with more probability. 
# Or, you have to make your labels for multitask learning.
