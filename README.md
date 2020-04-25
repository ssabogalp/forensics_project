# forensics_project

Installation:

## Run:
On linux:

apt-get install python3-pip python3-dev
python3 -m pip install tensorflow
python3 -m pip install keras
python3 -m pip install numpy
python3 -m pip install sklearn
python3 -m pip install nltk
python3 -m pip install langdetect
python3 -m pip install gensim

Then, to install wordnet and punkt, in a python shell run:

import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

An interface will open, press d, and write wordnet

## Usage:

## thoughts

left stopwords
match exact synonyms combinatorix

Credit:
deeplearning.ai for
Andrew Ng

NOTE: In the file evidence_predictor.py, some parts of the code are from the skeleton for a deep learning course I took to learn about this. The skeleton was provided by Andrew NG.