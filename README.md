# forensics_project

## Installation:

On linux:

clone the proyect from this repo.

Then install dependencies:


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



Then, use it!

## Usage

positional arguments:

file_path             Path of the file that will be search for strings in language

optional arguments:

-h, --help            show this help message and exit

-L LANG, --lang LANG  language to filter, if a language different from english is picked, it only prints strings in that language,
                        because search or synonym techniques are only supported in english

-O OUT, --out OUT     Output file for strings obtained in specific a if this is not chosen, the default file name is
                        "out_lang_strings.txt"

-Q QUERY, --query QUERY
                        search for word or similar phrase

-M MAX, --max MAX     max results returned

-s                    Search using exact match for synonyms of a word in --query.

-lsy                  list synonyms of each word in --query

-V, --version         show program version

-P, --predict         make predictions according previously trained dataset

-T TRAIN, --train TRAIN


## Usage examples:

python3 run.py sample_file.txt -O "EnglishStrings.txt"

python3 run.py sample_file.txt -Q use

python3 run.py ../fat16.dd  -Q "clicking drivers" -O "fat16strings.txt"

python3 run.py sample_file.txt -Q "using a computer"

python3 run.py sample_file.txt -L 'es'

python3 run.py sample_file.txt --lsy -s -Q "girl"

python3 run.py a.txt -Q click -P

python3 run.py a.txt -Q click -T "deep_learning_layer/data/abuse.csv"

## Credit:

Professor Bill Reed for allowing to develop this project on his Host Based Forensics class, and sdeeplearning.ai and Andrew Ng for teaching me deep learning for free online.

NOTE: In the file evidence_predictor.py, some parts of the code are from the skeleton for a deep learning course I took to learn about deep learning. The skeleton was provided by Andrew NG.
