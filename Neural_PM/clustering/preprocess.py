import nltk

# nltk.download("vader_lexicon")
# nltk.download('punkt')
# nltk.download("stopwords")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("wordnet")
from bs4 import BeautifulSoup
import re

from nltk.stem import WordNetLemmatizer

from nltk.sentiment.util import *

from nltk.corpus import stopwords



def contractions(s):
    s = re.sub(r'won’t', 'will not', s)
    s = re.sub(r'would’t', 'would not', s)
    s = re.sub(r'could’t', 'could not', s)
    s = re.sub(r'\’d', ' would', s)
    s = re.sub(r'can\’t', 'can not', s)
    s = re.sub(r'n\’t', 'not', s)
    s = re.sub(r'\’re', ' are', s)
    s = re.sub(r'\’s', ' is', s)
    s = re.sub(r'\’ll', ' will', s)
    s = re.sub(r'\’t', ' not', s)
    s = re.sub(r'\’ve', ' have', s)
    s = re.sub(r'\'m', 'am', s)
    return s


def preprocess_text_df(text):
    stop = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    x = text
    x = " ".join(x.lower() for x in str(x).split())
    x = BeautifulSoup(x).get_text()
    x = re.sub(r'http\S+', '', x)
    x = contractions(x)
    x = ' '.join([re.sub('[^A-Za-z]+', '', x) for x in nltk.word_tokenize(x)])
    # x = re.sub('+', ' ', x)
    x = ' '.join([x for x in x.split() if x not in stop])
    x = ' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(x)])
    return x
