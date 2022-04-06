from sklearn_crfsuite import CRF
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics._classification import classification_report
from sklearn.naive_bayes import MultinomialNB
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import pickle
import glob
import os

# load dictionary
path = './dictionary'
filenames = glob.glob(path + "/*.txt")
dict = []

for file in filenames:
    with open(file= file, mode= 'r', encoding='utf-8') as file_in:
        for line in file_in:
            
            line = line.replace("\n" or " \n" , "")
            dict.append(line)


# check in dictionary
def check_dict(word, dict):
    for w in dict:
        if w == word:
            return 1
    return 0

# features extraction from word
def word2features(sent, i):
    word = sent[i][0]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 1:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.check_dict()': check_dict(" ".join([word1, word]), dict),
            '-2:word.lower()': sent[i-2][0].lower(),
            '-2:word.istitle()': sent[i-2][0].istitle(),
            '-2:word.isupper()': sent[i-2][0].isupper(),

        })
    else:
        features['BOS'] = True

    if i < len(sent)-2:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.check_dict()': check_dict(" ".join([word, word1]), dict),
            '+2:word.lower()': sent[i+2][0].lower(),
            '+2:word.istitle()': sent[i+2][0].istitle(),
            '+2:word.isupper()': sent[i+2][0].isupper(),
        })
    else:
        features['EOS'] = True

    if i>1 and i < (len(sent)-2):
        word_prev = sent[i-1][0]
        word_next = sent[i+1][0]
        features.update({
            'combined_word.check_dict()': check_dict(" ".join([word_prev, word, word_next]), dict),
        })

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]


# Define class
class CRF_tokenizer():
    def __init__(self):
        self.dict = dict
        self.model = CRF(algorithm = "lbfgs",
          c1 = 0.1, 
          c2 = 0.1, 
          max_iterations= 500, 
          all_possible_transitions=True)

    def train(self, X, Y):
        print("Training time...")
        self.model.fit(X,Y)

    def evaluate(self, X, Y):
        Y_pred = self.model.predict(X)
        print("Accuracy score: ", metrics.flat_accuracy_score(Y, Y_pred))

    def load(self, pretrained_model_path):
        with open(pretrained_model_path, 'rb') as f:
            self.model = pickle.load(f)       

    #predict a raw sentence
    def predict_new(self, text):
        text = text.replace(',', '')
        text = text.replace('.', '')
        text = text.replace(';', '')
        text = text.replace(':', '')
        text = text.replace('?', '')
        text = text.split(" ")
        text = [(word, -1) for word in text]
        untag_text = []
        untag_text.append(sent2features(text))
        prediction = self.model.predict(untag_text)
        tagged_text = []

        for i in range(len(prediction[0])):
            tagged_text.append((text[i][0], prediction[0][i]))
        token_list = []
        pre_word = tagged_text[0][0]

        for i in range(1, len(tagged_text)):
            if tagged_text[i][1] == "I_W":
                pre_word += "_" + tagged_text[i][0]
            else:
                token_list.append(pre_word)
                pre_word =  tagged_text[i][0]
        token_list.append(pre_word)
        return token_list

 
