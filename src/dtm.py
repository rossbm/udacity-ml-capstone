import os

import luigi
import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from ..data.clean import CleanData

#get top level of project (since this file in located two levels down from top)
PROJECT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

def create_dtm(train_text, test_text, max_ngram=1):
#inputs: the text component of train.pkl - for fit and transform
# the text test.pkl for transform
# outputs: train_dtm (a sparse matrix) and test_dtm (a sparse matrix)
    tfidf = TfidfVectorizer(analyzer = word_tokenize, ngram_range = (1,max_ngram))
    dtm_train = tfidf.fit_transform(train_text)
    dtm_test = tfidf.transform(test_text)
    return dtm_train, dtm_test, tfidf

class CreateDTM(luigi.Task):
    max_ngram = 3

    def run(self):
        out_path = self.output().path

        os.chdir(PROJECT_DIR)
        #read in data
        train = joblib.load('data/interim/train.pkl')
        test = joblib.load('data/interim/test.pkl')

        train_dtm, test_dtm, vectorizer = create_dtm(train.full_text, test.full_text, max_ngram=self.max_ngram)
        dtm_dict = {'train' : train_dtm,
                    'test' : test_dtm,
                    'vectorizer' : vectorizer}

        joblib.dump(dtm_dict, out_path)

    def requires(self):
        return CleanData()

    def output(self):
        #tell luigi where to output to
        output_path = 'data/processed/dtm_features.pkl'
        return luigi.LocalTarget(output_path)