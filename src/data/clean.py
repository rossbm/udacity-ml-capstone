import os
import luigi
import pandas as  pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

#get top level of project (since this file in located two levels down from top)
PROJECT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
os.chdir(PROJECT_DIR)

from src.data.download import DownloadData

class CleanData(luigi.Task):
    input_path = 'data/external/jokes.csv'

    def run(self):
        test_path = self.output()[0].path
        train_path = self.output()[1].path

        os.chdir(PROJECT_DIR)
        #read csv
        df = pd.read_csv(self.input_path)
        #drop short
        text_length = df.text.str.len()
        df = df[text_length >= 4]

        #drop neutral and create dummy
        df = df[(df.ups >= 10) | (df.ups <= 1)]
        df["funny"] = df.ups >= 10

        #create full text and drop unnecessary
        df["full_text"] = df.title + "\n " + df.text
        df = df.loc[:, ["funny", "full_text"]]

        #split and write
        train, test = train_test_split(df, test_size = 0.125, random_state = 10182017, stratify = df.funny)
        joblib.dump(train, train_path)
        joblib.dump(test, test_path)

    def requires(self):
        return DownloadData()

    def output(self):
        #tell luigi where to output to
        test_path = 'data/interim/test.pkl'
        train_path = 'data/interim/train.pkl'
        return luigi.LocalTarget(test_path), luigi.LocalTarget(train_path)