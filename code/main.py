import os
import pandas
import nltk
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score

class Constants() :
    class Data() :
        datadir = "../input"
        train = "train.csv"
        test = "test.csv"

        output_file = "submission.csv"


class DataManager() :
    # headers are qid, question_text, target
    def __init__(self, path) :
        self.data = pandas.read_csv(path)

    def tokenize(sentence) :
        tokens = nltk.word_tokenize(sentence)
        features = []
        for i in range(len(tokens)) :
            features.append(tokens[i])
            if i > 0 : features.append(tokens[i-1] + "_" + tokens[i])
            if i > 1 : features.append(tokens[i-2] + "_" + tokens[i-1] + "_" + tokens[i])

        return features


class LRModel :
    def __init__(self, lrThreshold = 0.5) :
        self.lrThreshold = lrThreshold

        self.cv = CountVectorizer(tokenizer = DataManager.tokenize, lowercase = False)
        self.lr = LogisticRegression(penalty = "l2", C = 0.5, tol = 1e-8, max_iter = 1000)
        self.pipeline = Pipeline(steps = [("cv",self.cv), ("lr",self.lr)])

    def fit(self, featureVector, labels, weights = None) :
        if weights == None :
            self.pipeline.fit(featureVector, labels)
        else :
            self.pipeline.fit(featureVector, labels, lr__sample_weight = weights)

    def predict_proba(self, featureVector) :
        return self.pipeline.predict_proba(featureVector)

    def predict(self, featureVector) :
        def _thresholdPredict(probs) :
            return [i[1] > self.lrThreshold for i in probs]
        return _thresholdPredict(self.pipeline.predict_proba(featureVector))


def main() :
    start_time = time.time()

    train_set = DataManager(os.path.join(Constants.Data.datadir, Constants.Data.train)).data
    test_set = DataManager(os.path.join(Constants.Data.datadir, Constants.Data.test)).data
    print("Data loaded")

    # train_set = train_set[:1000]
    # test_set = test_set[:1000]

    model = LRModel()
    model.fit(train_set.question_text, train_set.target)
    print("Model trained")
    pred = model.predict(test_set.question_text)
    print("Predictions calculated")

    test_set["prediction"] = [int(x) for x in pred]
    test_set.to_csv(Constants.Data.output_file, columns = ["qid", "prediction"], index = False)

    end_time = time.time()
    print("Total time : {} secs".format(end_time - start_time))

if __name__ == "__main__" :
    main()
