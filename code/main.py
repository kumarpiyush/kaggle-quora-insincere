import os
import pandas
import nltk
import time
import logging
from collections import Counter

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torchtext.vocab as torchvocab

import tensorboard_logger as tbrd

logging.basicConfig(level=logging.INFO)

class Constants() :
    class Data() :
        datadir = "../input"
        train = "train.csv"
        test = "test.csv"
        glove_dir = "embeddings/glove.840B.300d"

        model_dir = "../model_dir"

        output_file = "submission.csv"

    class SpecialTokens() :
        bos = "<s>"
        eos = "</s>"
        unk = "<UNK>"
        pad = "<pad>"

    embedding_size = 300
    max_token_length = 10

    class Training() :
        n_epochs = 50
        batch_size = 128

    class Lstm() :
        hidden_unit_size = 32


class DataManager() :
    def __init__(self) :
        glove = torchvocab.GloVe(name = '840B', dim = 300, cache = os.path.join(Constants.Data.datadir, Constants.Data.glove_dir))
        counter = Counter([w for w in glove.stoi])
        self.vocab = torchvocab.Vocab(counter, vectors = glove, specials = [Constants.SpecialTokens.pad, Constants.SpecialTokens.unk])
        self.embedding_layer = nn.Embedding.from_pretrained(self.vocab.vectors)

    def load_data(path) :
        # headers are qid, question_text, target
        return pandas.read_csv(path)

    def train_val_split(df, test_frac = 0.1, random_state = 5) :
        return train_test_split(df, test_size = test_frac, random_state = random_state)

    def batch(df, batch_size = Constants.Training.batch_size) :
        batches = []
        st = 0
        while st < len(df) :
            batches.append(df[st : st+batch_size])
            st += batch_size

        return batches

    def tokenize(sentence) :
        return nltk.word_tokenize(sentence)

    def tokens_to_indices(self, tokens) :
        return np.array([self.vocab.stoi[t] for t in tokens])

    def sentences_to_indices(self, untokenized_sentences) :
        tokenized_sentences = [
            [Constants.SpecialTokens.bos]
            + DataManager.tokenize(s)[:Constants.max_token_length]
            + [Constants.SpecialTokens.eos]
        for s in untokenized_sentences]

        for ts in tokenized_sentences :
            for i in range(2+Constants.max_token_length - len(ts)) :
                ts.append(Constants.SpecialTokens.pad)

        indexed_sentences = [self.tokens_to_indices(s) for s in tokenized_sentences]

        return indexed_sentences

    def batch_indices_to_tensor(batch) :
        nd = np.ndarray([len(batch), 2+Constants.max_token_length], np.int64)

        for i in range(len(batch)) :
            for j in range(2+Constants.max_token_length) :
                nd[i][j]=batch.iloc[i][j]

        return torch.from_numpy(nd)


class LRModel() :
    def __init__(self, lrThreshold = 0.5) :
        self.lrThreshold = lrThreshold

        self._cv = CountVectorizer(tokenizer = self.featurize, lowercase = False)
        self._lr = LogisticRegression(penalty = "l1", C = 0.5, tol = 1e-8, max_iter = 1000)
        self.pipeline = Pipeline(steps = [("cv",self._cv), ("lr",self._lr)])

    def featurize(self, sentence) :
        tokens = DataManager.tokenize(sentence)

        features = []
        for i in range(len(tokens)) :
            features.append(tokens[i])
            if i > 0 : features.append(tokens[i-1] + "_" + tokens[i])
            if i > 1 : features.append(tokens[i-2] + "_" + tokens[i-1] + "_" + tokens[i])

        return features

    def fit(self, dataset, weights = None) :
        if weights == None :
            self.pipeline.fit(dataset.question_text, dataset.target)
        else :
            self.pipeline.fit(dataset.question_text, dataset.target, lr__sample_weight = weights)

    def predict_proba(self, dataset) :
        return self.pipeline.predict_proba(dataset.question_text)

    def predict(self, dataset) :
        def _thresholdPredict(probs) :
            return [i[1] > self.lrThreshold for i in probs]
        return _thresholdPredict(self.pipeline.predict_proba(dataset.question_text))

    def calculate_f1(self, val_set) :
        pred = [int(x) for x in self.predict(val_set)]
        return f1_score(val_set.target, pred)

    def feature_importance(self) :
        print("LR bias : {}".format(self._lr.intercept_[0]))

        weights = [(k, self._lr.coef_[0][v]) for k,v in self._cv.vocabulary_.items()]
        weights = sorted(weights, key = lambda x : -abs(x[1]))

        for i in range(20) :
            print(weights[i][0], weights[i][1])


class LstmModel(nn.Module) :
    def __init__(self, embedding_layer) :
        super().__init__()
        self.embedding_layer = embedding_layer
        self.lstm = nn.LSTM(Constants.embedding_size, Constants.Lstm.hidden_unit_size, batch_first = True)
        self.linear = nn.Linear(Constants.Lstm.hidden_unit_size, 2)
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, batch) :
        t_embedded = self.embedding_layer(batch)
        t_lstm, _ = self.lstm(t_embedded)
        t_lstm = t_lstm[:, Constants.max_token_length-1, :]
        t_lin = self.linear(t_lstm)
        return t_lin

    def step(self, batch, labels) :
        t_presoftmax = self.forward(batch)
        t_loss = self.loss(t_presoftmax, labels)
        t_loss.backward()
        return t_loss

    def predict_proba(self, batch) :
        with torch.no_grad() :
            t_presoftmax = self.forward(batch)
            t_softmax = self.softmax(t_presoftmax)
            return t_softmax[:,1].numpy()

    def predict(self, batch) :
        return [0 if x<0.5 else 1 for x in self.predict_proba(batch)]

    def calculate_f1(self, batch, labels) :
        pred = self.predict(batch)
        return f1_score(labels, pred)


class Trainer() :
    def __init__(self, model) :
        self.model = model
        self.optimizer = torch.optim.Adadelta([p for p in model.parameters() if p.requires_grad])

    def train(self, train_set, val_set = None) :
        for epoch in range(Constants.Training.n_epochs) :
            logging.info("Epoch : {}".format(epoch))

            for batch in DataManager.batch(train_set) :
                self.optimizer.zero_grad()
                loss = self.model.step(DataManager.batch_indices_to_tensor(batch.token_ids), torch.from_numpy(batch.target.as_matrix()))
                tbrd.log_value("loss", loss)
                self.optimizer.step()

            if val_set is not None :
                val_f1 = self.model.calculate_f1(DataManager.batch_indices_to_tensor(val_set.token_ids), torch.from_numpy(val_set.target.as_matrix()))
                logging.info("Validation F1 : {}".format(val_f1))
                tbrd.log_value("Validation F1", val_f1, epoch)


def main() :
    modelName = "LSTM"

    if not os.path.exists(Constants.Data.model_dir) :
        os.mkdir(Constants.Data.model_dir)
    tbrd.configure(Constants.Data.model_dir)

    start_time = time.time()

    train_set = DataManager.load_data(os.path.join(Constants.Data.datadir, Constants.Data.train))
    train_set, val_set = DataManager.train_val_split(train_set)
    test_set = DataManager.load_data(os.path.join(Constants.Data.datadir, Constants.Data.test))
    logging.info("Data loaded")

    train_set = train_set[:1000000]
    val_set = val_set[:10000]
    test_set = test_set[:1000]

    if modelName == "LR" :
        model = LRModel()
        model.fit(train_set)
        logging.info("Model trained")

        acc = model.calculate_f1(val_set)
        logging.info("Validation F1 : {}".format(acc))

        pred = model.predict(test_set)
        logging.info("Predictions calculated")

        test_set["prediction"] = [int(x) for x in pred]
        test_set.to_csv(Constants.Data.output_file, columns = ["qid", "prediction"], index = False)

        model.feature_importance()

    elif modelName == "LSTM" :
        dm = DataManager()
        logging.info("DataManager loaded")

        train_set["token_ids"] = dm.sentences_to_indices(train_set.question_text)
        val_set["token_ids"] = dm.sentences_to_indices(val_set.question_text)
        test_set["token_ids"] = dm.sentences_to_indices(test_set.question_text)

        model = LstmModel(dm.embedding_layer)

        trainer = Trainer(model)
        trainer.train(train_set, val_set)
        logging.info("Model trained")

    end_time = time.time()
    logging.info("Total time : {} secs".format(end_time - start_time))


if __name__ == "__main__" :
    main()
