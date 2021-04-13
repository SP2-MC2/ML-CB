"""
__version__ = '1.0'
__author__ = 'Nathan Reitinger'


This is a work in progress:

- results vary run-to-run, in part, because of the downsampling used, which
  affects the examples trained on---final results collected from running this
  codebase will vary slightly from those noted in the paper (see Table 4)
- the models' hyperparameters are not tuned
- the models' architectures could likely be improved
- cross validation is used for testing purposes, deployment of the models would
  likely be better served with alternative techniques; for example, using the
  entire corpus of training data, or even combining the original dataset with
  the test suite, to train the model---the objective of this release is to
  validate the results shown in the paper, not to produce a deployed model
- a variety of models were picked for testing, though it is possible that other
  models will perform better or that some of these models are unsuited for a
  deployment scenario
- downsampling was done in a naive way, with randomly removed examples from the
  majority class to meet the minority class's example count; using SMOTE or
  a similar technique may improve results. In this regard, adding more negative
  examples would also likely improve the classifier's predictions on difficult,
  but close, non-fingerprinting cases
- code setup to run on CPU

"""

import sys
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import getpass
import gc
import argparse
import pickle
import warnings

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score as f1_score_sklearn
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    MaxPooling2D,
    Dropout,
    Activation,
    MaxPooling1D,
    LSTM,
)
from tensorflow.keras.preprocessing import text
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import utils
from tensorflow.keras.metrics import categorical_accuracy

from tqdm.keras import TqdmCallback

pd.set_option("chained_assignment", None)

def save_model(model):

    """
    to be used later, saves models, weights, and tokenizer
    """

    save_name_model = (
        "models/"
        + str(MODEL)
        + "__"
        + CORPUS.split("/")[2].split(".")[0]
        + "__"
        + "model"
        + ".json"
    )
    save_name_weights = (
        "models/"
        + str(MODEL)
        + "__"
        + CORPUS.split("/")[2].split(".")[0]
        + "__"
        + "weights"
        + ".h5"
    )
    save_name_tokenizer = (
        "models/"
        + str(MODEL)
        + "__"
        + CORPUS.split("/")[2].split(".")[0]
        + "__"
        + "tokenizer"
        + ".pickle"
    )

    # serialize model to JSON
    model_save = model.to_json()
    with open(save_name_model, "w") as json_file:
        json_file.write(model_save)
    # serialize weights to HDF5
    model.save_weights(save_name_weights)
    # saving tokenizer (fit on train)
    with open(save_name_tokenizer, "wb") as handle:
        pickle.dump(tokenize, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("[+] saved model (and tokenization if applicable) to disk")

    # # load model
    # json_file = open(save_name_model, 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights(save_name_weights)
    # # loading tokenizer
    # loaded_tokenizer = None
    # with open(save_name_tokenizer, 'rb') as handle:
    #     loaded_tokenizer = pickle.load(handle)
    # print("Loaded model from disk")

    ##  depends on method being used (embedding here)
    # x_predict = tokenize.texts_to_sequences([my_test])
    # x_predict = pad_sequences(x_predict, maxlen=max_length, padding='post')
    # prediction = loaded_model.predict(x_predict, verbose=0)
    # predicted_index = np.argmax(prediction)
    # print(predicted_index)

    ##  depends on method being used (BoW here)
    # x_predict_array = [my_test]
    # x_predict = loaded_tokenizer.texts_to_matrix(x_predict_array)
    # prediction = loaded_model.predict(x_predict, verbose=0)
    # predicted_index = np.argmax(prediction)
    # print(predicted_index)
    # sys.exit()
    return


def number():
    """
    helper for counting
    """
    if hasattr(number, "num"):
        number.num += 1
    else:
        number.num = 1
    return number.num

svm_fold_f1 = []
svm_fold_accuracy = []
svm_fold_precision = []
svm_fold_recall = []

def set_n(obj):
    global svm_fold_f1
    global svm_fold_accuracy
    global svm_fold_precision
    global svm_fold_recall
    if obj == "clear_out":
        svm_fold_f1.clear()
        svm_fold_accuracy.clear()
        svm_fold_precision.clear()
        svm_fold_recall.clear()
        return
    if obj == "get_results":
        return svm_fold_f1, svm_fold_accuracy, svm_fold_precision, svm_fold_recall
    else:
        svm_fold_f1.append(obj["f1"])
        svm_fold_accuracy.append(obj["accuracy"])
        svm_fold_precision.append(obj["precision"])
        svm_fold_recall.append(obj["recall"])
        return svm_fold_f1, svm_fold_accuracy, svm_fold_precision, svm_fold_recall

def classification_report_with_accuracy_score(y_test, y_pred):
    # print(classification_report(y_test, y_pred))
    # cnf_matrix = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(25,20))
    # plot_confusion_matrix(cnf_matrix, classes=['0', '1'], title="Confusion matrix", save="SVM")
    # # plt.show()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    f1 = f1_score_sklearn(y_test, y_pred)  # f1_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    print("fold number:", number())
    print("fold accuracy", accuracy)
    print("fold precision", precision)
    print("fold recall", recall)
    print("fold kappa", kappa)
    print("fold f1", f1)
    print("fold matrix\n", matrix, "\n")
    fold_results = {
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }
    set_n(fold_results)
    return f1


def plot_confusion_matrix(
    cm, classes, title="Confusion matrix", cmap=plt.cm.Blues, save=None
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)

    fmt = ".2f"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label", fontsize=25)
    plt.xlabel("Predicted label", fontsize=25)

    # plt.savefig(save)

def visualize_accuracy(history, save_name):
    """
    Plots out the accuracy measures given a keras history object
    :param history: return value from model.fit()
    """
    if history is None:
        return

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["train", "test"])
    # plt.show()

    # extra visual for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train", "test"])
    # plt.show()


# ███████ ██    ██ ███    ███
# ██      ██    ██ ████  ████
# ███████ ██    ██ ██ ████ ██
#      ██  ██  ██  ██  ██  ██
# ███████   ████   ██      ██


def svm(df):

    ############################################################################
    # some global text pre-processing
    ############################################################################

    print("\n\npreprocessing\n\n")

    # drop nans
    print("---------------- starting to remove nans (↓ before)")
    print(df["label"].value_counts())
    is_nan_idx = df[pd.isnull(df["program"])].index.tolist()
    df.drop(df.index[is_nan_idx], inplace=True)
    print(df["label"].value_counts())
    print("---------------- done nans (↑ after)\n")

    # remove duplicates
    print("---------------- starting to remove duplicates (↓ before)")
    print(df["label"].value_counts())
    df = df.drop_duplicates()
    print(df["label"].value_counts())
    print("---------------- done duplicates (↑ after)\n")
    # print(data.describe())

    # remove really small programs
    print("---------------- starting to remove small programs (↓ before)")
    print(df["label"].value_counts())
    to_drop = []
    for index, row in df.iterrows():
        if len(row["program"]) <= 100:
            to_drop.append(df.index.get_loc(index))
    df = df.drop(df.index[to_drop])
    print(df["label"].value_counts())
    print("---------------- done small programs (↑ after)")

    ############################################################################
    # training
    ############################################################################

    print("\n\ntraining\n\n")

    for i in range(5):

        df = shuffle(df)

        ## downsample uneven classes
        class_0, class_1 = df.label.value_counts()
        df_majority = df[df.label == 0]
        df_minority = df[df.label == 1]
        df_majority_downsampled = resample(
            df_majority, replace=False, n_samples=int(class_1), random_state=42
        )
        df_downsampled = pd.concat([df_majority_downsampled, df_minority])
        data = df_downsampled

        ## make unicode
        print("---------------- starting to make unicode")
        x = data.program.values.astype("U")
        y = data.label
        print("X", len(x), "Y", len(y))
        print("---------------- done unicode\n")

        ## SVM model architecture
        sgd = Pipeline(
            [
                ("vect", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                (
                    "clf",
                    SGDClassifier(
                        loss="hinge",
                        penalty="l2",
                        alpha=1e-3,
                        random_state=42,
                        max_iter=5,
                        tol=-np.infty,
                    ),
                ),
            ]
        )
        skfold = StratifiedKFold(n_splits=10, shuffle=True)
        # each fold has 10% of data withheld for training
        f1_score = model_selection.cross_val_score(
            sgd,
            x,
            y,
            cv=skfold,
            scoring=make_scorer(classification_report_with_accuracy_score),
        )
        # print("f1 score all:\n", f1_score)
        print("\nmean of f1 score:", f1_score.mean())
        overall_f1, overall_accuracy, overall_precision, overall_recall = set_n(
            "get_results"
        )
        # set_n("clear_out") ## unnecessary becuase we loop only once with main
        gc.collect()

        ########################################################################
        # testing on test suite
        ########################################################################

        print("\n\ntest suite\n\n")

        fold_test_suite_results = []  # once per run, not once per fold
        sgd.fit(x, y)

        test_suite = "data/TEXT/test_suite.csv"
        chunk = pd.read_csv(test_suite)

        options = [
            "program_plain",
            "program_jsnice_v2",
            "program_plain_obfuscated",
            "program_plain_obfuscated_jsnice_v2",
        ]
        results = {}

        print(chunk.label.value_counts())

        count = 0

        for option in options:

            false_negative = 0
            false_positive = 0
            true_negative = 0
            true_positive = 0

            for index, row in chunk.iterrows():

                tester = row[option]
                x_predict_array = [tester]
                try:
                    predicted_index = sgd.predict(x_predict_array)

                    if predicted_index != row["label"]:
                        count += 1
                        if predicted_index == 1 and row["label"] == 0:
                            false_positive += 1
                        if predicted_index == 0 and row["label"] == 1:
                            false_negative += 1
                    if predicted_index == row["label"]:
                        if predicted_index == 1 and row["label"] == 1:
                            true_positive += 1
                        if predicted_index == 0 and row["label"] == 0:
                            true_negative += 1

                    # print(count, "/", len(result))
                except Exception as e:
                    print("pass here")
                    print(e)
                    # sys.exit()

            results[option] = {
                "false_negative": false_negative,
                "false_positive": false_positive,
                "true_positive": true_positive,
                "true_negative": true_negative,
            }
        print(results)
        fold_test_suite_results.append(results)

        print("\n\n==================== round complete ====================\n\n")

        print(fold_test_suite_results)
        return (
            overall_f1,
            overall_accuracy,
            overall_precision,
            overall_recall,
            fold_test_suite_results,
        )


# ██████   ██████  ██     ██
# ██   ██ ██    ██ ██     ██
# ██████  ██    ██ ██  █  ██
# ██   ██ ██    ██ ██ ███ ██
# ██████   ██████   ███ ███


def BOW(data):
    """
    starter ideas, credit to:
    https://github.com/MarcoGhise/MachineLearningCarReview/tree/master/jupyter
    """

    ############################################################################
    # some global text pre-processing
    ############################################################################

    print("\n\npreprocessing\n\n")

    # drop nans
    print("---------------- starting to remove nans (↓ before)")
    print(data["label"].value_counts())
    is_nan_idx = data[pd.isnull(data["program"])].index.tolist()
    data.drop(data.index[is_nan_idx], inplace=True)
    print(data["label"].value_counts())
    print("---------------- done nans (↑ after)\n")

    # remove duplicates
    print("---------------- starting to remove duplicates (↓ before)")
    print(data["label"].value_counts())
    data = data.drop_duplicates()
    print(data["label"].value_counts())
    print("---------------- done duplicates (↑ after)\n")

    # remove really small programs
    print("---------------- starting to remove small programs (↓ before)")
    print(data["label"].value_counts())
    to_drop = []
    for index, row in data.iterrows():
        if len(row["program"]) <= 100:
            to_drop.append(data.index.get_loc(index))
    data = data.drop(data.index[to_drop])
    print(data["label"].value_counts())
    print("---------------- done small programs (↑ after)")

    ############################################################################
    # training
    ############################################################################

    print("\n\ntraining\n\n")

    overall_f1 = []
    overall_accuracy = []
    overall_precision = []
    overall_recall = []
    overall_kappa = []
    overall_matrix = []
    overall_test_suite_results = []
    for i in range(1):
        # quick shuffle
        data = shuffle(data)
        ## downsample uneven classes
        class_0, class_1 = data.label.value_counts()
        df_majority = data[data.label == 0]
        df_minority = data[data.label == 1]
        df_majority_downsampled = resample(
            df_majority, replace=False, n_samples=int(class_1), random_state=42
        )
        df_downsampled = pd.concat([df_majority_downsampled, df_minority])
        df = df_downsampled

        ## make unicode
        print("---------------- starting to make unicode")
        x = df.program.values.astype("U")
        y = np.array(df.label)
        # print("X", len(x), "Y", len(y))
        print("---------------- done unicode")

        number_of_splits = 10
        skfold = StratifiedKFold(n_splits=number_of_splits, shuffle=True)
        fold_f1 = []
        fold_accuracy = []
        fold_precision = []
        fold_recall = []
        fold_kappa = []
        fold_matrix = []
        fold_test_suite_results = []
        for train_ix, test_ix in skfold.split(x, y):

            ####################################################################
            # folds
            ####################################################################

            print("\n\nfold", number(), "\n\n")

            print("---------------- starting initializing per-fold examples")
            train_program, test_program = x[train_ix], x[test_ix]
            train_class, test_class = y[train_ix], y[test_ix]
            print("---------------- done initializing")

            ## tokenize
            max_words = 100000
            # max_words = 100
            print("---------------- starting tokenization")
            tokenize = text.Tokenizer(num_words=max_words, char_level=False)
            tokenize.fit_on_texts(train_program)  # only fit on train
            # print("done tokenizing")
            x_train = tokenize.texts_to_matrix(train_program)
            x_test = tokenize.texts_to_matrix(test_program)
            print("---------------- done tokenization")
            # print("done text to matrix")
            ## encode
            print("---------------- starting to encode")
            encoder = LabelEncoder()
            encoder.fit(train_class)
            # print("done fit to train class")
            y_train = encoder.transform(train_class)
            y_test = encoder.transform(test_class)
            print("---------------- done encoding")
            num_classes = np.max(y_train) + 1
            y_test_backend = y_test.copy()
            y_train = utils.to_categorical(y_train, num_classes)
            y_test = utils.to_categorical(y_test, num_classes)

            print("\ny train", len(y_train), "y test", len(y_test))

            # print("done to categorical")
            # batch_size = 64
            epochs = 10
            ## model architecture
            model = Sequential()
            model.add(Dense(512, input_shape=(max_words,)))
            model.add(Activation("relu"))
            model.add(Dropout(0.1))
            model.add(Dense(num_classes))
            model.add(Activation("softmax"))

            ## metrics
            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=[categorical_accuracy],
            )
            history = model.fit(
                x_train,
                y_train,
                validation_split=0.3,
                epochs=epochs,
                verbose=0,
                callbacks=[TqdmCallback(verbose=1)],
            )
            y_softmax = model.predict(x_test)
            y_pred_1d = []
            for i in range(0, len(y_softmax)):
                probs = y_softmax[i]
                predicted_index = np.argmax(probs)
                y_pred_1d.append(predicted_index)
            y_pred_1d = np.array(y_pred_1d)

            print("\ntest length", len(x_test))

            # accuracy: (tp + tn) / (p + n)
            accuracy = accuracy_score(y_test_backend, y_pred_1d)
            print("Accuracy: %f" % accuracy)
            # precision tp / (tp + fp)
            precision = precision_score(y_test_backend, y_pred_1d)
            print("Precision: %f" % precision)
            # recall: tp / (tp + fn)
            recall = recall_score(y_test_backend, y_pred_1d)
            print("Recall: %f" % recall)
            # f1: 2 tp / (2 tp + fp + fn)
            f1_output = f1_score_sklearn(y_test_backend, y_pred_1d)
            print("F1 score: %f" % f1_output)
            # kappa
            kappa = cohen_kappa_score(y_test_backend, y_pred_1d)
            print("Cohens kappa: %f" % kappa)
            # confusion matrix
            matrix = confusion_matrix(y_test_backend, y_pred_1d)
            print(matrix)

            fold_f1.append(f1_output)
            fold_accuracy.append(accuracy)
            fold_precision.append(precision)
            fold_recall.append(recall)
            fold_kappa.append(kappa)
            fold_matrix.append(matrix)

            ####################################################################
            # testing on test suite
            ####################################################################

            print("\n\ntest suite\n\n")

            chunksize = 500000000000000
            test_suite = "data/TEXT/test_suite.csv"

            for chunk in pd.read_csv(test_suite, chunksize=chunksize):

                count = 0
                result = chunk

                options = [
                    "program_plain",
                    "program_jsnice_v2",
                    "program_plain_obfuscated",
                    "program_plain_obfuscated_jsnice_v2",
                ]
                results = {}

                for option in options:

                    false_negative = 0
                    false_positive = 0
                    true_negative = 0
                    true_positive = 0

                    for index, row in result.iterrows():

                        tester = row[option]
                        x_predict_array = [tester]
                        try:
                            x_predict = tokenize.texts_to_matrix(x_predict_array)
                            prediction = model.predict(x_predict, verbose=0)
                            predicted_index = np.argmax(prediction)

                            if predicted_index != row["label"]:
                                count += 1
                                if predicted_index == 1 and row["label"] == 0:
                                    false_positive += 1
                                if predicted_index == 0 and row["label"] == 1:
                                    false_negative += 1
                            if predicted_index == row["label"]:
                                if predicted_index == 1 and row["label"] == 1:
                                    true_positive += 1
                                if predicted_index == 0 and row["label"] == 0:
                                    true_negative += 1

                            # print(count, "/", len(result))
                        except Exception as e:
                            print("pass here")
                            print(e)
                            # sys.exit()

                    results[option] = {
                        "false_negative": false_negative,
                        "false_positive": false_positive,
                        "true_positive": true_positive,
                        "true_negative": true_negative,
                    }
                # print("number incorrect", count, "FN", false_negative, "FP", false_positive,  "TP", true_positive, "TN", true_negative)
                # print(count / len(result))
                print("results per fold, to be averaged later:\n", results)
                fold_test_suite_results.append(results)

            gc.collect()

        print("\n\n==================== round complete ====================\n\n")

        print(fold_f1)
        overall_f1.append(fold_f1)
        overall_accuracy.append(fold_accuracy)
        overall_precision.append(fold_precision)
        overall_recall.append(fold_recall)
        overall_kappa.append(fold_kappa)
        overall_matrix.append(fold_matrix)
        overall_test_suite_results.append(fold_test_suite_results)
        return (
            overall_f1,
            overall_accuracy,
            overall_precision,
            overall_recall,
            overall_kappa,
            overall_matrix,
            overall_test_suite_results,
        )


# ███████ ███    ███ ██████  ███████ ██████  ██████  ██ ███    ██  ██████
# ██      ████  ████ ██   ██ ██      ██   ██ ██   ██ ██ ████   ██ ██
# █████   ██ ████ ██ ██████  █████   ██   ██ ██   ██ ██ ██ ██  ██ ██   ███
# ██      ██  ██  ██ ██   ██ ██      ██   ██ ██   ██ ██ ██  ██ ██ ██    ██
# ███████ ██      ██ ██████  ███████ ██████  ██████  ██ ██   ████  ██████


def embedding(data):

    """
    likely improvements
    - downsampling rate (or better yet, use something like SMOTE)
    - tokenization number of words (max_words)
    - model architecture
    - GloVe package (100d or 300d) or better embedding (source code)
    """

    ############################################################################
    # some global text pre-processing
    ############################################################################

    print("\n\npreprocessing\n\n")

    # drop nans
    print("---------------- starting to remove nans (↓ before)")
    print(data["label"].value_counts())
    is_nan_idx = data[pd.isnull(data["program"])].index.tolist()
    data.drop(data.index[is_nan_idx], inplace=True)
    print(data["label"].value_counts())
    print("---------------- done nans (↑ after)\n")

    # remove duplicates
    print("---------------- starting to remove duplicates (↓ before)")
    print(data["label"].value_counts())
    data.drop_duplicates(inplace=True)
    print(data["label"].value_counts())
    print("---------------- done duplicates (↑ after)\n")
    # print(data.describe())

    # remove really small programs
    print("---------------- starting to remove small programs (↓ before)")
    print(data["label"].value_counts())
    to_drop = []
    for index, row in data.iterrows():
        if len(row["program"]) <= 100:
            to_drop.append(data.index.get_loc(index))
    data.drop(data.index[to_drop], inplace=True)
    print(data["label"].value_counts())
    print("---------------- done small programs (↑ after)")

    ############################################################################
    # training
    ############################################################################

    print("\n\ntraining\n\n")

    overall_f1 = []
    overall_accuracy = []
    overall_precision = []
    overall_recall = []
    overall_kappa = []
    # overall_auc = []
    overall_matrix = []
    overall_test_suite_results = []
    for i in range(1):
        # quick shuffle
        data = shuffle(data)

        ## downsample uneven classes
        class_0, class_1 = data.label.value_counts()
        df_majority = data[data.label == 0]
        df_minority = data[data.label == 1]
        df_majority_downsampled = resample(
            df_majority, replace=False, n_samples=int(class_1 * 1), random_state=42
        )
        df_downsampled = pd.concat([df_majority_downsampled, df_minority])
        df = df_downsampled

        ## make unicode
        print("---------------- starting to make unicode")
        x = df.program.values.astype("U")
        y = np.array(df.label)
        print("X", len(x), "Y", len(y))
        print("---------------- done unicode")

        number_of_splits = 10
        skfold = StratifiedKFold(n_splits=number_of_splits, shuffle=True)
        # kfold = KFold(n_splits=number_of_splits, shuffle=True)
        fold_f1 = []
        fold_accuracy = []
        fold_precision = []
        fold_recall = []
        fold_kappa = []
        fold_matrix = []
        fold_test_suite_results = []
        for train_ix, test_ix in skfold.split(x, y):

            ####################################################################
            # folds
            ####################################################################

            print("\n\nfold", number(), "\n\n")

            print("---------------- starting initialization per-fold examples")
            train_program, test_program = x[train_ix], x[test_ix]
            train_class, test_class = y[train_ix], y[test_ix]
            print("---------------- done initializing")

            ## tokenize
            max_words = 100000
            # max_words = 100
            print("---------------- starting tokenization")
            tokenize = text.Tokenizer(num_words=max_words, char_level=False)
            tokenize.fit_on_texts(train_program)  # only fit on train
            vocab_size = len(tokenize.word_index) + 1
            print("---------------- done tokenization")

            print("---------------- starting encoding")
            encoded_docs = tokenize.texts_to_sequences(train_program)
            encoded_docs_test = tokenize.texts_to_sequences(test_program)
            print("---------------- done encoding")

            ## GloVe embedding
            print("---------------- starting GloVe")
            embeddings_index = {}
            f = open(r"data/TEXT/glove.42B.300d.txt", encoding="utf8")
            for line in f:
                values = line.split()
                word = "".join(values[:-300])
                coefs = np.asarray(values[-300:], dtype="float32")
                embeddings_index[word] = coefs
            f.close()
            embedding_matrix = np.zeros((vocab_size, 300))
            for word, index in tokenize.word_index.items():
                if index > vocab_size - 1:
                    break
                else:
                    embedding_vector = embeddings_index.get(word)
                    if embedding_vector is not None:
                        embedding_matrix[index] = embedding_vector
            # embeddings_index = None
            print("---------------- done GloVe")

            ## add padding
            # max_length = int(vocab_size * .001) # testing
            max_length = 1000  # performance on adversarial perspective
            # max_length = 100  # performance on test suite
            print("---------------- starting padding")
            padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding="post")
            padded_docs_test = pad_sequences(
                encoded_docs_test, maxlen=max_length, padding="post"
            )
            print("---------------- done padding")

            ## encode
            print("---------------- starting to encode")
            encoder = LabelEncoder()
            encoder.fit(train_class)
            # print("done fit to train class")
            y_train = encoder.transform(train_class)
            y_test = encoder.transform(test_class)
            print("---------------- done encoding\n")
            y_test_backend = y_test.copy()
            # print("done encoder.transform")
            num_classes = np.max(y_train) + 1
            y_train = utils.to_categorical(y_train, num_classes)
            y_test = utils.to_categorical(y_test, num_classes)

            print("y train", len(y_train), "y test", len(y_test))

            epochs = 10

            ## model2 architecture
            model = Sequential()
            model.add(
                Embedding(
                    vocab_size,
                    300,
                    input_length=max_length,
                    weights=[embedding_matrix],
                    trainable=True,
                )
            )

            ## option-1 for paper
            model.add(Conv1D(64, 5, activation="relu"))
            model.add(MaxPooling1D(pool_size=4))
            model.add(LSTM(300, activation="sigmoid"))
            model.add(Dense(2))
            model.add(Activation("softmax"))
            # model.add(Dense(1, activation='sigmoid'))

            # ## option-2 possible better adversarial training (also change downsampling to 1.5)
            # model.add(SpatialDropout1D(0.2))
            # model.add(Bidirectional(LSTM(128, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)))
            # model.add(Bidirectional(LSTM(64, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)))
            # model.add(Conv1D(64, 4))
            # model.add(GlobalMaxPool1D())
            # model.add(Dense(2, activation='relu'))
            # model.add(Activation('softmax'))

            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=[categorical_accuracy],
            )
            # fit the model
            history = model.fit(
                padded_docs,
                y_train,
                validation_split=0.3,
                epochs=epochs,
                verbose=0,
                callbacks=[TqdmCallback(verbose=1)],
            )

            y_softmax = model.predict(padded_docs_test)
            y_pred_1d = []
            y_pred_percentages = []
            for i in range(0, len(y_softmax)):
                probs = y_softmax[i]
                y_pred_percentages.append(
                    np.amax(probs)
                )  # Returns the indices of the maximum values along an axis
                predicted_index = np.argmax(probs)
                y_pred_1d.append(predicted_index)

            print("test length", len(padded_docs_test))

            y_pred_1d = np.array(y_pred_1d)

            # accuracy: (tp + tn) / (p + n)
            accuracy = accuracy_score(y_test_backend, y_pred_1d)
            print("Accuracy: %f" % accuracy)
            # precision tp / (tp + fp)
            precision = precision_score(y_test_backend, y_pred_1d)
            print("Precision: %f" % precision)
            # recall: tp / (tp + fn)
            recall = recall_score(y_test_backend, y_pred_1d)
            print("Recall: %f" % recall)
            # f1: 2 tp / (2 tp + fp + fn)
            f1_output = f1_score_sklearn(y_test_backend, y_pred_1d)
            print("F1 score: %f" % f1_output)
            # kappa
            kappa = cohen_kappa_score(y_test_backend, y_pred_1d)
            print("Cohens kappa: %f" % kappa)
            # confusion matrix
            matrix = confusion_matrix(y_test_backend, y_pred_1d)
            print(matrix)

            fold_f1.append(f1_output)
            fold_accuracy.append(accuracy)
            fold_precision.append(precision)
            fold_recall.append(recall)
            fold_kappa.append(kappa)
            fold_matrix.append(matrix)

            ####################################################################
            # testing on test suite
            ####################################################################

            print("\n\ntest suite\n\n")

            chunksize = 500000000000000
            test_suite = "data/TEXT/test_suite.csv"

            for chunk in pd.read_csv(test_suite, chunksize=chunksize):

                count = 0
                result = chunk

                options = [
                    "program_plain",
                    "program_jsnice_v2",
                    "program_plain_obfuscated",
                    "program_plain_obfuscated_jsnice_v2",
                ]
                results = {}

                print(result.label.value_counts())

                for option in options:

                    false_negative = 0
                    false_positive = 0
                    true_negative = 0
                    true_positive = 0

                    for index, row in result.iterrows():

                        tester = row[option]
                        x_predict_array = [tester]
                        try:
                            x_predict = tokenize.texts_to_sequences(x_predict_array)
                            x_predict = pad_sequences(
                                x_predict, maxlen=max_length, padding="post"
                            )
                            prediction = model.predict(x_predict, verbose=0)
                            predicted_index = np.argmax(prediction)

                            if predicted_index != row["label"]:
                                count += 1
                                if predicted_index == 1 and row["label"] == 0:
                                    false_positive += 1
                                if predicted_index == 0 and row["label"] == 1:
                                    false_negative += 1
                            if predicted_index == row["label"]:
                                if predicted_index == 1 and row["label"] == 1:
                                    true_positive += 1
                                if predicted_index == 0 and row["label"] == 0:
                                    true_negative += 1

                        except Exception as e:
                            print("pass here")
                            print(e)
                            # sys.exit()

                    results[option] = {
                        "false_negative": false_negative,
                        "false_positive": false_positive,
                        "true_positive": true_positive,
                        "true_negative": true_negative,
                    }
                print(results)
                fold_test_suite_results.append(results)

            gc.collect()

        print(fold_f1)
        overall_f1.append(fold_f1)
        overall_accuracy.append(fold_accuracy)
        overall_precision.append(fold_precision)
        overall_recall.append(fold_recall)
        overall_kappa.append(fold_kappa)
        overall_matrix.append(fold_matrix)
        overall_test_suite_results.append(fold_test_suite_results)
        print(
            overall_f1,
            overall_accuracy,
            overall_precision,
            overall_recall,
            overall_kappa,
            overall_matrix,
            overall_test_suite_results,
        )
        return (
            overall_f1,
            overall_accuracy,
            overall_precision,
            overall_recall,
            overall_kappa,
            overall_matrix,
            overall_test_suite_results,
        )


def main():
    global MODEL, CORPUS, CHUNK_SIZE
    arg = argparse.ArgumentParser(description="machine learning models!")
    arg.add_argument(
        "--model", action="store", dest="model", help="select: svm, bow, embedding"
    )
    arg.add_argument(
        "--corpus", action="store", dest="corpus", help="select: plaintext, jsnice"
    )
    options = arg.parse_args()
    if options.model:
        if (
            options.model == "svm"
            or options.model == "bow"
            or options.model == "embedding"
        ):
            try:
                MODEL = options.model
            except Exception:
                sys.exit("Error on argparse selecting model")
        else:
            sys.exit(
                "[-] invalid model selection. Please pick 'svm' or 'bow' or 'embedding'"
            )
    if options.corpus:
        if options.corpus == "plaintext" or options.corpus == "jsnice":
            try:
                if options.corpus == "plaintext":
                    CORPUS = "data/TEXT/plain_examples.csv"
                if options.corpus == "jsnice":
                    CORPUS = "data/TEXT/jsnice_examples_v2.csv"
            except Exception:
                sys.exit("Error on argparse selecting corpus")
        else:
            sys.exit(
                "[-] invalid corpus (text to train on) selection. Please pick 'plaintext' or 'jsnice'"
            )

    name = "final_outputs/" + MODEL + "--" + str(options.corpus) + ".csv"

    print("model selected  ==>", MODEL)
    print("corpus selected ==>", CORPUS)
    print("output selected ==>", name)

    chunk = pd.read_csv(CORPUS)
    data = chunk
    if MODEL == "embedding":
        (
            overall_f1,
            overall_accuracy,
            overall_precision,
            overall_recall,
            overall_kappa,
            overall_matrix,
            overall_test_suite_results,
        ) = embedding(data)
    if MODEL == "bow":
        (
            overall_f1,
            overall_accuracy,
            overall_precision,
            overall_recall,
            overall_kappa,
            overall_matrix,
            overall_test_suite_results,
        ) = BOW(data)
    if MODEL == "svm":
        (
            overall_f1,
            overall_accuracy,
            overall_precision,
            overall_recall,
            overall_test_suite_results,
        ) = svm(data)

    # take average of the fold scores, then take average of all 10 runs
    print("\n================== results ==================\n")
    print("f1 fold scores", overall_f1, np.array(overall_f1).mean(), "\n\n")
    print(
        "accuracy fold scores",
        overall_accuracy,
        np.array(overall_accuracy).mean(),
        "\n\n",
    )
    print(
        "precision fold scores",
        overall_precision,
        np.array(overall_precision).mean(),
        "\n\n",
    )
    print("recall fold scores", overall_recall, np.array(overall_recall).mean(), "\n\n")
    print("test suite results per fold", overall_test_suite_results, "\n\n")

    temp = {}
    list = []
    temp["f1_original"] = np.array(overall_f1).mean()
    temp["accuracy_original"] = np.array(overall_accuracy).mean()
    temp["precision_original"] = np.array(overall_precision).mean()
    temp["recall_original"] = np.array(overall_recall).mean()
    # temp['kappa_original'] = np.array(overall_kappa).mean()

    options = [
        "program_plain",
        "program_jsnice_v2",
        "program_plain_obfuscated",
        "program_plain_obfuscated_jsnice_v2",
    ]
    for obj in overall_test_suite_results:  # run level
        fold_precision = []
        fold_recall = []
        fold_accuracy = []
        fold_f1 = []
        for option in options:
            if MODEL == "svm":

                true_positive = obj[option]["true_positive"]
                true_negative = obj[option]["true_negative"]
                false_positive = obj[option]["false_positive"]
                false_negative = obj[option]["false_negative"]

                if true_positive == 0 and (true_positive + false_positive) == 0:
                    precision = 0
                else:
                    precision = true_positive / (true_positive + false_positive)
                recall = true_positive / (true_positive + false_negative)
                accuracy = (true_positive + true_negative) / (
                    true_positive + false_positive + false_negative + true_negative
                )
                if (precision + recall) == 0:
                    f1 = 0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)

                print("f1", f1)
                print("acc", accuracy)
                print("precision", precision)
                print("recall", recall)
                print("==========")

                fold_precision.append(precision)
                fold_recall.append(recall)
                fold_accuracy.append(accuracy)
                fold_f1.append(f1)

                option_name = str(option)
                temp["f1_" + option_name] = np.array(fold_f1).mean()
                temp["accuracy_" + option_name] = np.array(fold_accuracy).mean()
                temp["precision_" + option_name] = np.array(fold_precision).mean()
                temp["recall_" + option_name] = np.array(fold_recall).mean()

            else:
                for item in obj:  # fold level
                    true_positive = item[option]["true_positive"]
                    true_negative = item[option]["true_negative"]
                    false_positive = item[option]["false_positive"]
                    false_negative = item[option]["false_negative"]

                    if true_positive == 0 and (true_positive + false_positive) == 0:
                        precision = 0
                    else:
                        precision = true_positive / (true_positive + false_positive)
                    recall = true_positive / (true_positive + false_negative)
                    accuracy = (true_positive + true_negative) / (
                        true_positive + false_positive + false_negative + true_negative
                    )
                    if (precision + recall) == 0:
                        f1 = 0
                    else:
                        f1 = 2 * (precision * recall) / (precision + recall)

                    # print("f1", f1)
                    # print("acc", accuracy)
                    # print("precision", precision)
                    # print("recall", recall)
                    # print("==========")

                    fold_precision.append(precision)
                    fold_recall.append(recall)
                    fold_accuracy.append(accuracy)
                    fold_f1.append(f1)
                option_name = str(option)
                temp["f1_" + option_name] = np.array(fold_f1).mean()
                temp["accuracy_" + option_name] = np.array(fold_accuracy).mean()
                temp["precision_" + option_name] = np.array(fold_precision).mean()
                temp["recall_" + option_name] = np.array(fold_recall).mean()
    list.append(temp)
    df = pd.DataFrame(list)
    if not os.path.isfile(name):
        df.to_csv(name, header=True, index=False)
    else:
        df.to_csv(name, mode="a", header=False, index=False)
    df = pd.read_csv(name)
    print("saved==>", name)
    print(df.head())
    gc.collect()


if __name__ == "__main__":

    main()
