import json
import pathlib

import networkx as nx
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import BernoulliNB

from hfs.data_utils import create_mapping_columns_to_nodes
from hfs.preprocessing import HierarchicalPreprocessor
from hfs.selectors import HIP, HNB, MR, RNB, TAN, HNBs


def data():
    dir = pathlib.Path(__file__).parent.parent.absolute()
    rel = pathlib.Path("hfs/data/go_digraph2.gml")
    path = dir / rel
    graph = nx.read_gml(path)
    rel = pathlib.Path("hfs/data/train_test.csv")
    path = dir / rel
    df = pd.read_csv(path)
    columns = create_mapping_columns_to_nodes(df, graph)
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    rel = pathlib.Path("hfs/data/gene2gomod3.txt")
    path = dir / rel
    g2g = pd.read_csv(path, sep=",")
    go_terms = g2g["GO_ID"].unique()
    y_train = train["longevity influence"].to_numpy()
    y_test = test["longevity influence"].to_numpy()
    columns = create_mapping_columns_to_nodes(train, graph)
    train = train[go_terms].to_numpy()
    test = test[go_terms].to_numpy()
    hierarchy = nx.to_numpy_array(graph)
    return (hierarchy, train, y_train, test, y_test, columns)


def hnb(hierarchy, train, y_train, test, y_test, k, columns, path):
    model = HNB(hierarchy=hierarchy, k=k)
    model.fit_selector(X_train=train, y_train=y_train, X_test=test, columns=columns)
    pred = model.select_and_predict(predict=True, saveFeatures=True)
    score = model.get_score(y_test, pred)
    with open(path, "a") as file:
        file.write("\nHNB:\n")
        file.write(json.dumps(score))


def hnbs(hierarchy, train, y_train, test, y_test, k, columns, path):
    model = HNBs(hierarchy=hierarchy)
    model.fit_selector(X_train=train, y_train=y_train, X_test=test, columns=columns)
    pred = model.select_and_predict(predict=True, saveFeatures=True)
    score = model.get_score(y_test, pred)
    with open(path, "a") as file:
        file.write("\nHNBs:\n")
        file.write(json.dumps(score))


def rnb(hierarchy, train, y_train, test, y_test, k, columns, path):
    model = RNB(hierarchy=hierarchy, k=k)
    model.fit_selector(X_train=train, y_train=y_train, X_test=test, columns=columns)
    pred = model.select_and_predict(predict=True, saveFeatures=True)
    score = model.get_score(y_test, pred)
    with open(path, "a") as file:
        file.write("\nRNB:\n")
        file.write(json.dumps(score))


def mr(hierarchy, train, y_train, test, y_test, k, columns, path):
    model = MR(hierarchy=hierarchy)
    model.fit_selector(X_train=train, y_train=y_train, X_test=test, columns=columns)
    pred = model.select_and_predict(predict=True, saveFeatures=True)
    score = model.get_score(y_test, pred)
    with open(path, "a") as file:
        file.write("\nMR:\n")
        file.write(json.dumps(score))


def tan(hierarchy, train, y_train, test, y_test, k, columns, path):
    model = TAN(hierarchy=hierarchy)
    model.fit_selector(X_train=train, y_train=y_train, X_test=test, columns=columns)
    pred = model.select_and_predict(predict=True, saveFeatures=True)
    score = model.get_score(y_test, pred)
    with open(path, "a") as file:
        file.write("\nTAN:\n")
        file.write(json.dumps(score))


def hip(hierarchy, train, y_train, test, y_test, k, columns, path):
    model = HIP(hierarchy=hierarchy)
    model.fit_selector(X_train=train, y_train=y_train, X_test=test, columns=columns)
    pred = model.select_and_predict(predict=True, saveFeatures=True)
    score = model.get_score(y_test, pred)
    with open(path, "a") as file:
        file.write("\nHIP:\n")
        file.write(json.dumps(score))


def naive_bayes(hierarchy, train, y_train, test, y_test, k, columns, path):
    clf = BernoulliNB()
    clf.fit(train, y_train)
    predictions = clf.predict(test)
    score = classification_report(y_true=y_test, y_pred=predictions, output_dict=True)
    with open(path, "a") as file:
        file.write("\nBaseline:\n")
        file.write(json.dumps(score))


# Evalueate feature selection of HNB
def evaluate(data, k):
    hierarchy, train, y_train, test, y_test, columns = data()
    preprocessor = HierarchicalPreprocessor(hierarchy=hierarchy)
    preprocessor.fit(train, columns=columns)
    train = preprocessor.transform(train)
    test = preprocessor.transform(test)

    hierarchy = preprocessor.get_hierarchy()
    graph = nx.DiGraph(hierarchy)
    columns = create_mapping_columns_to_nodes(pd.DataFrame(train), graph)

    dir = pathlib.Path(__file__).parent.parent.absolute()
    rel = pathlib.Path(f"hfs/results/new/all_{k}.txt")
    path = dir / rel
    for function in [tan, mr, rnb, hnbs, hnb]:
        function(
            hierarchy=hierarchy,
            train=train,
            y_train=y_train,
            test=test,
            y_test=y_test,
            k=k,
            columns=columns,
            path=path,
        )


if __name__ == "__main__":
    evaluate(data, 20)
