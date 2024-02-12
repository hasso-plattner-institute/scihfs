import time

import networkx as nx
import wandb
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

from hfs.data_utils import create_mapping_columns_to_nodes, load_data
from hfs.preprocessing import HierarchicalPreprocessor
from hfs.selectors import (
    BottomUpSelector,
    GreedyTopDownSelector,
    SHSELSelector,
    TopDownSelector,
    TSELSelector,
)

from .experiments import data


def preprocess_data(hierarchy, X_train, X_test, columns):
    preprocessor = HierarchicalPreprocessor(hierarchy)
    preprocessor.fit(X_train, columns=columns)
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    hierarchy_updated = preprocessor.get_hierarchy()
    columns_updated = preprocessor.get_columns()
    return X_train_transformed, X_test_transformed, hierarchy_updated, columns_updated


def get_gene_data():
    return data()


def get_tweet_data():
    X, y, hierarchy = load_data()
    columns = create_mapping_columns_to_nodes(X, hierarchy)
    X = X.to_numpy()
    hierarchy = nx.to_numpy_array(hierarchy)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    return hierarchy, X_train, y_train, X_test, y_test, columns


def get_data(dataset: str):
    if dataset == "gene":
        return get_gene_data()
    if dataset == "tweets":
        return get_tweet_data()
    else:
        raise ValueError("Invalid dataset")


def shsel(X, y, X_test, hierarchy, columns):
    print("SHSEL Feature Selection")
    selector = SHSELSelector(hierarchy)
    selector.fit(X, y, columns=columns)
    X_transformed = selector.transform(X)
    X_test_transformed = selector.transform(X_test)
    return X_transformed, X_test_transformed


def hfe(X, y, X_test, hierarchy, columns):
    print("HFE Feature Selection")
    selector = SHSELSelector(
        hierarchy, relevance_metric="Correlation", use_hfe_extension=True
    )
    selector.fit(X, y, columns=columns)
    X_transformed = selector.transform(X)
    X_test_transformed = selector.transform(X_test)
    return X_transformed, X_test_transformed


def tsel(X, y, X_test, hierarchy, columns):
    print("TSEL Feature Selection")
    selector = TSELSelector(hierarchy)
    selector.fit(X, y, columns=columns)
    X_transformed = selector.transform(X)
    X_test_transformed = selector.transform(X_test)
    return X_transformed, X_test_transformed


def top_down(X, y, X_test, hierarchy, columns):
    print("Hill Climbing Top Down Feature Selection")
    selector = TopDownSelector(hierarchy)
    selector.fit(X, y, columns=columns)
    X_transformed = selector.transform(X)
    X_test_transformed = selector.transform(X_test)
    return X_transformed, X_test_transformed


def bottom_up(X, y, X_test, hierarchy, columns):
    print("Hill Climbing Bottom Up Feature Selection")
    selector = BottomUpSelector(hierarchy)
    selector.fit(X, y, columns=columns)
    X_transformed = selector.transform(X)
    X_test_transformed = selector.transform(X_test)
    return X_transformed, X_test_transformed


def gtd(X, y, X_test, hierarchy, columns):
    print("Greedy Top Down Feature Selection")
    selector = GreedyTopDownSelector(hierarchy)
    selector.fit(X, y, columns=columns)
    X_transformed = selector.transform(X)
    X_test_transformed = selector.transform(X_test)
    return X_transformed, X_test_transformed


def baseline(X, y, X_test, hierarchy, columns):
    print("Baseline (without feature selection)")
    return X, X_test


def classify(X_train, y_train, X_test, y_test, classifier):
    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)
    return accuracy


def initialize_wandb(experiment: str, dataset: str):
    wandb.init(
        name=experiment,
        project="hfs",
        group=dataset,
        config={"experiment": experiment, "dataset": "subset300"},
    )


def get_experiment():
    return {
        "shsel": shsel,
        "hfe": hfe,
        "tsel": tsel,
        "bottom_up": bottom_up,
        "top_down": top_down,
        "gtd": gtd,
        "baseline": baseline,
    }


def classification_experiments(
    use_wandb=False,
    experiments: list[str] = [
        "baseline",
        "shsel",
        "hfe",
        "tsel",
        "gtd",
        "bottom_up",
        "top_down",
    ],
    dataset="tweets",
):
    classifier = BernoulliNB()
    hierarchy, X_train, y_train, X_test, y_test, columns = get_data(dataset)
    X_train, X_test, hierarchy, columns = preprocess_data(
        hierarchy, X_train, X_test, columns
    )

    for experiment_name in experiments:
        if use_wandb:
            initialize_wandb(experiment_name, dataset)
        experiment = get_experiment()[experiment_name]
        start_time = time.time()
        X_train_transformed, X_test_transformed = experiment(
            X_train, y_train, X_test, hierarchy, columns
        )
        transform_time = time.time()
        accuracy = classify(
            X_train_transformed, y_train, X_test_transformed, y_test, classifier
        )
        classfiy_time = time.time()

        # Calculate metrics
        compression_rate = X_train_transformed.shape[1] / X_train.shape[1]
        preprocess_time = transform_time - start_time
        num_features = X_train_transformed.shape[1]
        classify_time = classfiy_time - transform_time

        # Print metrics
        print(f"Accuracy: {accuracy}")
        print(f"Compression rate: {compression_rate}")

        # Loggging
        if use_wandb:
            wandb.log(
                {
                    "accuracy": accuracy,
                    "compression_rate": compression_rate,
                    "preprocess_time": preprocess_time,
                    "classify_time": classify_time,
                    "num_features": num_features,
                }
            )
            wandb.finish()
