import re
import numpy as np
from typing import List
from collections import defaultdict, Counter

node_info_re = re.compile(r"f(\d+)<([0-9\.]+)")


def parse_node_info(node_info: str):
    matched = node_info_re.findall(node_info)
    if not matched:
        return None, None
    feature, threshold = matched[0]
    return int(feature), float(threshold)


def extract_node_data_from_tree(tree_number, tree):
    node_data = defaultdict(dict)
    for node in tree.split("\n"):
        cleaned_node = re.sub(r"\s+", "", node)
        if ":" not in cleaned_node:
            continue
        node_id, node_info = map(str.strip, cleaned_node.split(":"))
        feature, threshold = parse_node_info(node_info)
        if feature is not None and threshold is not None:
            node_data[(tree_number, int(node_id))] = {
                "feature": feature,
                "threshold": threshold,
            }
    return node_data


class TreeDrivenEncoder:
    def __init__(self, min_freq=2):
        self.all_conditions = []
        self.min_freq = min_freq

    def fit(self, trees: List[str]):
        node_data = defaultdict(dict)
        for tree_number, tree in enumerate(trees):
            node_data.update(extract_node_data_from_tree(tree_number, tree))

        feature_counter = Counter(data["feature"] for data in node_data.values())

        self.all_conditions = [
            data
            for key, data in sorted(node_data.items())
            if feature_counter[data["feature"]] >= self.min_freq
        ]

    def transform(self, X):
        X = np.array(X)
        
        # Validate input
        if X.ndim != 2:
            raise ValueError("Input X must be a 2-dimensional array")
        if not self.all_conditions:
            raise ValueError("TreeDrivenEncoder has not been fitted. Call fit() first.")
            
        # Get the maximum feature index used in conditions
        max_feature_idx = max(condition["feature"] for condition in self.all_conditions)
        if X.shape[1] <= max_feature_idx:
            raise ValueError(f"Input X has {X.shape[1]} features, but the encoder requires at least {max_feature_idx + 1} features")
            
        n_conditions = len(self.all_conditions)
        encoded_X = np.zeros((X.shape[0], n_conditions), dtype=int)
        
        try:
            for i, condition in enumerate(self.all_conditions):
                encoded_X[:, i] = (
                    X[:, condition["feature"]] < condition["threshold"]
                ).astype(int)
        except Exception as e:
            raise RuntimeError(f"Error during transformation: {str(e)}")
            
        return encoded_X
