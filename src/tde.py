import re
import gc
import numpy as np
import multiprocessing
from multiprocessing import Pool


class TreeDrivenEncoder:
    """
    TreeDrivenEncoder algorithm.

    OOP is beautiful
    """

    def __init__(self):
        self.all_conditions = None

    def fit(self, trees):
        self.all_conditions = []
        node_data = {}

        for tree_number, tree in enumerate(trees):
            tree_nodes = tree.split('\n\t')
            for node in tree_nodes:
                cleaned_node = re.sub(r'\s+', '', node)
                node_id = int(cleaned_node.split(':')[0])
                node_info = cleaned_node.split(':')[1]

                if 'leaf' not in node_info:
                    key = (tree_number, node_id)
                    node_data[key] = self.get_node_data(node_info)

        for key in sorted(node_data.keys()):
            feature, value, _ = node_data[key]
            if value != 0:
                self.all_conditions.append([feature, value])

    def transform(self, data):
        data_array = np.array(data)
        with Pool(processes=max(1, multiprocessing.cpu_count() - 2)) as pool:
            encoded_data = pool.map(self._dt2v, data_array)
        
        gc.collect()
        return np.array(encoded_data)

    def _dt2v(self, sample, debug=False):
        node_vector = []
        for feature, value in self.all_conditions:
            if debug: print(feature, value)
            node_vector.append(int(sample[feature] < float(value)))

        if debug: print(node_vector)
        return node_vector

    @staticmethod
    def get_node_data(node_info):
        condition = re.findall(r'\[([^]]*)\]', node_info)
        feature, value = re.split(r'[`=<>]', condition[0])
        feature = int(feature[1:])
        value = float(value)
        return [feature, value, 0]
