import re 
import gc
import numpy as np
import multiprocessing
from multiprocessing import Pool



class TreeDrivenEncoder:
    '''
    TreeDrivenEncoder algorithm. 

    OOP is beautiful
    '''
    def __init__(self):
        self.all_conditions = None

    def fit(self, trees):
        self.all_conditions = []
        d = {}
        len_same = []
        for tree_number, i_tree in enumerate(trees):
            len_same.append(len(i_tree.split('\n\t')))
            for i_node in i_tree.split('\n\t'):
                    raw = re.sub('\s+', '', i_node)
                    node_id = int(raw.split(':')[0])
                    if 'leaf' not in raw.split(':')[1]:
                        d[(tree_number,node_id)] = self.get_node_data(raw.split(':')[1])
    
        for i in sorted(d.keys()):
            if d[i][1] != 0:
                self.all_conditions.append([d[i][0], d[i][1]])

    def transform(self, data):
        data = np.array(data)
        pool = Pool(multiprocessing.cpu_count()-2)
        encoded_data = pool.map(self._dt2v, data)
        pool.close()
        del pool
        gc.collect()
        return np.array(encoded_data)#.astype(np.uint8)

    def _dt2v(self, sample, debug=False):
        if debug: print(self.all_conditions)
        node_vector = []
        feature_vector = []
        for i_cond in self.all_conditions:
            feature, value = i_cond[0], i_cond[1]
            if debug: print(feature, value)
            if sample[feature] < float(value):
                node_vector.append(1)
            else:
                node_vector.append(0)
        if debug: print(node_vector)
        return node_vector
    
    @staticmethod
    def get_node_data(text_inf):
        condition = re.findall(r'\[([^]]*)\]', text_inf)
        #print(condition)
        feature, value = re.split(r'[`=<>]', condition[0])
        #print(feature)
        feature = int(feature[1:])
        value = float(value)
        return [feature, value, 0]