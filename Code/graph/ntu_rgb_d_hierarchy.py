from audioop import reverse
import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 16

class Graph:
    def __init__(self, CoM=8, labeling_mode='spatial'):
        self.num_node = num_node
        self.CoM = CoM
        self.A = self.get_adjacency_matrix(labeling_mode)
        #print("a",self.A)
    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial': #获取层次结构图
            A = tools.get_hierarchical_graph(num_node, tools.get_edgeset(dataset='NTU', CoM=self.CoM)) # L, 3, 25, 25
            #print("get_adjacency_matrix",A.shape)
        else:
            raise ValueError()
        return A, self.CoM


if __name__ == '__main__':
    import tools
    g = Graph().A
    import matplotlib.pyplot as plt
    for i, g_ in enumerate(g[0]):
        plt.imshow(g_, cmap='gray')
        cb = plt.colorbar()
        plt.savefig('./graph_{}.png'.format(i))
        cb.remove()
        plt.show()