import numpy as np


def edge2mat(link, num_node):  # 构建自身矩阵
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, hierarchy):
    A = []
    for i in range(len(hierarchy)):
        A.append(normalize_digraph(edge2mat(hierarchy[i], num_node)))

    A = np.stack(A)

    return A


def get_spatial_graph_original(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def get_graph(num_node, edges):
    I = edge2mat(edges[0], num_node)
    Forward = normalize_digraph(edge2mat(edges[2], num_node))
    Reverse = normalize_digraph(edge2mat(edges[1], num_node))
    A = np.stack((I, Forward, Reverse))
    #print("get_graph", A.shape)  # AA (3, 16, 16)
    return A  # 3, 25, 25


def get_hierarchical_graph(num_node, edges):  # 获取层次结构图
    A = []
    #print("edges", edges)
    for edge in edges:
        A.append(get_graph(num_node, edge))  # 调用 get_graph 函数，根据当前边集合生成对应的邻接矩阵，并将其添加到列表 A 中。
        #print("edge", edge)
    A = np.stack(A)  # 将列表 A 中的邻接矩阵堆叠为一个 NumPy 数组，表示整个层次结构图的邻接矩阵
    # 此时将代表4个边集合关系的数组A堆叠为一个4X3X16X16的数组，普通的只有一个集合关系，为3X16X16
    #print("get_hierarchical_graph", A.shape)  # (4, 3, 16, 16)
    return A


def get_groups(dataset='NTU', CoM=8):
    groups = []
    if dataset == 'NTU':
        if CoM == 8:
            groups.append([8])
            groups.append([0, 9])
            groups.append([1, 2, 3, 10, 11])
            groups.append([4, 5, 12, 13])
            groups.append([6, 7, 14, 15])

        elif CoM == 0:
            groups.append([0])
            groups.append([1, 2, 3, 8])
            groups.append([4, 5, 9])
            groups.append([6, 7, 10, 11])
            groups.append([12, 13, 14, 15])

        elif CoM == 9:
            groups.append([9])
            groups.append([8, 10, 11])
            groups.append([0, 12, 13])
            groups.append([1, 2, 3, 14, 15])
            groups.append([4, 5, 6, 7])
        # if CoM == 2:
        #     groups.append([2])
        #     groups.append([1, 21])
        #     groups.append([13, 17, 3, 5, 9])
        #     groups.append([14, 18, 4, 6, 10])
        #     groups.append([15, 19, 7, 11])
        #     groups.append([16, 20, 8, 12])
        #     groups.append([22, 23, 24, 25])
        #
        # ## Center of mass : 21
        # elif CoM == 21:
        #     groups.append([21])
        #     groups.append([2, 3, 5, 9])
        #     groups.append([4, 6, 10, 1])
        #     groups.append([7, 11, 13, 17])
        #     groups.append([8, 12, 14, 18])
        #     groups.append([22, 23, 24, 25, 15, 19])
        #     groups.append([16, 20])
        #
        # ## Center of Mass : 1
        # elif CoM == 1:
        #     groups.append([1])
        #     groups.append([2, 13, 17])
        #     groups.append([14, 18, 21])
        #     groups.append([3, 5, 9, 15, 19])
        #     groups.append([4, 6, 10, 16, 20])
        #     groups.append([7, 11])
        #     groups.append([8, 12, 22, 23, 24, 25])
        else:
            raise ValueError()

    return groups


def get_edgeset(dataset='NTU', CoM=8):
    groups = get_groups(dataset=dataset, CoM=CoM)  # 函数返回一些分组数据
    for i, group in enumerate(
            groups):  # 遍历每个分组#group = [i - 1 for i in group] #对于每个分组中的每个节点，将其索引减去1。这可能是为了将索引从1-based调整为0-based。
        groups[i] = group  # 将调整后的分组重新赋值给groups列表中的相应位置。
    identity = []
    forward_hierarchy = []
    reverse_hierarchy = []
    for i in range(len(groups) - 1):
        self_link = groups[i] + groups[i + 1]  # 将当前组和下一组的节点合并，得到当前层与下一层之间的自环边
        self_link = [(i, i) for i in self_link]  # 将自环边的节点对构造为元组 (i, i) 的形式。
        identity.append(self_link)  # 将自环边添加到 identity 列表中。
        forward_g = []
        for j in groups[i]:  # 表示正向层次间的边。通过遍历当前层节点，构建节点对。
            for k in groups[i + 1]:
                forward_g.append((j, k))
        forward_hierarchy.append(forward_g)
        reverse_g = []
        for j in groups[-1 - i]:  # 表示反向的层次间的边。通过遍历下一层的节点，构建节点对。
            for k in groups[-2 - i]:
                reverse_g.append((j, k))
        reverse_hierarchy.append(reverse_g)
    edges = []
    for i in range(len(groups) - 1):
        edges.append([identity[i], forward_hierarchy[i], reverse_hierarchy[-1 - i]])
    return edges