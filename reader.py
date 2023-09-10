from collections import defaultdict
import networkx as nx


# def read_labels(filename):
#     with open(filename, 'r') as f:
#         # skip summary line
#         f.readline()
#
#         labels = {}
#         for line in f:
#             node, label = map(int, line.strip().split())
#             if node not in labels:
#                 labels[node] = [0] * 39
#             labels[node][label] = 1
#
#     return labels


def read_labels(filename):
    with open(filename, 'r') as f:
        # skip summary line
        f.readline()

        labels = [None] * 10312
        for line in f:
            node, label = map(int, line.strip().split())
            # if node not in labels:
            #     labels[node] = [0] * 39
            if not labels[node]:
                labels[node] = [0] * 39
            labels[node][label] = 1

    return labels


def read_graph(filename):
    with open(filename, 'r') as f:
        # skip summary line
        f.readline()

        G = nx.Graph()
        for line in f:
            node1, node2 = map(int, line.strip().split())
            G.add_edge(node1, node2)

    return G
