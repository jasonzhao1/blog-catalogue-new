from collections import defaultdict


# using simple label propagation to generate some initial feature for the ML model
# TODO: compare this with spectral embedding as initial features
def generate_initial_features(graph, labels, num_iterations=100):
    features = {}

    for _ in range(num_iterations):
        new_labels = defaultdict(list)

        # propagate a few iterations for label propagation
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            # calculate average label for each class
            avg_labels = []
            for i in range(len(labels[node])):
                avg = sum([labels[neighbor][i] for neighbor in neighbors]) / len(neighbors)
                avg_labels.append(avg)
            new_labels[node] = avg_labels

        labels = new_labels

    features = labels

    return features
