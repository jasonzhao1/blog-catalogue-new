# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from generate_initial_features import generate_initial_features
from model_training import get_model
from processing import get_spectral_embedding, simple_spectral_embedding
from reader import read_labels, read_graph



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    labels = read_labels('labels.txt')
    graph = read_graph('graph.txt')
    embeddings = simple_spectral_embedding(graph)
    get_model(embeddings, labels)
    # initial_features = generate_initial_features(graph,labels)
    print('done')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
