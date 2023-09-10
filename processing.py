
import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, dropout_adj
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh



def get_spectral_embedding(netx_graph):
    data = from_networkx(netx_graph)
    embeddings = spectral(data, 'blogcatalog')
    return embeddings


def spectral(data, post_fix):
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Main

    Main.include("./norm_spec.jl")
    from julia.Main import main
    print('Setting up spectral embedding')
    data.edge_index = to_undirected(data.edge_index)
    np_edge_index = np.array(data.edge_index.T)

    N = data.num_nodes
    row, col = data.edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.to_scipy(layout='csr')
    result = torch.tensor(main(adj, 128)).float()
    torch.save(result, f'embeddings/spectral{post_fix}.pt')

    return result


def simple_spectral_embedding(G, dim=2):
    # Compute the normalized Laplacian matrix
    L_norm = nx.normalized_laplacian_matrix(G).asfptype()

    # Compute the eigenvectors of the normalized Laplacian corresponding to the smallest eigenvalues
    # 'which="SM"' means smallest eigenvalues. We compute dim+1 smallest eigenvalues because the smallest one is 0.
    eigenvalues, eigenvectors = eigsh(L_norm, k=dim+1, which="SM")

    # Use the eigenvectors corresponding to the smallest non-zero eigenvalues as the embedding
    return eigenvectors[:, 1:]


