from sklearn.model_selection import train_test_split

from correct_smoothing import process_adj, gen_normalized_adjs, double_correlation_autoscale, evaluate_params
from data_reader import read_graph, convert_nx_to_pyg
from evaluator import Evaluator
import glob


# The logic to get model output
# and then do post processing with label propagation
def run_main_logic():
    networkx_data = read_graph('graph.txt')
    data = convert_nx_to_pyg(networkx_data)
    # inverse sqrt of degree of each node;
    # used in GCN for normalization
    # can help in the training of GCN, ensuring scale of features doesn't explode or vanish as they pass through layers
    adj, deg_inv_sqrt = process_adj(data)
    normalized_adjs = gen_normalized_adjs(adj, deg_inv_sqrt)
    # DAD: symmetrically normalized adjacency matrix
    # DA: left-normalized adjacency matrix
    # AD: right-normalized adjacency matrix
    DAD, DA, AD = normalized_adjs

    # get the evaluator that calculates the f1 score
    evaluator = Evaluator()

    # Split data into train, validation, and test sets
    train_idx, temp_idx = train_test_split(range(len(X)), test_size=0.4, random_state=42)
    valid_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    def eval_test(result, idx):
        return evaluator.eval(y_true=data.y[idx], y_pred=result[idx].argmax(dim=-1, keepdim=True))

    # TODO: figure out why the hyperparameter is like this?
    mlp_dict = {
        'train_only': True,
        'alpha1': 0.9791632871592579,
        'alpha2': 0.7564990804200602,
        'A1': DA,
        'A2': AD,
        'num_propagations1': 50,
        'num_propagations2': 50,
        'display': False,
    }

    mlp_fn = double_correlation_autoscale

    # TODO: figure out actual file of import
    model_outs = glob.glob(f'test.pt')

    evaluate_params(data, eval_test, model_outs, split_idx, mlp_dict, fn=mlp_fn)
