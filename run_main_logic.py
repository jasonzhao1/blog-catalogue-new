def run_main_logic():
    data = ???
    adj, D_isqrt = process_adf(data)
    normalized_adjs = gen_normalized_adjs(adj, D_isqrt)
    DAD, DA, AD = normalized_adjs
    evaluator = ???
    split_idx = ???
    def eval_test():
        return

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

    evaluate_params(data, eval_test, model_outs, split_idx, mlp_dict, fn=mlp_fn)