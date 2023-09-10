from sklearn.metrics import f1_score


class Evaluator:
    def eval(selfself, y_true, y_pred):
        # Convert tensors to numpy arrays
        y_true_numpy = y_true.cpu().numpy()
        y_pred_numpy = y_pred.cpu().numpy()

        # we get the f1 score for each class individually
        f1_macro = f1_score(y_true, y_pred > 0.5, average=None, zero_division=1)

        return {
            'f1_score_macro': f1_macro
        }
