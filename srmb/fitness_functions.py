from gplearn.fitness import make_fitness
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def custom_metric_f1(y, y_pred, w):
    y_pred = (y_pred > 0.5).astype('int')
    # print(y, y_pred)
    return f1_score(y_true=y, y_pred=y_pred)

def custom_metric_accuracy(y, y_pred, w):
    y_pred = (y_pred > 0.5).astype('int')
    # print(y, y_pred)
    return accuracy_score(y_true=y, y_pred=y_pred)

def custom_metric_precision_recall(y, y_pred, w):
    alpha=0.5
    y_pred = (y_pred > 0.5).astype('int')
    return alpha*precision_score(y_true=y, y_pred=y_pred, zero_division=0.0) + (1-alpha)*recall_score(y_true=y, y_pred=y_pred)

def custom_metric_auroc(y, y_pred, w):
    try:
        return roc_auc_score(y, y_pred)
    except ValueError:
        return 0.0
    
    
def custom_weighted_f1_auroc(y, y_pred, w):
    alpha=0.5
    y_pred_class = (y_pred > 0.5).astype('int')
    f1 = f1_score(y_true=y, y_pred=y_pred_class)
    roc = 0.0
    try:
        roc = roc_auc_score(y, y_pred)
    except ValueError as e:
        print(f'Error exception: {e}')
    
    return alpha*roc + (1-alpha)*f1
    

customacc = make_fitness(function=custom_metric_accuracy, greater_is_better=True)
customf1 = make_fitness(function=custom_metric_f1, greater_is_better=True)
customauroc = make_fitness(function=custom_metric_auroc, greater_is_better=True)
customrocf1 = make_fitness(function=custom_weighted_f1_auroc, greater_is_better=True)
customprecrec = make_fitness(function=custom_metric_precision_recall, greater_is_better=True)