import os
import pickle
import pygraphviz
from networkx.drawing import nx_agraph
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# seeds for 20 trials
RANDOM_SEEDS_FOR_UNDERSAMPLING = [42, 2024, 1234, 2405, 11, 9345,  858, 8590, 4754, 1959,
                                  707, 10524, 83946, 63297, 78035, 22664, 49283, 35253, 82273, 90378]

def calculate_metrics(model, x_train, y_train, x_test, y_test):
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print(str(model.__class__).split('.')[-1].split("'")[0])
    # print(f'Training accuracy: {accuracy_score(y_train, y_train_pred):.4f}')
    acc = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    print(f'Test accuracy: {acc:.4f} Test AUROC: {roc:.4f} Test F1 score: {f1:.4f}')
    #classification_report_str = classification_report(y_test, y_test_pred)
    #print("Classification Report:\n", classification_report_str)
    # print('---'*10)
    return (acc, f1, roc)

def save_sr_models(list_of_sr_models, key='SR', save_dir='../res_sr/'):
    SEEDS_USED = RANDOM_SEEDS_FOR_UNDERSAMPLING
    dir_path = save_dir

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    
    for i, sr_model in enumerate(list_of_sr_models):
        sr_model_object_name = os.path.join(dir_path, f'{key}_{SEEDS_USED[i]}')
        # save model by pickling
        with open(sr_model_object_name, 'wb') as output:
            pickle.dump(sr_model, output, pickle.HIGHEST_PROTOCOL)


def load_sr_models(key='SR', save_dir='../res_sr/'):
    SEEDS_USED = RANDOM_SEEDS_FOR_UNDERSAMPLING
    dir_path = save_dir
    
    models = []
    for seed in SEEDS_USED:
        sr_model_object_name = os.path.join(dir_path, f'{key}_{seed}')
        # load the model
        with open(sr_model_object_name, 'rb') as inp:
            sr_model = pickle.load(inp)
        models.append(sr_model)
    return models


def create_graph(sr_estimator):
    """
    Args:
        sr_estimator (_type_): symbolic classifier object from gplearn
    Returns:
        _type_: networkx graph object
    """
    dot_data = sr_estimator._program.export_graphviz()
    G = nx_agraph.from_agraph(pygraphviz.AGraph(dot_data))
    
    for key in G.nodes:
        if G.nodes[key]['label'].startswith('s__'):
            label = G.nodes[key]['label'][3:].split('_')
            label = '\n'.join(label)
            G.nodes[key]['label'] = label
            G.nodes[key]['fontname'] = 'times italic'
            G.nodes[key]['fontsize'] = 15
            G.nodes[key]['fillcolor'] = 'darkolivegreen3'
        else:
            label = G.nodes[key]['label']
            G.nodes[key]['fillcolor'] = 'white'
            G.nodes[key]['fontname'] = 'courier new'
            
    return G

def graph_to_jpg(T, path='abcd.jpg', formatstr='jpg'):
    """
    save the networkx digraph to an image.
    Args:
        T: networkx digraph object of the jst as returned from `simpler_common_trunk_tree_to_digraph` function
        path: path to save the jpg file.

    Returns:
        path of the saved image
    """
    from networkx.drawing.nx_agraph import to_agraph
    A = to_agraph(T)
    A.layout('dot')
    A.draw(path, format=formatstr)
    return path