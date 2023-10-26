import numpy as np
from numpy import log2


def label_H(y):
    H = 0
    vals, counts = np.unique(y, return_counts=True)
    for count in counts:
        H+=(count / len(y))*log2(count / len(y))
    return -H
def feature_H(x_i, y):
    H = 0
    for val,count in np.column_stack(np.unique(x_i, return_counts=True)):
        H+= (int(count) / len(y)) * label_H(y[np.where(x_i == val)[0]])
    return H


def label_ME(y):
    H = 0
    vals, counts = np.unique(y, return_counts=True)
    ME = (len(y) - max(counts)) / len(y)
    return ME
    
def feature_ME(x_i, y):
    ME = 0
    for val,count in np.column_stack(np.unique(x_i, return_counts=True)):
        ME+= (int(count) / len(y)) * label_ME(y[np.where(x_i == val)[0]])
    return ME
    
def label_GINI(y):
    GINI = 1
    vals, counts = np.unique(y, return_counts=True)
    for count in counts:
        GINI -= (count / len(y)) ** 2
    return GINI
    
def feature_GINI(x_i, y):
    GINI = 0
    for val,count in np.column_stack(np.unique(x_i, return_counts=True)):
        GINI += (int(count) / len(y)) * label_GINI(y[np.where(x_i == val)[0]])
    return GINI


# returns feature index w/ highest IG based on the IG metric employed
def calc_IG(X,y,metric='gini'):
    IGs = np.zeros((X.shape[1]))
    for i, col in enumerate(X.T):
        if metric == 'gini':
            IGs[i] = label_GINI(y) - feature_GINI(col,y)
        elif metric == 'entropy':
            IGs[i] = label_H(y) - feature_H(col, y)
        elif metric == 'me':
            IGs[i] = label_ME(y) - feature_ME(col, y)
        else:
            raise exception('Not a valid type of IG')

    IGs = np.round(IGs, decimals=10)
    return np.argmax(IGs)

def mode(arr):
    vals, counts = np.unique(arr, return_counts=True)
    max_count_i = np.argmax(counts)
    return vals[max_count_i]


def subtree(X, y, original_X, original_y, feature, feature_name): # for a particular feat.
    classes = np.unique(y)
    tree = {}
    tree[feature_name] = {}
    x_i = X[:,feature]
    for attr_val in np.unique(original_X[:,feature]):
        attr_val_loc = np.where(x_i == attr_val)[0]
        attr_val_labels = y[attr_val_loc]
        attr_count = len(attr_val_loc)
        pure = False
        for cl in classes:
            class_members = np.where(attr_val_labels==cl)[0]
            if int(attr_count) == len(class_members):
                tree[feature_name][attr_val] = cl
                X,x_i,y = [np.delete(arr, attr_val_loc, axis=0) for arr in [X,x_i,y]]
                pure = True

        if not pure:
            tree[feature_name][attr_val] = "?"
        if attr_count == 0:
            tree[feature_name][attr_val] = mode(original_y)
    return tree, X, y



# X is dataset where rows are examples and columns are features, y is the vector of the labels
def tree_maker(X, 
               y, 
               original_X, 
               original_y, 
               feat_names, 
               max_depth=10, 
               IG_metric='entropy', 
               allowed_attrs=[], 
               curr_depth=0,
               attr_subset_len=2):
    if max_depth > X.shape[1]: max_depth = X.shape[1]
        
    if X.shape[0] != 0:
        if curr_depth == 0: # fresh
            allowed_attrs=np.arange(X.shape[1]) # haven't used up any of the features yet
            
        # when choosing next feature, choose from a random subset of features available,
        # unless the subset len is greater than the number of features left, then just pick from 
        # features that are left 
        if len(allowed_attrs) <= attr_subset_len:
            subi = allowed_attrs
        else:
            subi = np.random.choice(len(allowed_attrs), attr_subset_len,replace=False) 
        best_feat = subi[calc_IG(X[:,subi],
                                 y,
                                 IG_metric)] # best feature, given random subset of unused features
        
        tree, X, y = subtree(X,y,original_X, original_y, best_feat, feat_names[best_feat])
        curr_depth+=1
        for val, label in list(tree[feat_names[best_feat]].items()): # different feature values are the key, labels is vals
            if label == '?':
                if curr_depth >= max_depth:
                    tree[feat_names[best_feat]][val] = mode(y[X[:,best_feat] == val])
                else:
                    tree[feat_names[best_feat]][val] = tree_maker(X[X[:,best_feat] == val],
                                                      y[X[:,best_feat] == val],
                                                      original_X,
                                                      original_y,
                                                      feat_names,
                                                      max_depth=max_depth,
                                                      IG_metric=IG_metric,
                                                      allowed_attrs=np.delete(allowed_attrs,
                                                                              np.where(allowed_attrs == best_feat)[0]
                                                                             ),
                                                      curr_depth=curr_depth,
                                                      attr_subset_len=attr_subset_len) # placing the subtree

    return tree
    

def ID3(X,y,feat_names, max_depth=10, IG_metric='entropy'):
    return tree_maker(X, y, X, y, 
                      feat_names, 
                      max_depth=max_depth, 
                      IG_metric=IG_metric,
                      attr_subset_len=X.shape[1])


def predict(ex,tree,feat_names):
    feature = list(tree.keys())[0]
    feature_i = np.where(feat_names == feature)[0][0]
    val = tree[feature][ex[feature_i]]
    while isinstance(val, dict):
        feature = list(val.keys())[0]
        feature_i = np.where(feat_names == feature)[0][0]
        val = val[feature][ex[feature_i]]
    return val


