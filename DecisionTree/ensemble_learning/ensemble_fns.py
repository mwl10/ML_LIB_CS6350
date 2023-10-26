import numpy as np
import sys

# for parent directory import
sys.path.append('..')
from decision_tree_fns import predict, tree_maker, ID3

#############################################################################################
# adaboost functions 
#############################################################################################
def weight_update(weights, is_right ): # update for each tree 
    error = (1-np.mean(is_right)) 
    epsilon = 0.00001
    if error == 1: error-= epsilon
    if error == 0: error+= epsilon
    a = 0.5 * np.log((1-error)/error) 
    # e^-a (decay) if is right, e^a (scale) if is wrong 
    weights[is_right] *= np.e**-a
    weights[~is_right] *= np.e ** a
    return weights / weights.sum(), a # normalize


# return resampled indices of X given weights for adaboost
def resample(X, weights):
    shuffle = np.random.choice(np.arange(len(weights)), p=weights,size=X.shape[0],replace=True)
    return shuffle


def adaboost(X,y,feat_names, num_stumps=100, IG_metric='entropy'):
    X_,y_ = X,y
    # set initial weights
    weights = np.ones((X.shape[0])) / X.shape[0]
    amount_of_says = np.zeros((num_stumps))
    stumps = []
    for i in range(num_stumps):

        # make a stump & predict with it
        stump = tree_maker(X_, y_, X, y,
                           feat_names,
                           max_depth=1,
                           IG_metric=IG_metric,
                           attr_subset_len=X.shape[1])

        preds = np.array([predict(X_[ex],stump,feat_names) for ex in range(len(X_))])

        # boolean array to reflect the correct predcitions from the stump
        is_right = (y_[0] == preds)
        weights,amount_of_say = weight_update(weights, is_right)
        
        # keep the amount of say for the stump
        amount_of_says[i] = amount_of_say
        stumps.append(stump)
        shuffle = resample(X_,weights)
        X_,y_ = X_[shuffle], y_[shuffle]
        # reset weights after the resample 
        weights = np.ones((X.shape[0])) / X.shape[0]
        
    return stumps,amount_of_says

def adaboost_pred(ex, stumps, feat_names, amount_of_says):
    preds_across_stumps = [predict(ex,stump,feat_names) for stump in stumps]
    label_vals = np.unique(preds_across_stumps)
    
    # sum amount of say across stumps that make the same prediction
    amount_of_says_sums = [amount_of_says[np.where(preds_across_stumps == val)[0]].sum() for val in \
                         label_vals]
    # whichever index has the highest sum is the index we use in label_vals for the prediction
    return label_vals[np.argmax(amount_of_says_sums)]

def adaboost_acc(X,y,stumps,feat_names, amount_of_says):
    return (y == [adaboost_pred(X[ex], stumps, feat_names, amount_of_says) \
                  for ex in range(len(X))]).mean()


####################################################################################################
# bagging and random forest functions
#####################################################################################################

def random_forest(X,y,feat_names,tree_count=10, max_depth=10, IG_metric='entropy', attr_subset_len=2): 
    m=X.shape[0]
    bootstraps = np.random.choice(m, size=(tree_count,m),replace=True)
    forest = [tree_maker(X[bootstraps[i]], y[bootstraps[i]], 
                        X, y,
                        feat_names,
                        max_depth=max_depth,
                        IG_metric=IG_metric,
                        attr_subset_len=attr_subset_len) for i in range(tree_count)]
    return forest


# random forest, without randomly subsetting the features to decide on the split 
def bagging(X,y,feat_names,tree_count=10, max_depth=10, IG_metric='entropy'):
    m=X.shape[0]
    bootstraps = np.random.choice(m, size=(tree_count,m),replace=True)
    forest = [tree_maker(X[bootstraps[i]], y[bootstraps[i]], 
                        X, y,
                        feat_names,
                        max_depth=max_depth,
                        IG_metric=IG_metric,
                        attr_subset_len=X.shape[1]) for i in range(tree_count)]

    return forest
                  
def forest_pred(ex, forest, feat_names):
    preds = np.array([predict(ex,tree,feat_names) for tree in forest])
    unique, counts = np.unique(preds, return_counts=True)
    return unique[np.argmax(counts)]

def forest_acc(X,y,forest,feat_names):
    return (y == [forest_pred(X[ex], forest,feat_names) for ex in range(len(X))]).mean()

