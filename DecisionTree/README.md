### Code within ./


The 'datasets.py' file contains a helper function, get_bank_data, to load the bank data and convert numerical features to binary features (using the media) given the file paths to the train/test csv files. 

The code for hw 1, i.e. the ID3 code and associated helper functions pre ensemble learning, exists in 'decision_tree_fns.py.' The tree_maker function in this file was updated to accomodate random forests such that it takes a new argument, attr_subset_len, which is the number that decides how many features we randomly subsample and consider in deciding the next split. 

### Code within ./linear_regression


There is a jupyter notebook called 'linear_regression.ipynb' that runs the experiments, and I converted it to a python script with an associated 'run.sh' file. The convergence plots for batch GD and stochastic GD are saved as 'batch_gradient_descent_loss_curve.png' and 'stochastic_gradient_descent_loss_curve.png' respectively. 



### Code within ./emsemble_learning

Within the ensemble learning directory, I saved the forests I made in directories that correspond to the problem they were made for, i.e. 2a,2b,2c,2d,2e. The forests were saved as python pkl files, which are large and so I won't upload them to github. I realize that I should've probably saved them as text files and read them as jsons or something, but alas. 

The code for the experiments of problem 2 in the homework exists in ensemble_learning.ipynb. Of course, you probably won't want to remake the trees because it takes a long while, so I made a separate script, ensemble_learning.py, 
to at least print some of the results. Any of the *.png files are the figures from the experiments and are named relative to the problem they're associated with. 

Code for the bonus dataset in problem 3 exists in 3_bonus_credit_card.ipynb.


The file 'ensemble_fns.py' contains the functions / helper functions to create adaboost, bagged forests and random forests. Here are explanations of the main function declarations: 

############################################################################################################
#### random_forest(X,y,feat_names,tree_count=10, max_depth=10, IG_metric='entropy', attr_subset_len=2)

This function returns a random forest as a python list of 'tree_count' trees all which have a 'max_depth'
and whose split decisions were made using 'IG_metric' (one of 'gini' 'entropy' or 'me' (majority error). As previously mentioned, 'attr_subset_len' is the number that decides how many features we randomly subsample and consider in deciding the each split. 

X should be a numpy array w/ dimensions as (num_examples, num_features)
y should be a numpy array w/ dimensions as (num_examples,1)
feat_names should be a list of strings of the feature names in an order that corresponds correctly with the columns of X. 

For the bank data, this is: feat_names = np.array(['age', 'job', 'marital', 'education', 'default', 'balance',
       'housing', 'loan', 'contact', 'day_of_week', 'month', 'duration',
       'campaign', 'pdays', 'previous', 'poutcome'], dtype='<U11')


##### forest_pred(ex, forest, feat_names)

Takes in an example, a forest, and the feat_names of the data, and returns a prediction that is the
mode of the predictions across each tree in the forest. 

##### forest_acc(X,y,forest,feat_names)

This function returns the accuracy of our forest ensemble across the input dataset.

Note: Both forest_pred & forest_acc can be used on bagging forests. 

############################################################################################################
#### bagging(X,y,feat_names,tree_count=10, max_depth=10, IG_metric='entropy')
This function returns a bagging forest as a python list of 'tree_count' trees all which have a 'max_depth'
and whose split decisions were made using 'IG_metric' (one of 'gini' 'entropy' or 'me' (majority error).

X should be a numpy array w/ dimensions as (num_examples, num_features)
y should be a numpy array w/ dimensions as (num_examples,1)
feat_names should be a list of strings of the feature names in an order that corresponds correctly with the columns of X. 


############################################################################################################

#### adaboost(X,y,feat_names, num_stumps=100, IG_metric='entropy')
This function returns an adaboost ensemble with 'num_stumps,' whose split decisions were made using 'IG_metric' (one of 'gini' 'entropy' or 'me' (majority error). 
It also returns an array 'amount_of_says' that is of dimensions (num_stumps,) which corresponds to the amount of say of each stump. 
The ith index of stumps corresponds with the ith index of 
amount_of_says. 

X should be a numpy array w/ dimensions as (num_examples, num_features)
y should be a numpy array w/ dimensions as (num_examples,1)
feat_names should be a list of strings of the feature names in an order that corresponds correctly with the columns of X. 

##### adaboost_pred(ex, stumps, feat_names, amount_of_says)
This function takes in an example, i.e. X[0], the feat_names, as well as all the stumps/amount_of_says array output from adaboost, and makes a prediction by

1) summing over the amount_of_says for all the stumps that make the same prediction for the example
2) choosing the prediction value which corresponds to the highest of these sums

##### adaboost_acc(X,y,stumps,feat_names, amount_of_says)

This function returns the accuracy of our adaboost stump ensemble across the input dataset.


############################################################################################################

The run.sh file prints the results for these experiments, but doesn't run the tree learning as it is quite a timely process


