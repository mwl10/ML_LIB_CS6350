### Code within ./


The 'datasets.py' file contains a helper function, get_bank_data, to load the bank data and convert numerical features to binary features (using the media) given the file paths to the train/test csv files. 

The code for hw 1, i.e. the ID3 code and associated helper functions pre ensemble learning, exists in 'decision_tree_fns.py.' The tree_maker function in this file was updated to accomodate random forests such that it takes a new argument, attr_subset_len, which is the number that decides how many features we randomly subsample and consider in deciding the next split. 

### Code within ./linear_regression


There is a jupyter notebook called 'linear_regression.ipynb' that runs the experiments, and I converted it to a python script with an associated 'run.sh' file. The convergence plots for batch GD and stochastic GD are saved as 'batch_gradient_descent_loss_curve.png' and 'stochastic_gradient_descent_loss_curve.png' respectively. 



### Code within ./emsemble_learning

Within the ensemble learning directory, I saved the forests I made in directories that correspond to the problem they were made for, i.e. 2a,2b,2c,2d,2e. The forests were saved as python pkl files, which are large and so I won't upload them to github. I realize that I should've probably saved them as text files and read them as jsons or something, but alas. 

The file 'ensemble_fns.py' contains the functions / helper functions to create adaboost, bagged forests and random forests. Here are explanations of the main function declarations: 

############################################################################################################
#### random_forest(X,y,feat_names,tree_count=10, max_depth=10, IG_metric='entropy', attr_subset_len=2)

This function returns a random forest as a python list of 'tree_count' trees all which have a 'max_depth'
and whose split decisions were made using 'IG_metric' (one of 'gini' 'entropy' or 'me' (majority error). As previously mentioned, 'attr_subset_len' is the number that decides how many features we randomly subsample and consider in deciding the each split. 

X should be a numpy array w/ dimensions as (num_examples, num_features)
y should be a numpy array w/ dimensions as (num_examples,)
feat_names should be a list of strings of the feature names in an order that corresponds correctly with the columns of X. 

For the bank data, this is: feat_names = np.array(['age', 'job', 'marital', 'education', 'default', 'balance',
       'housing', 'loan', 'contact', 'day_of_week', 'month', 'duration',
       'campaign', 'pdays', 'previous', 'poutcome'], dtype='<U11')


############################################################################################################
#### bagging(X,y,feat_names,tree_count=10, max_depth=10, IG_metric='entropy')
This function returns a bagging forest as a python list of 'tree_count' trees all which have a 'max_depth'
and whose split decisions were made using 'IG_metric' (one of 'gini' 'entropy' or 'me' (majority error).

X should be a numpy array w/ dimensions as (num_examples, num_features)
y should be a numpy array w/ dimensions as (num_examples,)
feat_names should be a list of strings of the feature names in an order that corresponds correctly with the columns of X. 


#### adaboost(X,y,feat_names, num_stumps=100, IG_metric='entropy')
This function returns an adaboost ensemble with 'num_stumps,' whose split decisions were made using 'IG_metric' (one of 'gini' 'entropy' or 'me' (majority error). 

X should be a numpy array w/ dimensions as (num_examples, num_features)
y should be a numpy array w/ dimensions as (num_examples,)
feat_names should be a list of strings of the feature names in an order that corresponds correctly with the columns of X. 


############################################################################################################

The run.sh file prints the results for these experiments, but

