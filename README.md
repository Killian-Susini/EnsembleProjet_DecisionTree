# EnsembleProjet_DecisionTree
Implementation of a decision tree (clasification, regression)

The implementation is contained in DecisionTree.py

Also, two jupyter notebooks. 

    - Demonstration.ipynb is a demo of the implementation

    - EL_Cyberbullying.ipynb contains the models that were trained for the cyberbullying classification task

The implementation tries to copy how scikit-learn decision trees are called and used. So, a tree is firstbuilt by calling the constructor with a few parameters that can be set. Then the .fit(X,y) method is called, with input features X and prediction y. X is a numpy array with shape (n samples,d features) and y has shape (n,). Finally, the method .predict(X) is called on X, which output a vector y, same shape as for fit. The general functions and methods to grow a decision tree are implemented in the abstract class DecisionTree, and then DecisionTreeClassifier and DecisionTreeRegressor implements the two method that control how a node is evaluated and how to compute the output of a leaf (get_impurity and get_output_value).

On both tree it is possible to control the max_depth, min_samples_split and min_samples_leaf (same meaning as the scikit-learn implementation). The Classifier also can be set too use either gini or entropy as a impurity estimation.

The main weakness of the implementation is its use of recursion (theoratical limit to growth due to python call stack limit). Also some parameters that can be set in scikit can't be in this implementation, for example splitter and max_features. Those in particular would need to be implemented to be able to use those DT in a Random Forest for example. However it should be easy to implement them in this implementation.
