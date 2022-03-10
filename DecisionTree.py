import numpy as np

class InternalNode:
    def __init__(self, decision, leftNode, rightNode):
        self.decision = decision
        self.leftNode = leftNode
        self.rightNode = rightNode

    def __str__(self) -> str:
        return f"Decision: {str(self.decision)}"

    def predict(self, X):
        if self.decision.left_or_right(X):
            return self.leftNode.predict(X)

        else:
            return self.rightNode.predict(X)

class Leaf:
    def __init__(self, value):
        self.prediction = value

    def __str__(self) -> str:
        return f"Leaf: {self.prediction:.3f}"

    def predict(self, X):
        return self.prediction


class Decision:
    """implement the decision process (boolean only)"""
    def __init__(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold
    
    def __str__(self) -> str:
        return f"{self.feature} < {self.threshold:.3f} ?"
    
    def split(self, X):
        X_feat = X[:,self.feature]
        below_threshold = X_feat < self.threshold
        X_left = X[below_threshold]
        X_right = X[~below_threshold]
        return X_left, X_right

    def split(self, X, y):
        X_feat = X[:,self.feature]
        below_threshold = X_feat < self.threshold
        X_left = X[below_threshold]
        y_left = y[below_threshold]
        X_right = X[~below_threshold]
        y_right = y[~below_threshold]
        return X_left, y_left, X_right, y_right

    def left_or_right(self, X):
        return X[self.feature] < self.threshold

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def __str__(self) -> str:
        if self.root == None:
            return "Uninitialized DecisionTreeClassifier"
        return self.str_helper_(self.root, 0, sep=" "*2)
    
    def str_helper_(self, root, depth, sep):
        if isinstance(root, Leaf):
            return f'{depth*sep}{str(root)}'
        ret_str = f'{depth*sep}{str(root)}'
        ret_str += f'\n{self.str_helper_(root.leftNode , depth+1, sep)}'
        ret_str += f'\n{self.str_helper_(root.rightNode, depth+1, sep)}'
        return ret_str
        

    def fit(self, X, y):
        self.num_features = X.shape[1]
        self.impurity = self.compute_impurity(y)
        self.root = self.fit_(X,y)


    def fit_(self, X, y,depth=0):
        if self.max_depth !=None:
            if depth==self.max_depth: # We are at max depth, create a leaf
                value = self.get_output_value(y)
                return Leaf(value) 
        if X.shape[0]<self.min_samples_split: # Too few sample to split, create a leaf
           value = self.get_output_value(y)
           return Leaf(value) 
        
        decision, impurity_improvement = self.find_best_split(X,y)
        if impurity_improvement == 0.0: # No good split possible within our parameters, create a leaf
            value = self.get_output_value(y)
            return Leaf(value)
        
        X_left, y_left, X_right, y_right = decision.split(X,y)


        leftNode = self.fit_(X_left,y_left, depth+1)
        rightNode = self.fit_(X_right,y_right, depth+1)

        return InternalNode(decision, leftNode, rightNode)

    def find_best_split(self, X, y):
        best_improvement = 0.0
        best_feature = -1
        best_threshold = None
        num_samples = X.shape[0]
        for feature in range(self.num_features):
            sorted_values = np.sort(np.unique(X[:,feature])) # get all the unique values, we split in between them
            for split_index in range(sorted_values.shape[0]-1):

                # pick the threshold to be halfway between the two values
                threshold_candidate = (sorted_values[split_index] + sorted_values[split_index+1])/2
                below_threshold = X[:,feature] < threshold_candidate
                if below_threshold.sum() < self.min_samples_leaf or num_samples - below_threshold.sum() < self.min_samples_leaf:
                    # we don't consider splits that create region with less than min_sample_leaf samples
                    continue
                
                improvement = self.compute_relative_improvement(y, y[below_threshold], y[~below_threshold])
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_feature = feature
                    best_threshold = threshold_candidate
        return Decision(best_feature, best_threshold), best_improvement
    
    def compute_relative_improvement(self, y, y_left, y_right):
        left_impurity = self.compute_impurity(y_left)
        right_impurity = self.compute_impurity(y_right)
        current_impurity = self.compute_impurity(y)

        n = y.shape[0]
        n_left = y_left.shape[0]
        n_right = y_right.shape[0]

        improvement = current_impurity - (n_left/n)*left_impurity - (n_right/n)*right_impurity
        return improvement

    def compute_impurity(self, y):
        raise NotImplementedError("This is an abstract decision tree, use DecisionTreeClassifier or DecisionTreeRegressor")


    def predict(self,X):
        y_pred = np.empty((X.shape[0]))
        for x in range(X.shape[0]):
            y_pred[x] = self.root.predict(X[x,:])
        return y_pred

    def get_output_value(self, y):
        """returns the value that is returned by a leaf with samples y"""
        raise NotImplementedError("This is an abstract decision tree, use DecisionTreeClassifier or DecisionTreeRegressor")
        
    

class DecisionTreeClassifier(DecisionTree):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='gini'):
        super().__init__(max_depth, min_samples_split, min_samples_leaf)
        self.criterion=criterion


    def compute_impurity(self, y):
        if self.criterion == 'gini':
            _, counts = np.unique(y, return_counts = True)
            return 1 - ((counts / y.shape[0])**2).sum()

        if self.criterion == 'entropy':
            # counts is never zero, np unique prunes classes with no samples
            _, counts = np.unique(y, return_counts = True)
            ps = (counts / y.shape[0])
            log_ps = np.log2(ps)
            return -(ps*log_ps).sum()

    def get_output_value(self, y):
        # for decision Tree, we give the (first) most frequent value
        unique, counts = np.unique(y, return_counts = True)
        max_count_value = unique[counts == counts.max()][0]
        return max_count_value
        
    

class DecisionTreeRegressor(DecisionTree):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        super().__init__(max_depth, min_samples_split, min_samples_leaf)
    
    def compute_impurity(self, y):
        #squared variance
        return y.var()

    def get_output_value(self, y):
        # for regression, we return the mean value
        return y.mean()

