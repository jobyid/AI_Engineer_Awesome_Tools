##Handy Repos

[Machine Learning Tooling](https://github.com/ml-tooling) - Great selection of tools to make your Machine Learning life easier 


## Handy knowledge

### CURSE OF DIMENSIONALITY - THE LESS THE BETTER (PCA and LDA say hello!)

### Major data categories:

-- Categorical data
__NOMINAL__
categorical variables (pet: cat / dog / hamster) = characteristics

__ORDINAL__
variables can be ordered (level of education)

-- Numerical data
__DISCRETE__
variables can take only certain values

__CONTINOUS__
fully arithmetical variables, can be anything

### Encoding categorical data
IF your data is ORDINAL, it is sufficient to use simple integer encoding (0, 1, 2, 3 etc..) BUT! 
More often than not, such solution would slow down the ML algorhitm (reinterpretation, in-between predictors), that's why we use 
__ONE HOT ENCODING (having a boolean column for each category)__

### Distances
#### Hamming (bitwise)
between boolean values, usual for one hot encoded tables
> sum(e1 != e2 for e1, e2 in zip(a, b)
> scipy.spatial.distance.hamming(a, b)

#### Euclidean
between real valued vectors, usual for tables
> np.linalg.norm(x - y)
> np.sqrt(np.sum(np.square(x-y))) 

#### Manhattan
between real valued vectors, preferable for uniform grids and integer feature spaces
like those rectangular streets in Manhattan you got 4 directions of movement
> sum(abs(e1-e2) for e1, e2 in zip(a, b))
> scipy.spatial.distance.cityblock(a, b)

#### Minkowski Distance
__great exploratory tool__
generalization of the Euclidean and Manhattan. 
> sum(abs(e1-e2)**p for e1, e2 in zip(a,b))**(1/p)
> scipy.spatial.minkowski_distance(a, b, p) Manhattan p = 1, Euclidean p = 2 and everything between

## MODELS:


### UNSUPERVISED


#### DECISION TREES (CART)
POWERFUL SUPERVISED CLASSIFIERS

- Essentialy - chains of boolean categories. 
- The more factors, the more SPLITS. The more SPLITS, the more DEEP is the tree.
- Can do either classification or regression 
- At the end of the tree, in the sub-trees, are the LEAFS (leaf nodes) that make the prediction based on previous splits. 
- Quite good at mapping non-linear relationships

__root node__ : entire sample that gets further divided
__pruning__ : removing sub-nodes of a decision node
__purity__ : subset composed of only a single class is considered pure
__entropy__ : quanitifes the randomness (disorder) within a set of class values. Used to calculate the homogeneity (impurity) of a sample. Completely homogenous - 0, equally divided - 1.If group's entropy is high, it is very diverse and cannot give us much info about other items that belong to the same group.
__gini impurity__ : used at each node to decide which feature is best to split on. all cases in the node fall into a single target category - 0. The closer to 0 the better.


##### RandomForestClassifier 
- N tree estimators built on randomly sampled training data
- Random subsets of features when splitting nodes
- Final prediction is the average of trees __not correlated__ with each other.

__STRENGHTS__							__WEAKNESSES__
| Performs well on most problems		| Not easily interpretable
| Handles noisy or missing data			| Needs thorough tuning
| Reduces risk of overfitting
| Categorical&continous features
| Selects only most important features
| Resistant to large datasets

##### Gradient Boosting
- Like RFC, but not so random. Kind of like tree-breeding.
- Incorporates back-propagation through a cost function to evaluate parameters for the next tree being created, based on the weakest data points in the previous tree.- 

#### Support vector machine (SVM)
- Abstract concept of a machine, that works as a LINEAR CLASSIFIER.
- Combination of KNN and Linear Regression
- Uses a boundary called hyperplane to partition data into groups
- Considers data as either linearly separable or not
- If data is not linearly separable it maps the problem into a __higher dimension called ALTITUDE__ through a procces called __KERNEL TRICK__. That serves the case when data is accumulated in the centre of XY.

__LINEAR KERNEL__ : simply a dot product xi * xj, good for linearly separable data
__POLYNOMIAL KERNEL__: (dot product + 1)^alpha good for non lin sep data
__SIGMOID KERNEL__: tanh(kappa * dotproduct - delta) 
__GAUSSIAN RBF KERNEL__: usually a first try kernel good for starters

__C parameter__ - inverse of tolerance of separation margin. The bigger the value, the lesser the margin.

__STRENGHTS:__ 							__WEAKNESSES:__
| Universal application					| Requires tuning
| Resistant to noisy data 				| Slow training
and overfitting
| Easier to use than neural net			| Complex black box
| High accuracy in data mining

  
### SUPERVISED


#### KNeighborsClassifier 
- Predicts value/label of input based on fitted data
- Input must be subject to same dimensions as fitted data
- Algorithm picks K neighbours of the input based on chosen proximity measure (closest points)
- Mean/mode of the K neighbours is the predicted value/label of the input

#### Naive Bayes Classifier
- Used in spam filtering
- Naive because it doesn't consider the order of words. However, still effective.
- Zipf's law for ranking on the words in a frequency table?
a = not spam mails
b = spam mails

probabilities based on that

given prob that message is spam, what is a probability of occurance of word 1 and word 2 and word 3 etc...

P(S) * P(w1) * P(w2) ....

