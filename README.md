# VanillaML
VanillaML contains a collection of popular machine learning algorithms. 

All implementations are made to be as simple as possible. That said, there are no bells and whistles to 
speed up model's performance for production usage. The implementations follows scikit-learn's API conventions.

 <p align="center">[Demos](#demos)</p>

## Requirements
* Python 2.7
* Numpy and matplotlib which can be installed via:

```
$ sudo pip install -r requirements.txt
```

## Implemented algorithms
### 1. Supervised
- <b>Classification</b>
    - Nearest Neighbors
    - Perceptron
    - Linear Support Vector Machines
    - Logistic Regression
    - Maximum Entropy
    - Decision Trees
    - Random Forest
    - Feed-forward Neural Network
    - Adaboost
- <b>Regression</b>
    - Nearest Neighbors
    - Linear Regression
    - Decision Trees
    - Random Forest
    - Feed-forward Neural Network    
    - Gradient Boosting

### 2. Unsupervised
- <b>Clustering</b>
    - K-Means
- <b>Decomposition</b>
    - PCA

## Demos

<p align="center">
    <img width="500" high="500" src="http://i.imgur.com/uZKqKXi.gif">
    <br>
    KMeans clustering.
</p>

<p align="center">
    <img width="500" high="500" src="http://i.imgur.com/uSDPY0x.gif">
    <br>
    A gradient boosted regressor fitting the curve x * sin(x) using a decision tree with the depth 3 as the base regressor.
</p>

## License
BSD
