##===============Assignment 2=============
## ======================================

from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn import svm
from sklearn import metrics
from sklearn import pipeline
from sklearn import tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.tree import DecisionTreeRegressor, export_text
import graphviz
from sklearn.tree import export_graphviz
from IPython.display import Image


#===================TASK 1: LINEAR SUPPORT VECTOR MACHINES===================
#=============================================================================


# a) Import the dataset from OpenML by using the respective scikit-learn methods.
banknote = fetch_openml('banknote-authentication')
X=banknote.data #features
y=banknote.target #target


df=pd.DataFrame()
df['y'] = y
df['y'].unique()


# goal of t-SNE is to project multi-dimensional points to 2- or 3-dimensional plots so that if two points were close in the initial high-dimensional space, they stay close in the resulting projection. If the points were far from each other, they should stay far in the target low-dimensional space too.



# b) Visualize the features by converting them to a two-dimensional format with the help of t-SNE. Does this dataset appear to be suitable for linear SVM classification?
tsne = TSNE(n_components=2, random_state=7).fit_transform(X)
#plt.figure(figsize=(6,5))

# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]

colors=["red" if val == '1' else "green" for val in y ]
plt.scatter(tx, ty, color=colors)
plt.show()
#fitting for linear SVM since data can be obviously divided into two groups



# c) Split the data into a training set and a test set with a test ratio of 0.2.
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=7)


# d) Setup a scikit-learn pipeline that does the following:
# 1. Scale the features by using one of the feature scalers from scikit-learn. Which one did you choose and why?
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 2. Train a linear SVM classifier with the hyperparameter C=1, max_iter=10000, and hinge loss.
svc = svm.LinearSVC(C=1, max_iter=10000, loss='hinge', random_state=7)
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
y_pred1 = svc.predict(X_train)



#e What is the performance of this classifier on the test set regarding accuracy and the confusion matrix? Also compare it to the performance on the training set. Is overfitting or underfitting a problem?
print('Confusion matrix: \n', metrics.confusion_matrix(y_test, y_pred))
print('Accuracy test data:', metrics.accuracy_score(y_test,y_pred))
print('Accuracy training data:', metrics.accuracy_score(y_train,y_pred1))

#pretty good accuracy, no overfitting, or undefitting

print('\n \n')



#============TASK 2: NON-LINEAR SUPPORT VECTOR MACHINES============================
#==================================================================================


# a) Fetch the dataset. Split it up into a training set and a test set with a test ratio of 0.2.
cancer = datasets.load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=7)

# b) Build three scikit-learn pipelines (see below) that do the following:
# 1. Scale the features by using one of the feature scalers from scikit-learn. Which one did you choose and why?
pipe1 = pipeline.make_pipeline(MinMaxScaler(), svm.LinearSVC(C=1, max_iter=1000, loss='hinge', random_state=7))
#pipe2 = pipeline.make_pipeline(StandardScaler(), svm.LinearSVC(C=1, max_iter=1000, loss='hinge', degree=3, random_state=7))
pipe3 = pipeline.make_pipeline(StandardScaler(), svm.SVC(C=1, coef0=1, kernel='poly', degree=3, random_state=7))

# 2. Train an SVM classifier on the training data.

#c) Compare the approaches in terms of accuracy on the test set. Explain the results
pipe1.fit(X_train,y_train)
print('Pipeline 1: ', pipe1.score(X_test,y_test))
#pipe2.fit(X_train,y_train)
#print('Pipeline 2: ', pipe2.score(X_test,y_test))
pipe3.fit(X_train,y_train)
print('Pipeline 3: ', pipe3.score(X_test,y_test), '\n')


#d) Try to improve the results by performing a grid search with the best performing classifier. Keep the configuration that was used before and search for better parameter settings of C (range between 0.1 and 100) and max_iter (range between 100 and 10000). Use appropriate steps in the grid.
parameters = [ {'C': [0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
               'max_iter': [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]}]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

clf = model_selection.GridSearchCV(svm.SVC(coef0=1, kernel='poly', degree=3, random_state=7), param_grid=parameters)
clf.fit(X_train, y_train)
print("Best parameters SVC: ", clf.best_params_)

## ==================OPTIONAL TASK 3: INSTANCE-BASED LEARNING===============
#---------------------------------------------------------------------------

#a) Create an exemplary classification dataset:

from sklearn.datasets import make_classification

# Create a synthetic classification dataset
X, y = make_classification(n_samples=10, n_features=5, n_classes=2, random_state=42)

# Display the dataset
print("Dataset:")
print("Features:")
print(X)
print("Labels:")
print(y)

# b) Compute distance values between all attributes:
import numpy as np

# Function to compute distance between attribute values
def compute_distance(v1, v2):
    return np.abs(v1 - v2) / (np.max([v1, v2]) - np.min([v1, v2]))

# Compute distance matrix
distance_matrix = np.zeros((10, 10, 5))

for i in range(10):
    for j in range(10):
        for k in range(5):
            distance_matrix[i, j, k] = compute_distance(X[i, k], X[j, k])

# Display the distance matrix
print("Distance Matrix:")
print(distance_matrix)



#c) Apply similarity measures on different attributes:

# Exponential similarity measure for attributes 1, 2, 3
similarity_exp = np.exp(distance_matrix[:, :, :3] * (-0.1))

# Binary measure for attributes 4, 5
similarity_binary = np.where(distance_matrix[:, :, 3:] < 0.3, 1, 0)

# Display the similarity matrices
print("Exponential Similarity Matrix:")
print(similarity_exp)

print("Binary Similarity Matrix:")
print(similarity_binary)

#d) Finish KNN computations:


# Combine similarity matrices for all attributes
combined_similarity = np.concatenate([similarity_exp, similarity_binary], axis=2)

# Average values for attribute similarities
average_similarity = np.mean(combined_similarity, axis=2)

# Determine k-nearest neighbors for example number 3
example_3_similarity = average_similarity[2, :]  # Example number 3
k_nearest_neighbors_indices = np.argsort(example_3_similarity)[1:4]  # Exclude itself

# Display the results
print("Average Similarity Matrix:")
print(average_similarity)

print(f"K-nearest neighbors for example 3 (k=3): {k_nearest_neighbors_indices}")


##======================TASK 4: CLASSIFICATION WITH DECISION TREES==============
#===============================================================================


# a) Fetch the dataset from OpenML and split-up the data into a training and a test set (test ratio of 0.2).
banknote = fetch_openml('banknote-authentication')
X=banknote.data #features
y=banknote.target #target
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=7)



# b) Train a Decision Tree Classifier with max_depth=2 and visualize the resulting tree. Explain the visualization and how this tree can be used for predictions.
clf = tree.DecisionTreeClassifier(max_depth=2, random_state=7)
clf.fit(X_train,y_train)

plt.figure()
tree.plot_tree(clf, filled=True)
plt.show()




# c) Use a grid search to find the best parameter settings for the classifier. Search in depth values between 2 and 15, test both criteria (entropy and gini) for measuring the split quality, and test both criteria to select the split (best and random).
parameters = [{'max_depth': [2, 4, 6, 8, 10, 12, 14, 15],
               'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random']}]

gridsearch = model_selection.GridSearchCV(tree.DecisionTreeClassifier(random_state=7), param_grid=parameters)
gridsearch.fit(X_train, y_train)
print("Best Parameters Classifier: ", gridsearch.best_params_)



##================TASK 5: REGRESSION WITH DECISION TREES===================
#==========================================================================

#a) Fetch the OpenML dataset about cholesterol:

from sklearn.datasets import fetch_openml

# Fetch the cholesterol dataset from OpenML
cholesterol_data = fetch_openml(name='cholesterol')
X, y = cholesterol_data.data, cholesterol_data.target

#b) Train a Decision Tree Regressor and visualize the resulting tree:

from sklearn.tree import DecisionTreeRegressor, export_text
import graphviz
from sklearn.tree import export_graphviz
from IPython.display import Image

# Train a Decision Tree Regressor with max_depth=2
dt_regressor = DecisionTreeRegressor(max_depth=2)
dt_regressor.fit(X, y)

# Visualize the resulting tree
dot_data = export_graphviz(dt_regressor, out_file=None, feature_names=cholesterol_data.feature_names, filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("cholesterol_tree", format="png")

# Display the visualization
Image(filename="cholesterol_tree.png")

#c) Use a grid search to find the best parameter settings:

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'max_depth': list(range(2, 16)),
    'criterion': ['mse', 'friedman_mse'],
    'splitter': ['best', 'random']
}

# Create Decision Tree Regressor
dt_regressor = DecisionTreeRegressor()

# Perform Grid Search
grid_search = GridSearchCV(dt_regressor, param_grid, cv=5)
grid_search.fit(X, y)

# Print the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

