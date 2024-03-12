import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import ensemble
from sklearn import pipeline
from sklearn import linear_model
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn import svm

#===================TASK 1:   BAGGING K-MEANS ================================
#=============================================================================


print(" Task1 ", "\n")
# a) Fetch the Wisconsin Breast Cancer dataset with the respective scikit-learn methods and
# split into training and test set (test ratio of 0.2).

cancer_data = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, test_size=0.2, random_state=7)


# b) Train a k-Means clustering algorithm (𝑘=2) on the training data and
# inspect its performance on the test set w.r.t. the Adjusted Rand Index (ARI). 
# What does this measure express?

kmeans = KMeans(2, random_state=7)
kmeans.fit(X_train, y_train)
y_pred = kmeans.predict(X_test)
print("ARI performance: ", metrics.adjusted_rand_score(y_test, y_pred))
# It measures the similarity or degree of agreement.

# c) Train a BaggingClassifier with k-Means (𝑘=2) as the base estimator.
# The classifier should use all features and 30 % of the training examples for each created classifier.
# Also, it should be allowed to use the same example for different classifier instances.
# There should be 20 estimators used during bagging.

bagging = ensemble.BaggingRegressor(base_estimator=KMeans(2, random_state=7), n_estimators=20,
                                    random_state=7, max_samples=0.3, bootstrap=True).fit(X_train, y_train)


# d) Evaluate the ARI performance of the bagging classifier.
# How does it perform?
# In which scenarios is bagging probably outperforming a single estimator?


y_pred = bagging.predict(X_test)
print("ARI performance of the bagging classifier: ", metrics.adjusted_rand_score(y_test, y_pred))
# performs little better, Bagging used to reduce the variance
print("\n")


#===================TASK 2:  RANDOM FOREST  ==================================
#=============================================================================


print(" task 2", "\n")
# a) Generate some synthetic regression data with the respective scikit-learn method


data = datasets.make_regression(n_samples=1000, n_features=20, n_informative=10, noise=0.2, random_state=7)

# b) Build a pipeline that scales the data and afterwards applies a Random Forest regressor.
# Set the parameters of the Random Forest to 50 estimators and a maximum number of ten leaf nodes.
# How does a Random Forest estimator work?

pipe = pipeline.make_pipeline(StandardScaler(), ensemble.RandomForestRegressor(n_estimators=50,
                                                                               max_leaf_nodes=10, random_state=7))



# Random Forest is a multiple Decision Tree based algorithm, using a random factor to build multiple decision trees, 
# to get rid of bias

# c) Train this pipeline and measure its performance on the test set. Use a suitable performance measure.

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print("ARI performance: ", metrics.adjusted_rand_score(y_test, y_pred))

# d) Check the importance values of the features estimated by the Random Forest.
# Are they compliant with the generated data?
print(pipe.steps[:])
print(pipe.steps[1][1].feature_importances_)
# only three of the features are given importance
print("\n")


#===================TASK 3:  ADABOOST WITH LOGISTIC REGRESSION ==============
#=============================================================================


print(" Task 3", "\n")
# a) Fetch the Wisconsin Breast Cancer dataset with the respective scikit-learn methods
# and split into training and test set (test ratio of 0.2). Scale the data to [0,1].

cancer_data = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, test_size=0.2, random_state=7)
scaler = MinMaxScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.fit_transform(X_test)

# b) Train a Logistic Regression that classifies the training data. Use default parameters.
# What is the performance of the trained model w.r.t. accuracy?

logReg = linear_model.LogisticRegression(random_state=7).fit(X_train, y_train)
y_pred = logReg.predict(X_test)
print("ARI performance of standard approach: ", metrics.adjusted_rand_score(y_test, y_pred))

# c) Train an AdaBoostClassifier with Logistic Regression (same configuration as before) as the base estimator.
# There should be 20 estimators used during boosting.

adaBoost = ensemble.AdaBoostClassifier(base_estimator=linear_model.LogisticRegression(random_state=7),
                                       n_estimators=20).fit(X_train, y_train)

# d) Compare the accuracy on the test set of the boosted approach and the standard approach.
# How does AdaBoost work?
y_pred = adaBoost.predict(X_test)
print("ARI performance of boosted approach: ", metrics.adjusted_rand_score(y_test, y_pred))
# adaboost increase the efficiency by classifying correctly
print("\n")


#===================TASK 4: STACKING MULTIPLE REGRESSORS =====================
#=============================================================================

print("Task 4", "\n")
# a) Fetch the diabetes dataset and split the data into train and test with 20% test data.

diabetes_data = datasets.load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes_data.data, diabetes_data.target, test_size=0.2, random_state=7)


# b) Train three regressors (linear regression, decision tree, linear support vector machine) on this dataset.
# How do the individual models perform?

# Linear Regression
linear_reg = linear_model.LinearRegression()
linear_reg.fit(X_train, y_train)

# Decision Tree
decision_tree_reg = tree.DecisionTreeRegressor()
decision_tree_reg.fit(X_train, y_train)

# Linear Support Vector Machine (SVR)
svm_reg = svm.SVR()
svm_reg.fit(X_train, y_train)

# Evaluate performance
for model, name in zip([linear_reg, decision_tree_reg, svm_reg], ['Linear Regression', 'Decision Tree', 'Linear SVM']):
    y_pred = model.predict(X_test)
    mse = metrics.mean_squared_error(y_test, y_pred)
    print(f"{name} MSE: {mse}")



# c) Train a stacking regressor with the three types of models from part b) as base models
# and a linear regression as the final model. How does this model perform compared to the three individual models?

# Stacking Regressor
stacking_reg = ensemble.StackingRegressor(
    estimators=[('linear_reg', linear_reg), ('decision_tree_reg', decision_tree_reg), ('svm_reg', svm_reg)],
    final_estimator=linear_model.LinearRegression()
)
stacking_reg.fit(X_train, y_train)

# Evaluate performance
y_pred_stacking = stacking_reg.predict(X_test)
mse_stacking = metrics.mean_squared_error(y_test, y_pred_stacking)
print(f"Stacking Regressor MSE: {mse_stacking}")


#=======  OPTIONAL TASK 5 : PRINCIPAL COMPONENT ANALYSIS FOR DIMENSIONALITY REDUCTION===================
#=======================================================================================================
print(" Task 5", "\n")


# a) Create a synthetic regression dataset with ten toy example

# noinspection PyRedeclaration
data = datasets.make_regression(n_samples=10, n_features=2, n_targets=1, random_state=7)
X = data[0]
y = data[1]
df = pd.DataFrame(X, columns=['1', '2'])
df['targets'] = y  # not sure what to assign with

for column in df.columns:
    df[column] = (df[column] - df[column].mean())
# print(df)


# b) Visualize the data with matplotlib in a 3D plot where the data points are on the X- and Y-axis
# and the target values are on the Z-axis.

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(df['1'], df['2'], df['targets'])
ax.set_xlabel('1')
ax.set_ylabel('2')
ax.set_zlabel('targets')


# c) Perform a linear regression with the data and visualize the linear model together with the data in a 3D plot.
# You can simply add the linear model to the visualization from part b) of this exercise.

# d) Compute the Principal Components (PCs) of the data using the Singular Value Decomposition (SVD) of numpy.
# Project all data points onto the prior computed axis resulting from SVD.

U, Sigma, Vh = np.linalg.svd(X, full_matrices=False, compute_uv=True)
X_svd = np.dot(U, np.diag(Sigma))
print(X_svd)

# e) Visualize the resulting data and the respective targets in a 2D scatter plot.
# Compare it with the visualization of the data transformed by the PCA estimator
# that is provided by scikit-learn.

fig = plt.figure()
plt.scatter(X_svd, U)  # blue points

pca = decomposition.PCA().fit(X)
X_pca = pca.transform(X)
plt.scatter(X_pca, U)  # orange points

plt.show()


