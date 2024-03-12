def t2and3():

    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_wine
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import minmax_scale

    # TASK2 : Data preparation.
    # a) create a new attribute

    # load the wine data set
    wine_data = load_wine()

    # create a new attribute “substracted_phenols” by subtracting the values of
    # “nonflavanoid_phenols” from the values of “total_phenols”

    df1 = pd.DataFrame(data=np.c_[wine_data['data'], wine_data['target']],
                       columns=wine_data['feature_names'] + ['target'])

    df1['substracted_phenols'] = df1['total_phenols'] - df1['nonflavanoid_phenols']
    print('dataframe with subtracted phenols')
    print(df1)

    # Increase all values of the attribute “alcohol” by 1.0
    df1['alcohol'] = df1['alcohol'] + 1.0

    # b) Data augumentation
    # sample method samples all the rows without replacement.
    rdf = df1.sample(20)
    print(rdf)
    new_set = list()
    print('shape', rdf.shape)

    # select 20 random examples from the original dataset and
    # apply Gaussian noise (mean = 0 and std. dev. = 1) to those examples
    for i in range(rdf.shape[0]):
        row = rdf.iloc[i, rdf.columns != 'target']
        mu, sigma = 0, 1
        gauss_noise = np.random.normal(mu, sigma, len(row))
        new_row = row + gauss_noise
        new_set.append(new_row)
    ndf = pd.DataFrame(new_set)
    print(ndf)

    # append it to the dataframe
    fdf = df1.append(ndf, ignore_index=True)
    print(fdf)

    print('Before adding', df1.shape)
    print('After adding', fdf.shape)

    print(fdf.columns)

    # c) training preparation
    print(fdf['ash'])

    print('minmaxscale')
    # min max scale scales the output between 0 and 1
    fdf['ash'] = minmax_scale(fdf['ash'])
    print(fdf['ash'])

    # I chose min-max scaler because it transforms features by scaling each feature to the given range.


    # TASK 3: classification
    # np.c_ is the numpy concatenate function
    # which is used to concat iris['data'] and iris['target'] arrays
    data1 = pd.DataFrame(data=np.c_[wine_data['data'], wine_data['target']],
                         columns=wine_data['feature_names'] + ['target'])

    print(data1)
    print(data1.info())
    data1['target'] = data1['target'].apply(lambda x: str(x))

    # Transform the problem to a binary classification task by combining labels of
    # class 1 and 2. Name the resulting labels class-0 and not-class-0
    data1['target'] = data1['target'].str.replace('1.0', 'class_not_0', regex=True)
    data1['target'] = data1['target'].str.replace('2.0', 'class_not_0', regex=True)
    data1['target'] = data1['target'].str.replace('0.0', 'class_0', regex=True)

    print(data1.columns)

    # Split the original data in a training set and a test set with a training ratio of
    # 0.85.
    # Data split
    X_train, X_test, Y_train, Y_test = train_test_split(data1.drop('target', axis=1), data1['target'], test_size=0.15)

    # train a classifier of your choice with the training data
    clf = RandomForestClassifier()

    # build the Random forest model
    # trains the model based on the training data
    clf.fit(X_train, Y_train)

    # performs prediction on the test set
    print('predicted class labels')
    print(clf.predict(X_test))

    print('Actual class labels')
    print(Y_test)

    predicted = clf.predict(X_test)
    expected = Y_test

    # accuracy score
    print('accuracy score', accuracy_score(expected, predicted))

    # confusion matrix
    results = confusion_matrix(expected, predicted)
    print(results)

    # Random classifier classifies the data so accurately. It gave a model performance score 1.0.
    # In the confusion matrix, there is no false positives/false negatives

    print('Dataframe from task 2', fdf.columns)

    # dataframe from task 2
    df2 = fdf.copy()

    # repeat task3 for the dataframe from task 2
    df2['target'] = df2['target'].apply(lambda x: str(x))

    df2['target'] = df2['target'].str.replace('1.0', 'class_not_0', regex=True)
    df2['target'] = df2['target'].str.replace('2.0', 'class_not_0', regex=True)
    df2['target'] = df2['target'].str.replace('0.0', 'class_0', regex=True)

    print(df2.columns)

    # Data split
    X_train, X_test, Y_train, Y_test = train_test_split(df2.drop('target', axis=1), df2['target'], test_size=0.15)

    clf1 = RandomForestClassifier()

    # build the Random forest model
    # trains the model based on the training data
    clf1.fit(X_train, Y_train)

    # performs prediction on the test set
    print('predicted class labels')
    print(clf1.predict(X_test))

    print('Actual class labels')
    print(Y_test)

    # Model performance
    print(clf1.score(X_test, Y_test))

    predicted = clf1.predict(X_test)
    expected = Y_test

    # confusion matrix
    results = confusion_matrix(expected, predicted)
    print(results)

    # I don't think transformation is useful in this particular case. Since the data before transformation
    # provides better performance score than the data after transformation.

