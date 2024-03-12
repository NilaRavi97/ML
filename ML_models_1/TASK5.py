def t5():
    # TASK 5: LASSO REGRESSION
    import numpy as np
    import sklearn
    import sklearn.pipeline
    from sklearn.datasets import fetch_california_housing
    import pandas as pd
    from sklearn.linear_model import Lasso
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline


    california_housing = fetch_california_housing()

    features = fetch_california_housing()['feature_names']

    ldf = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    ldf["target"] = california_housing.target

    print(ldf)

    # Data split
    X_train, X_test, Y_train, Y_test = train_test_split(ldf.drop('target', axis=1), ldf['target'], test_size=0.2)

    # Create an instance of Lasso Regression implementation
    lasso = Lasso(alpha=1.0)

    # train the LASSO model
    lasso.fit(X_train, Y_train)

    # Create the model score
    #
    print('model score before removing features')
    print(lasso.score(X_test, Y_test))
    print(lasso.score(X_train, Y_train))

    pred = lasso.predict(X_test)

    # Mean absolute error
    print(mean_absolute_error(pred, Y_test))

    coef = lasso.coef_
    print(coef)
    # Yes. some coefficients of the trained model are closer or equal to zero
    # If the coefficient of the feature is not equal to zero, then it is considered by the model.
    # If the coefficient of the feature is equal to zero, then is not considered by the model.

    # features where coefficients are not equal to zero
    print(np.array(features)[coef != 0])
    coeff_not_equal_zero = np.array(features)[coef != 0]

    coeff_not_equal_zero = np.append(coeff_not_equal_zero,'target')

    print(california_housing.feature_names)

    # taking only the columns where coefficient is not zero
    ndf = ldf.copy()
    print(ndf.columns)

    ndf = ndf.drop(columns=[col for col in ndf if col not in coeff_not_equal_zero])
    print(ndf.columns)

    # after leaving the features
    # Data split
    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(ndf.drop('target', axis=1), ndf['target'], test_size=0.2)

    # Create an instance of Lasso Regression implementation
    lasso1 = Lasso(alpha=1.0)

    # train the LASSO model
    lasso1.fit(X_train1, Y_train1)

    # Create the model score
    #
    print('model score - after removing features')
    print(lasso1.score(X_test1, Y_test1))
    print(lasso1.score(X_train1, Y_train1))


