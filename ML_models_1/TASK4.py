def t4():
    # TASK 4: Regression
    from sklearn.datasets import fetch_california_housing
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.model_selection import train_test_split

    california_housing = fetch_california_housing()
    reg_df = pd.DataFrame(california_housing.data, columns = california_housing.feature_names)
    reg_df["target"] = california_housing.target

    print(reg_df.corr())

    # number of missing values in a dataframe
    print(reg_df.isna().sum())

    # Using Pearson Correlation
    plt.figure(figsize=(12,10))
    cor = reg_df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
    plt.show()

    # From the plot we can see that there is high correlation between average number of rooms and average number of bedrooms

    print(reg_df.columns)

    print(reg_df)

    print(california_housing.target_names)

    reg_df['HOL'] = ((reg_df.HouseAge > 25) & (reg_df.AveBedrms > 3))

    print(reg_df.HOL.unique())

    # converting boolean values to 0 and 1's
    reg_df['HOL'] = reg_df['HOL'].astype(int)

    print(reg_df.HOL.unique())

    # Data split
    X_train, X_test, Y_train, Y_test = train_test_split(reg_df.drop('target', axis=1), reg_df['target'], test_size=0.2)

    rclf = LinearRegression()

    rclf.fit(X_train, Y_train)


    # Model performance
    # r squared
    print(rclf.score(X_test, Y_test))

    pred = rclf.predict(X_test)

    # mean squared error
    print(mean_squared_error(pred, Y_test))

    # mean absolute error
    print(mean_absolute_error(pred, Y_test))

    # root mean squared error.
    # mean_square_error(yactual,ypredicted)
    rms = mean_squared_error(Y_test, pred, squared=False)
    print(rms)

    # If the MAE is much worse for the testing data as compared to the training data then this suggests
    # that our model might be overfitting the training data








