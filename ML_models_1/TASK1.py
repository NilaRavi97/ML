def t1():

    import pandas as pd
    from matplotlib import pyplot as plt
    from sklearn.datasets import load_wine

    # TASK1 : Data analysis and visualization.

    # a) Analyze the data by using Python code and answer the questions by printing
    #  the console

    # load the wine data set
    wine_data = load_wine()

    # provides the information about the given dataset
    print(wine_data.DESCR)
    # 178 examples are contained in this dataset.
    # It contains 13 attributes.
    # attribute information in the output console

    # attribute names
    feature_names = wine_data.feature_names
    print(feature_names)

    # loading the wine data as datframe
    df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    print('wine data as dataframe', df)
    print('shape of the dataframe', df.shape[1])
    print('columns of the dataframe', df.columns)
    print('datatype of the columns')
    print(df.info())

    # Pick out the values of the attribute “ash”: What is the maximum,
    # minimum, median, and mean value for this attribute
    print('mean', df['ash'].mean())
    print('median', df['ash'].median())
    print('minimum', df['ash'].min())
    print('maximum', df['ash'].max())

    # No of classes with the label attribute
    target_names = wine_data.target_names
    print(target_names)
    # Three class labels associated with it
    # class 0, class 1, class 2
    df["target"] = wine_data.target
    print(df.target.unique())

    # Number of examples associated with each class.
    print(df.groupby(['target'])['target'].count())

    # b) Use matplotlib to visualize the data:
    #  Take the attribute “ash” and visualize the values as boxplot
    print(df['ash'])
    f = plt.figure(1)
    plt.boxplot(df['ash'])
    f.show()
    # Boxplot indicates the distribution of the data in the boxplot.
    # middle line indicates the average of the values of the attribute 'ash'.
    # top of the box indicates the upper quantile. Bottom of the box indicates the lower quantile.
    # outliers are represented as points outside the box.

    # Take the attribute “ash” and visualize the values as a discrete
    # histogram. Assume a number of 10 bins for the visualization
    g = plt.figure(2)
    plt.hist(df['ash'], bins=10)
    g.show()


