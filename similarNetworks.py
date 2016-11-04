import numpy as np
import pandas as pd
import renders as rs
from IPython.display import display # Allows the use of display() for DataFrames

# Show matplotlib plots inline (nicely formatted in the notebook)
#get_ipython().magic(u'matplotlib inline')


def load() :

    print "............................ Inside Data Loading ......................................................."
    # Load the wholesale customers dataset
    try:
        data = pd.read_csv("customers.csv")
        temp = pd.read_csv("customers.csv")
        data.drop(['Region', 'Channel'], axis = 1, inplace = True)
        temp.drop(['Region', 'Channel'], axis = 1, inplace = True)
        print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
    except:
        print "Dataset could not be loaded. Is the dataset missing?"

    return data, temp

def explore(data) :

    print "............................Inside Data Exploration ......................................................."

    # Display a description of the dataset
    display(data.describe())

    #Choosing a temp variable just to get the 'total' of each individual customers

    temp['total'] = temp.sum(axis=1)

    display(temp.describe())

    print temp[:10]

def selectSample() :

    indices = []

    indices.append(92)
    indices.append(200)
    indices.append(150)

    # Create a DataFrame of the chosen samples
    samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
    print "Chosen samples of wholesale customers dataset:"
    display(samples)

    # Create a DataFrame of the chosen samples
    samples1 = pd.DataFrame(temp.loc[indices], columns = temp.keys()).reset_index(drop = True)
    print "Chosen samples of wholesale customers dataset with totals:"
    display(samples1)

    return samples

def featureRelevance(data) :

    print "............................Inside Feature Relevance ......................................................."

    from sklearn.tree import DecisionTreeRegressor
    from sklearn import cross_validation
    from sklearn.metrics import r2_score

    # TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature

    target_cols = data.columns[-2]
    print target_cols
    y = data[target_cols]

    new_data = data.copy(deep=True)

    # Removing Milk as it is most correlated with others, it might turns out to be good target

    new_data.drop(['Detergents_Paper'], axis = 1, inplace = True)

    X = new_data

    seed = 7
    t_size = 0.25


    # TODO: Split the data into training and testing sets using the given feature as the target
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
      test_size=t_size, random_state=seed)


    # TODO: Create a decision tree regressor and fit it to the training set
    regressor = DecisionTreeRegressor(random_state = seed)
    regressor.fit(X_train, y_train)

    # TODO: Report the score of the prediction using the testing set
    predictions = regressor.predict(X_test)
    score = regressor.score(X_test, y_test)
    print score

def featureDistributions(data) :

    print "............................Inside Feature Distributions ......................................................."

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Produce a scatter matrix for each pair of features in the data
    pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

    # density
    data.plot(kind='density', subplots=True, layout=(3,2), sharex=False, legend=False,
    fontsize= 1)
    plt.show()

    import matplotlib.pyplot as plt
    names = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper' ,'Delicatessen' ]

    # correlation

    corr = data.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    #with sns.axes_style("white"):
    #    ax = sns.heatmap(corr, mask=mask, square=True, annot=True, cmap='RdBu')
    #    plt.xticks(rotation=45, ha='center');

    print "............................End Feature Distributions ......................................................."

def normalizeFeatures(data,samples) :

    import matplotlib.pyplot as plt

    print "............................ Inside Normalize / Feature scalaing using Log ......................................................."

    normalizedData = np.log(data)

    log_samples = np.log(samples)

    # Produce a scatter matrix for each pair of newly-transformed features
    pd.scatter_matrix(normalizedData, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

    #reviewer suggested to use log_data.boxplot(); but it didn't work well so I keeping my original.

    # box and whisker plots
    normalizedData.plot(kind='box', subplots=True, layout=(3,2), sharex=False, sharey=False,
    fontsize=8)
    plt.show()
    display(log_samples)

    print "............................ End Normalize / Feature scalaing using Log ......................................................."
    return normalizedData, log_samples

def removeOutliers(normalizedData) :

    print "............................Outliers removal ......................................................."

    from scipy import stats
    #from  more_itertools import unique_everseen

    # Keep outlier indices in a list and examine after looping thru the features
    idx = []


    # For each feature find the data points with extreme high or low values
    for feature in normalizedData.keys():

        # TODO: Calculate Q1 (25th percentile of the data) for the given feature
        Q1 = np.percentile(normalizedData[feature], 25)

        # TODO: Calculate Q3 (75th percentile of the data) for the given feature
        Q3 = np.percentile(normalizedData[feature], 75)

        # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
        step = 1.5*(Q3 - Q1)

        # Display the outliers
        print "Data points considered outliers for the feature '{}':".format(feature)
        display(normalizedData[~((normalizedData[feature] >= Q1 - step) & (normalizedData[feature] <= Q3 + step))])

        # Gather the indexes of all the outliers
        idx += normalizedData[~((normalizedData[feature] >= Q1 - step) & (normalizedData[feature] <= Q3 + step))].index.tolist()

    print(sorted(idx))

    # OPTIONAL: Select the indices for data points you wish to remove
    outliers  = []

    #outliers = list(unique_everseen(idx))

    import collections
    outliers = [item for item, count in collections.Counter(idx).items() if count > 1]


    print(sorted(outliers))

    # Remove the outliers, if any were specified
    good_data = normalizedData.drop(normalizedData.index[outliers]).reset_index(drop = True)


    print(normalizedData.shape)
    print(good_data.shape)

    print "............................End Outliers removal ......................................................."

    return good_data

def performPCA(good_data,log_samples) :

    print "............................PCA ......................................................."

    from sklearn.decomposition import PCA

    # TODO: Apply PCA to the good data with the same number of dimensions as features

    pca = PCA(n_components=6).fit(good_data)

    # TODO: Apply a PCA transformation to the sample log-data
    pca_samples = pca.transform(log_samples)
    #pca.fit(log_samples)

    # Generate PCA results plot
    pca_results = rs.pca_results(good_data, pca)

    # cumulative explaned variance
    #print '\n', np.cumsum(pca.explained_variance_ratio_)

    # Display sample log-data after having a PCA transformation applied
    display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))


    # TODO: Fit PCA to the good data using only two dimensions
    pca = PCA(n_components=2).fit(good_data)

    # TODO: Apply a PCA transformation the good data
    reduced_data = pca.transform(good_data)

    # TODO: Apply a PCA transformation to the sample log-data
    pca_samples = pca.transform(log_samples)

    # Create a DataFrame for the reduced data
    reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

    display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))
    print reduced_data[:10]

    return pca,reduced_data,pca_samples

def performClustering(reduced_data,pca_samples) :

    print "............................Inside Clustering ......................................................."

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    #keep the scores for each cluster size
    sil_scores = []

    random_state = 7

    for i in range(7,1,-1):
        clusterer = KMeans(i, random_state=random_state).fit(reduced_data)
        # TODO: Predict the cluster for each data point
        preds = clusterer.predict(reduced_data)

        # TODO: Find the cluster centers
        centers = clusterer.cluster_centers_

        # TODO: Predict the cluster for each transformed sample data point
        sample_preds = clusterer.predict(pca_samples)

        # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
        score = silhouette_score(reduced_data, preds)
        sil_scores.append(score)
        print i, 'clusters:', score.round(5)

    # plot the scores
    import matplotlib.pyplot as plt
    _ = plt.plot(np.arange(7,1,-1), sil_scores, '-o')

    print "............................End Clustering ......................................................."

    rs.cluster_results(reduced_data, preds, centers, pca_samples)

    return preds, centers, pca_samples

def identifyCenter(pca,reduced_data, preds, centers, pca_samples) :

    print "............................Identify Centers ......................................................."

    rs.cluster_results(reduced_data, preds, centers, pca_samples)

    log_centers = pca.inverse_transform(centers)
    true_centers = np.exp(log_centers)

    # Display the true centers
    segments = ['Segment {}'.format(i) for i in range(0,len(centers))]

    true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
    true_centers.index = segments
    true_centers['total'] = true_centers.sum(axis=1)

    display(true_centers)

    #true_centers.plot(kind = 'bar', figsize = (10, 4))

def displayPredictons(sample_preds,outliers,pca_samples) :

    #Display the predictions
    #print sample_preds

    for i, pred in enumerate(sample_preds):
        print "Sample point", i, "predicted to be in Cluster", pred

    rs.channel_results(reduced_data, outliers, pca_samples)


if __name__ == '__main__':

    data, temp = load()
    explore(data)
    samples = selectSample()
    featureRelevance(data)
    featureDistributions(data)
    normalizedData,log_samples = normalizeFeatures(data,samples)
    good_data = removeOutliers(normalizedData)
    pca,reduced_data,pca_samples = performPCA(good_data,log_samples)
    preds, centers, pca_samples = performClustering(reduced_data,pca_samples)
    identifyCenter(pca,reduced_data, preds, centers, pca_samples)
    #displayPredictons(sample_preds,outliers,pca_samples)

