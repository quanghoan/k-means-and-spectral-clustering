import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#load and print the data
data = pd.read_csv('data/Wholesale customers data.csv')
print(data.head())

#print the sum of null values
#print(data.isnull().sum())

#in this project, our aim is to cluster the data so that we can see the products, which are bought by the customer together. Therefore do not need the colmns 'Channel' and 'Region' for this analysis.Hence we will drop these two column

data = data.drop(labels=['Channel', 'Region'], axis=1)
#print(data.head())

#now we perform the clustering as below. Note that, the 'Normalizer()' is used
#for the preprocessing. We can try the different preprocessing methods (Ex: MaxAbsScaler, StandardScaler), to visualize the outputs and performance

#preprocessing
T = preprocessing.Normalizer().fit_transform(data)

#change n_clusters to 2, 3, 4 ... to see the output patters
n_clusters = 6

#clustering using KMeans
kmean_model = KMeans(n_clusters=n_clusters)
kmean_model.fit(T)
centroids, labels = kmean_model.cluster_centers_, kmean_model.labels_
#print(centroids)

#Now, we will perform the dimensionality reduction using PCA. We will reduce the dimensions to 2
#Since we have few features in this dataset, we are performing the clustering first and then dimensionality reduction
#If we have a very large number of features, then it is better to perform dimensionality rediction first and then use the clustering algorithm

#Dimensionality reduction to 2
pca_model = PCA(n_components=2)
pca_model.fit(T)
T = pca_model.transform(T) #transform the normalized model
#transform the centroids of KMean
centroid_pca = pca_model.transform(centroids)
#print(pca_model)

colors = ['blue', 'red', 'green', 'orange', 'black', 'brown']
#assing a color to each features(note that we are using features)
features_colors = [colors[labels[i]] for i in range(len(T))]

#plot the PCA components
plt.scatter(T[:,0], T[:,1], c=features_colors, marker='o', alpha=0.5)

# plot the centroids
plt.scatter(centroid_pca[:, 0], centroid_pca[:, 1],
            marker='x', s=100,
            linewidths=3, c=colors
        )

#sore the values of PCA component in variable for easy writing
xvertor = pca_model.components_[0] * max(T[:,0])
yvertor = pca_model.components_[1] * max(T[:,1])
columns = data.columns

#plot the name of indivisual features along with vertor length
for i in range(len(columns)):
	plt.arrow(0,0,xvertor[i],yvertor[i],
		color='b',width=0.005,head_width=0.02,alpha=0.8)
	plt.text(xvertor[i],yvertor[i],list(columns)[i],
		color='b',alpha=0.8)
plt.show()



