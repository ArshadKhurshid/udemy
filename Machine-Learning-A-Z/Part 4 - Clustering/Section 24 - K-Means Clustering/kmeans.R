#Kmeans Clustering

dataset <- read.csv('Mall_customers.csv')
X <- dataset[4:5]

#using the elbow method to fins nu number of clustering
set.seed(6)
wcss <- vector()
for (i in 1:10) wcss[i] <- sum(kmeans(X,i)$withinss)
plot(1:10, wcss, type = 'b', main = paste('Cluster of client'), xlab = 'Cluster of clients', ylab = 'wcss')

#Applying k-means to mall dataset
set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 100,nstart = 10)

#Visulalizing the clusters
library(cluster)
clusplot(X,
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Cluster of clients'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')