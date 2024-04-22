# ----------------------------SUPERVISED LEARNING----------------------------------

# Logistic Regression with regularization
library(glmnet)

load("C:\\Users\\sravy\\Downloads\\class_data.RData")
data = cbind(x,y)
data <- as.matrix(data)
alphas <- seq(0, 1, 0.05)
mean_error_rate_lg_list <- numeric(length(alphas))

#CV parameters
cv_folds <- 5
fold_size <- floor(nrow(data) / cv_folds)
error_rates <- rep(0, cv_folds)

for(j in seq_along(alphas)){
  for (i in 1:cv_folds) {
    # Split the data into training and testing sets
    set.seed(1234)
    test_indices <- ((i-1) * fold_size + 1):(i * fold_size)
    test_data <- data[test_indices, ]
    train_data <- data[-test_indices, ]
    
    # Train a KNN model and test
    set.seed(123)
    model <- cv.glmnet(train_data[,-ncol(train_data)], 
                       train_data[,ncol(train_data)],
                       alpha = alphas[j],
                       family = "binomial")
    optimal_alpha <- model$lambda.1se
    
    predictions <- predict(model, 
                           newx = test_data[,-ncol(test_data)], 
                           s = optimal_alpha, 
                           type = "response")
    predictions <- ifelse(predictions > 0.5, 1, 0)
  
    actuals <- test_data[,ncol(test_data)]
    
    error_rates[i] <- sum(predictions != actuals) / length(actuals)
  }
  mean_error_rate_lg_list[j] <- mean(error_rates)
}

best_alpha <- alphas[which.min(mean_error_rate_lg_list)]  # 0.05
mean_error_rate_lg <- min(mean_error_rate_lg_list)  # 0.37



# KNN
library(caret)
library(class)

load("C:\\Users\\sravy\\Downloads\\class_data.RData")
y = as.factor(y)
data = cbind(x,y)
k_vals <- seq(1, 50)
mean_error_rate_knn_list <- numeric(length(k_vals))

#CV parameters
cv_folds <- 5
fold_size <- floor(nrow(data) / cv_folds)
error_rates <- rep(0, cv_folds)

# Perform cross-validation
for (j in seq_along(k_vals)) {
  for (i in 1:cv_folds) {
  # Split the data into training and testing sets
  set.seed(1234)
  test_indices <- ((i-1) * fold_size + 1):(i * fold_size)
  test_data <- data[test_indices, ]
  train_data <- data[-test_indices, ]
  
  # Train a KNN model and test
  set.seed(123)
  knn_predictions <- knn(train_data[,-ncol(train_data)], 
                         test_data[,-ncol(test_data)], 
                         train_data[,ncol(train_data)], 
                         k_vals[j])
  actuals <- test_data[,ncol(test_data)]
  
  error_rates[i] <- sum(knn_predictions != actuals) / length(actuals)
  
  }

  # Calculate the mean error rate across all folds
  mean_error_rate_knn_list[j] <- mean(error_rates)
}
best_knn <- k_vals[which.min(mean_error_rate_knn_list)]    # 6
mean_error_rate_knn <- mean_error_rate_knn_list[best_k]  # 0.2825


#Naive Bayes 
library(e1071)
library(caret)

load("C:\\Users\\sravy\\Downloads\\class_data.RData")
y = as.factor(y)
data = cbind(x,y)

#CV parameters
cv_folds <- 5
fold_size <- floor(nrow(data) / cv_folds)
error_rates <- rep(0, cv_folds)

for (i in 1:cv_folds) {
  # Split the data into training and testing sets
  set.seed(1234)
  test_indices <- ((i-1) * fold_size + 1):(i * fold_size)
  test_data <- data[test_indices, ]
  train_data <- data[-test_indices, ]
  
  # Train the Naive Bayes and test
  set.seed(123)
  model <- naiveBayes(y ~ ., data = train_data)
  predictions <- predict(model, newdata = test_data)
  actuals <- test_data$y
  
  error_rates[i] <- sum(predictions != actuals) / length(actuals)
}
mean_error_rate_nb <- mean(error_rates)  # 0.3725

#Random Forest
library(randomForest)

load("C:\\Users\\sravy\\Downloads\\class_data.RData")
y = as.factor(y)
data = cbind(x,y)
p = ncol(x)
mtry = c(floor(sqrt(p)), floor(sqrt(p)) + 50, seq(100, 500, 50))
mean_error_rate_rf_list <- numeric(length(mtry))

#CV parameters
cv_folds <- 5
fold_size <- floor(nrow(data) / cv_folds)
error_rates <- rep(0, cv_folds)

for(j in seq_along(mtry)) {
  for (i in 1:cv_folds) {
    # Split the data into training and testing sets
    set.seed(1234)
    test_indices <- ((i-1) * fold_size + 1):(i * fold_size)
    test_data <- data[test_indices, ]
    train_data <- data[-test_indices, ]
    
    # Train the random forest and test
    set.seed(123)
    rf_model <- randomForest(y ~ ., 
                             data = train_data, 
                             ntree = 1000, 
                             mtry = mtry[j])
    predictions <- predict(rf_model, newdata = test_data)
    actuals <- test_data$y
    
    error_rates[i] <- sum(predictions != actuals) / length(actuals)
  }
  mean_error_rate_rf_list[j] <- mean(error_rates)
}
best_rf_mtry <- mtry[which.min(mean_error_rate_rf_list)]    # 100
mean_error_rate_rf <- min(mean_error_rate_rf_list)  # 0.3175

#SVM
library(e1071)  
library(caret) 

load("C:\\Users\\sravy\\Downloads\\class_data.RData")
y = as.factor(y)
data = cbind(x,y)
#SVM parameters
linear = expand.grid(cost = c(0.1, 1, 10, 100, 1000, 10000),
                     gamma = c(0.1, 1, 10, 100),
                     kernel = "linear",
                     degree = 1,
                     stringsAsFactors = F)
polynomial = expand.grid(cost = c(0.1, 1, 10, 100, 1000, 10000),
                   gamma = c(0.1, 1, 10, 100),
                   kernel = "polynomial",
                   degree = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                   stringsAsFactors = F)
radial = expand.grid(cost = c(0.1, 1, 10, 100, 1000, 10000),
                     gamma = c(0.1, 1, 10, 100),
                     kernel = "radial",
                     degree = 1,
                     stringsAsFactors = F)

grid = rbind.data.frame(linear, polynomial, radial)
mean_error_rate_svm_list <- numeric(nrow(grid))

#CV parameters
cv_folds <- 5
fold_size <- floor(nrow(data) / cv_folds)
error_rates <- rep(0, cv_folds)


for(j in seq_len(nrow(grid))){
  for (i in 1:cv_folds) {
    # Split the data into training and testing sets
    set.seed(1234)
    test_indices <- ((i-1) * fold_size + 1):(i * fold_size)
    test_data <- data[test_indices, ]
    train_data <- data[-test_indices, ]
    
    # Train the SVM model and test
    set.seed(123)
    model = tryCatch(svm(y ~ ., 
                data = train_data, 
                scale=F, 
                type="C", 
                kernel = grid[j, "kernel"],
                degree = grid[j, "degree"],
                cost = grid[j, "cost"],
                gamma= grid[j, "gamma"],
                coef0 = 1), 
                error = function(e) Inf)
    predictions <- tryCatch(predict(model, newdata = test_data), 
                            error = function(e) Inf)
    actuals <- test_data$y
    
    error_rates[i] <- sum(predictions != actuals) / length(actuals)
  }
  mean_error_rate_svm_list[j] <- mean(error_rates)
}
best_svm <- grid[which.min(mean_error_rate_svm_list), ] # cost=0.1,gamma=0.1 
                                                        # kernel=poly, 
                                                        # degree=4
mean_error_rate_svm <- min(mean_error_rate_svm_list)  # 0.2975

# XG Boost
library(xgboost)
load("C:\\Users\\sravy\\Downloads\\class_data.RData")
p = ncol(x)
n = nrow(x)
data = cbind(x,y)

grid = expand.grid(colsamp = c(sqrt(p)/n, .25, .5, 1),
                         gamma= c(0, 0.1, 0.5, 1),
                         nrounds = c(500),
                         max_depth = c(5,7,9),
                         eta = c(0.0001, 0.001, 0.01, 0.1)
)
mean_error_rate_xgb_list <- numeric(nrow(grid))


#CV parameters
cv_folds <- 5
fold_size <- floor(nrow(data) / cv_folds)
error_rates <- rep(0, cv_folds)

for(j in seq_len(nrow(grid))){
  for (i in 1:cv_folds) {
    # Split the data into training and testing sets
    set.seed(1234)
    test_indices <- ((i-1) * fold_size + 1):(i * fold_size)
    test_data <- data[test_indices, ]
    train_data <- data[-test_indices, ]
    
    # Train the XG Boost model and test
    set.seed(123)
    
    model <- xgboost(data = as.matrix(train_data[,-ncol(train_data)]),
            label = train_data[,ncol(train_data)],
            objective = "binary:logistic",
            colsample_bynode = grid[ j, "colsamp" ], 
            eta = grid[ j, "eta" ],
            gamma = grid[j, "gamma"],
            max_depth = grid[ j, "max_depth" ], 
            nrounds = grid[ j, "nrounds" ],
            verbose = 0)
    predictions <- predict(model, 
                           newdata = as.matrix(test_data[,-ncol(test_data)]))
    predictions = ifelse(predictions > 0.5, 1, 0)
    
    actuals <- test_data[,ncol(test_data)]
    error_rates[i] <- sum(predictions != actuals) / length(actuals)
  }
  mean_error_rate_xgb_list[j] <- mean(error_rates)
}
best_xgb <- grid[which.min(mean_error_rate_xgb_list), ] # colsamp=0.25
                                                        # gamma=0
                                                        # nrounds=500
                                                        # max_depth=9
                                                        # eta=0.01
mean_error_rate_xgb <- min(mean_error_rate_xgb_list)  # 0.2825


# ----------------------------UNSUPERVISED LEARNING----------------------------------

library(mclust)
library(mixtools)
library(ggcorrplot)
library(NbClust)
library(corrplot)
library (cluster)
library(factoextra)

cluster_data <- load("C:/Users/Aarthi Vasudevan/Texas A&M University/Semesters_&_Coursework/Spring 2023/STAT 639 - Data Mining and Analysis/Project/cluster_data.RData")
data <- y

set.seed(123)
# 1. PRINCIPAL COMPONENT ANALYSIS
pca <- prcomp(data, scale = TRUE, center = TRUE)

# Extracting eigenvalues from PCA
eigenvalues <- pca$sdev^2

# Computing cumulative proportion of variance explained
cumulative_proportion <- cumsum(eigenvalues) / sum(eigenvalues)

# Creating a scree plot
plot(1:10, cumulative_proportion[1:10], type = "b", xlab = "Number of Principal Components", ylab = "Cumulative Proportion of Variance Explained", main = "Scree Plot")

# Finding the elbow point
elbow_point <- which(diff(cumulative_proportion[1:10]) < 0.025)[1] # Setting the threshold as 0.025 for difference of cumulative proportion variance

abline(v = elbow_point, lty = "dashed", col = "red")
abline(h = cumulative_proportion[elbow_point], v = elbow_point, col = "red", lty = "dashed")  

legend("bottomright", legend = "Elbow Point", pch = 19,  col = "red",  cex = 1, bty = "n")
points(elbow_point, cumulative_proportion[elbow_point], pch = 19, col = "red", cex = 1)  
symbols(elbow_point, cumulative_proportion[elbow_point], circles = 0.1, inches = FALSE, add = TRUE)

cat("Optimal no of principal components returned by PCA = ", elbow_point)

#2. K-MEANS CLUSTERING
set.seed(124)
#(i) Silhouette plot
silhouette_score <- function(k){
  km <- kmeans(data, centers = k, nstart=25)
  ss <- silhouette(km$cluster, dist(y)) #Calculating silhouette score
  mean(ss[,3])
}
k <- 2:10
avg_sil <- sapply(k, silhouette_score)
plot(k, type='b', avg_sil, xlab='Number of clusters', ylab='Average Silhouette Scores', frame=FALSE)

# Finding the optimal number of clusters with the highest silhouette score
optimal_k <- k[which.max(avg_sil)]
cat("Optimal number of clusters based on silhouette score: ", optimal_k, "\n")

#(ii) Elbow plot
set.seed(125)
#Plot 2. WCSS Plot
k_values <- 2:10
wcss <- numeric(length(k_values))
for (k in k_values) {
  kmeans_result <- kmeans(data, centers = k)
  wcss[k-1] <- sum(kmeans_result$withinss)
}

# Plotting the within-cluster sum of squares (WCSS) against number of clusters
plot(k_values, wcss, type = "b", xlab = "Number of clusters", ylab = "WCSS", main = "Elbow Plot")

set.seed(126)
#K-means clustering for optimal k based on Silhouette score and WCSS
optimal_clustering <- kmeans(scale(data), optimal_k, nstart = 25)

#(iii) Cluster plot
fviz_cluster(optimal_clustering, data = scale(data))

#3. AGGLOMERATIVE HIERARCHICAL CLUSTERING

set.seed(127)

data <- data.frame(y)
d <- dist(data, method = "euclidean")
res.agnes <- agnes(y, method = "ward")
res.agnes$ac

pltree(res.agnes, cex = 0.6, hang = -1, main = "Dendrogram of Agnes") 
plot(as.hclust(res.agnes), cex = 0.6, hang = -1)
plot(as.dendrogram(res.agnes), cex = 0.6, horiz = TRUE)

#(i) Silhouette plot
silhouette_score <- function(k){
  grp <- cutree(res.agnes, k = k)
  table(grp)
  ss <- silhouette(grp, dist(y, method = "euclidean"))
  mean(ss[,3])
}
k <- 2:10
avg_sil <- sapply(k, silhouette_score)
plot(k, type='b', avg_sil, xlab='Number of clusters', ylab='Average Silhouette Scores', frame=FALSE)
# Find the optimal number of clusters with the highest silhouette score
optimal_k <- k[which.max(avg_sil)]
print(avg_sil)
cat("Optimal number of clusters based on silhouette score: ", optimal_k, "\n")

#(ii) Elbow plot
n_clusters_range <- 2:10 # Try different numbers of clusters
wcss <- numeric(5)
for (i in seq(2,10,by =2)) {
  clustering <- agnes(data, method = "ward")
  clustering <- cutree(clustering, k = i) # Cut the tree to get i clusters
  cluster_centers <- colMeans(data[clustering == 1:i, ]) # Compute cluster centers
  wcss[i/2] <- sum((data - cluster_centers[clustering])^2) # Compute WCSS
}

#(iii) Cluster plot
fviz_cluster(list(data = data, cluster = grp))

#4. T-SNE
set.seed(128)
data <- y
tsne_res <- tsne(data, perplexity = 30)

#(i) Silhouette Plot
silhouette_score <- function(k){
  km <- kmeans(tsne_res, centers = k, nstart=25)
  ss <- silhouette(km$cluster, dist(tsne_res))
  mean(ss[,3])
}

k <- 2:10
avg_sil <- sapply(k, silhouette_score)
plot(k, type='b', avg_sil, xlab='Number of clusters', ylab='Average Silhouette Scores', frame=FALSE)

# Finding the optimal number of clusters with the highest silhouette score
optimal_k <- k[which.max(avg_sil)]
cat("Optimal number of clusters based on silhouette score: ", optimal_k, "\n")

#(ii) Elbow plot
# Performing k-means clustering on t-SNE embeddings for different number of clusters
max_clusters <- 10 # maximum number of clusters to consider
wcss <- numeric(max_clusters) # to store WCSS for each number of clusters

for (k in 1:max_clusters) {
  # Perform k-means clustering on t-SNE embeddings
  kmeans_result <- kmeans(tsne_res, centers=k)
  # Calculate WCSS for the current number of clusters
  wcss[k] <- kmeans_result$tot.withinss
}

# Plotting the WCSS vs. number of clusters
plot(1:max_clusters, wcss, type="b", pch=19, xlab="Number of Clusters", ylab="WCSS")
km <- kmeans(tsne_res, centers = optimal_k, nstart=25)

#(iii) Cluster plot
fviz_cluster(km, data = scale(data))

