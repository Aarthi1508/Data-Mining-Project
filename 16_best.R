# XG Boost
library(xgboost)
load("C:\\Users\\sravy\\Downloads\\class_data.RData")
p = ncol(x)
n = nrow(x)
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
    
    # Train the XG Boost model and test
    set.seed(123)
    model <- xgboost(data = as.matrix(train_data[,-ncol(train_data)]),
                     label = train_data[,ncol(train_data)],
                     objective = "binary:logistic",
                     colsample_bynode = c(.25), 
                     eta = c(0.01),
                     gamma = c(0),
                     max_depth = c(9), 
                     nrounds = c(500),
                     verbose = 0)
    predictions <- predict(model, 
                           newdata = as.matrix(test_data[,-ncol(test_data)]))
    predictions = ifelse(predictions > 0.5, 1, 0)
    
    actuals <- test_data[,ncol(test_data)]
    error_rates[i] <- sum(predictions != actuals) / length(actuals)
}
test_error <- mean(error_rates)
cat(paste("Mean Error Rate : ", test_error))

#Final Model Training
set.seed(123)
final_model <- xgboost(data = as.matrix(data[,-ncol(data)]),
                 label = data[,ncol(data)],
                 objective = "binary:logistic",
                 colsample_bynode = c(.25), 
                 eta = c(0.01),
                 gamma = c(0),
                 max_depth = c(9), 
                 nrounds = c(500),
                 verbose = 0)
final_predictions <- predict(model, 
                       newdata = as.matrix(xnew))
ynew = ifelse(final_predictions > 0.5, 1, 0)

save(ynew,test_error,file="C:\\Users\\sravy\\Downloads\\16.RData")