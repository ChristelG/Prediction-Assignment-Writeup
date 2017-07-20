setwd("/Users/christel/desktop/coursera week8")

library(knitr)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)
library(parallel)
library(doParallel)

# Load the data
train_data <- read.csv('pml-training.csv')
test_data <- read.csv('pml-testing.csv')

# Filter the columns
train_data <- train_data[,8:dim(train_data)[2]]
test_data <- test_data[,8:dim(test_data)[2]]
NZV <- nearZeroVar(train_data)
train_data <- train_data[, -NZV]
test_data  <- test_data[, -NZV]
AllNA    <- sapply(train_data, function(x) mean(is.na(x))) > 0.95
train_data <- train_data[, AllNA==FALSE]
test_data  <- test_data[, AllNA==FALSE]

# Split the data
set.seed(122334444)
part <- createDataPartition(y=train_data$classe, p=0.7, list=FALSE)
train <- train_data[part,]
validation <- train_data[-part,]

## make the training go faster ##
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
# Train two models: Random Forest and SVM
model1 <- train(classe ~., data=train, method='rf', number=5)
model2 <- train(classe ~., data=train, method="svmLinear", number=5)
stopCluster(cl)

# Check performance and confusion matrix
predict_rf <- predict(model1, newdata=validation)
conf1 <- confusionMatrix(predict_rf, validation$classe)
print(conf1)
predict_svm <- predict(model2, newdata=validation)
conf2 <- confusionMatrix(predict_svm, validation$classe)
print(conf2)

# Predict classes for test data
predict_test <- predict(model1, newdata=test_data)
print(predict_test)
