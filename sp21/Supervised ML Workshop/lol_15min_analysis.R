# 2020-12-20

# setting working directory
setwd("~/Desktop/sisi")

# importing packages
install.packages("dplyr")
library(dplyr)
library(ggplot2)
library(factoextra)
library(NbClust)
library(tibble)
library(MLmetrics)
library(boot)
library(pscl)
library(survey)
library(caret)
library(pROC)
library(ROCR)
library(randomForest) 
library(party)
library(rpart)
library(corrplot)

# data preprocessing
df <- as_tibble(read.csv("MatchTimelinesFirst15.csv"))
summary(df)
names(df)
df <- select(df, 
             -c("X", "matchId", "blueDragonKills", "redDragonKills"))
summary(df)
df$blue_win <- as.factor(df$blue_win)
train_test_split <- function(df, train_size){
  train_id <- sample(nrow(df), floor(nrow(df)*train_size))
  df_train <- df[train_id, ]
  df_test <- df[-train_id, ]
  # y_train <- select(df_train, "blue_win")
  # X_train <- select(df_train, -c("blue_win"))
  # y_test <- select(df_test, "blue_win")
  # X_test <- select(df_test, -c("blue_win"))
  # datas <- list(y_train, X_train, y_test, X_test)
  datas <- list(df_train, df_test)
  return(datas)
}

# check null
cat("Null values: ", sum(sapply(df, is.null)))

# Logisitic Regression 
# train a 10k sample
df.10k <- df[sample(nrow(df), 10000), ]
datas <- train_test_split(df.10k, 0.75)
df_train_10k <- datas[[1]]
df_test_10k <- datas[[2]]

# baseline training
thres <- 0.5       # threshold
model.10k <- glm(blue_win ~ ., 
               family = binomial(link = "logit"),
               data = df_train_10k)
y_pred = predict(model.10k, select(df_test_10k, -c("blue_win")), 
                 type = "response")
y_pred <- as.factor(ifelse(y_pred >= thres, 1, 0))
cost_fn <- function(r, pi=0) {
  mean(abs(r - pi) > 0.5)
}
summary(model.10k)
confusionMatrix(y_pred, df_test_10k$blue_win)

# whole set training
df.all <- df
datas_all <- train_test_split(df.all, 0.75)
df_train <- datas_all[[1]]
df_test <- datas_all[[2]]

model_all <- glm(blue_win ~ ., 
                 family = binomial(link = "logit"),
                 data = df_train)
y_pred = predict(model_all, select(df_test, -c("blue_win")), 
                 type = "response")
y_pred <- as.factor(ifelse(y_pred >= thres, 1, 0))
confusionMatrix(y_pred, df_test$blue_win)
y_fitted <- as.factor(ifelse(model_all$fitted.values >= thres, 1, 0))
confusionMatrix(y_fitted, df_train$blue_win)

exp(model_all$coefficients)
  # coefficients: delete redDragonKills & blueDragonKills -- all zero

# Pseudo R Test
pR2(model_all)

# Wald Test: Variables Importance
for (n in names(select(df.all, -"blue_win"))) {
  reg <- regTermTest(model_all, n)
  cat(reg$test.terms, reg$p, "\n")
}

varImp(model_all)

# ROC (Receiving Operating Characteristic) & AUC
prob <- predict(model_all, 
                newdata=select(df_test, -c("blue_win")),
                type="response")
pred <- prediction(prob, df_test$blue_win)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, main = "ROC Curve")
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]
cat("AUC: ", auc)


# k-fold cross validation, k = 5 & 10
ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, savePredictions = TRUE)
mod_fit <- train(blue_win ~ .,  data=df.all, 
                 method="glm", family="binomial",
                 trControl = ctrl, tuneLength = 5)
pred = predict(mod_fit, newdata=select(df_test, -"blue_win"))
confusionMatrix(data=pred, df_test$blue_win)
summary(mod_fit)

# random forest classifier
rf <- randomForest(blue_win ~ .,  
                        data = df_train, importance = TRUE) 
rf
y_pred = predict(rf, newdata = select(df_test, -"blue_win")) 

# Confusion Matrix 
confusionMatrix(table(rf$predicted, df_train$blue_win))
confusionMatrix(table(y_pred, df_test$blue_win))

# Plotting model 
plot(rf, main = "Random Forest Result") 

# Importance plot 
importance(rf) 

# Variable importance plot 
varImpPlot(rf) 

# hierarchical clustering
df.hcut <- select(df.10k, -"blue_win")
df_scaled <- scale(df.hcut)
dist_mat <- dist(df_scaled, method = 'euclidean')
hclust_avg <- hclust(dist_mat, method = 'average')
plot(hclust_avg) #too long time

# visualize decision tree
png(filename = "decision_tree.png", 
    width = 5000, height = 5000)

# Create the tree.
output.tree <- ctree(
  blue_win ~ ., 
  data = df.all)

# Plot the tree.
plot(output.tree)

# Save the file.
dev.off()


# SVM
svm.mod = svm(formula = blue_win ~ ., 
                 data = df_train, 
                 type = 'C-classification', 
                 kernel = 'linear') 

# correlation matrix
corrplot(cor(df))


