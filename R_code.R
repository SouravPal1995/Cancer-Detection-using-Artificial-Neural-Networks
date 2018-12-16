
#installation of essential packages
install.packages("ggplot2")
install.packages("pROC")
install.packages("ROCR")
install.packages("caret", dependencies = TRUE)
install.packages("neuralnet")
install.packages("nnet")
install.packages("caTools")
install.packages("Rccp")
install.packages("purrr")
install.packages("dplyr")
library(pROC)
library(nnet)
library(neuralnet)
library(caret)
library(e1071)
library(caTools)
library(ggplot2)
library(ROCR)
library(MASS)
library(purrr)
library(caret)
library(Matrix)
library(tidyr)

#importing of data


getwd()
data <- read.table(file = "cancer.csv", header=FALSE, sep=",")

data <- read.table(
  file = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", 
  header = FALSE, sep = ",")


data <- data[,-1]
n <- c("diag", "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
       "mean_smoothness", "mean_compactness", "mean_concavity", "mean_concave_points", 
       "mean_symmetry", "mean_fractal_dimension", "radius_SE", "texture_SE",
       "perimeter_SE", "area_SE", "smoothness_SE", "compactness_SE", "concavity_SE",
       "concave_points_SE", "symmetry_SE", "fractal_dimension_SE", "worst_radius",
       "worst_texture", "worst_perimeter", "worst_area", "worst_smoothness",
       "worst_compactness", "worst_concavity", "worst_concave_points", "worst_symmetry",
       "worst_fractal_dimension")
colnames(data) = n
code <- class.ind(data$diag)
data1 <- as.data.frame(cbind(code, sapply(data[,2:31], 
                                          function(x){(x-min(x))/(max(x)-min(x))})))

n1 <- c("B",
        "M",
        "mean_radius",
        "mean_texture",
        "mean_perimeter",
        "mean_area",
        "mean_smoothness", 
        "mean_compactness", 
        "mean_concavity", 
        "mean_concave_points", 
        "mean_symmetry", 
        "mean_fractal_dimension",
        "radius_SE",
        "texture_SE",
        "perimeter_SE",
        "area_SE",
        "smoothness_SE",
        "compactness_SE",
        "concavity_SE",
        "concave_points_SE",
        "symmetry_SE",
        "fractal_dimension_SE",
        "worst_radius",
        "worst_texture",
        "worst_perimeter",
        "worst_area",
        "worst_smoothness",
        "worst_compactness",
        "worst_concavity",
        "worst_concave_points",
        "worst_symmetry",
        "worst_fractal_dimension")
head(data1)

colnames(data1) <- n1


#75% for training, and 25% for testing

#set.seed(123456)

set.seed(123454)
split <- sample.split(data[, 2], SplitRatio = 0.80)
data1_train <- subset(data1, split == TRUE)
data1_test <- subset(data1, split == FALSE)



##variable selection.

data2 <- data1_train

st1 <- Sys.time()
ga_ctrl <- gafsControl(functions = rfGA, # Assess fitness with RF
                       method = "cv",    # 10 fold cross validation
                       genParallel = TRUE, # Use parallel programming
                       allowParallel = TRUE)
lev <- c("M", "B")

set.seed(28)
model_2 <- gafs(x = data2[, -c(1, 2)], y = factor(data2[,2]),
            iters = 10, # generations of algorithm
                popSize = 5, # population size for each generation
                levels = lev,
                gafsControl = ga_ctrl)
et1 <- Sys.time()
tt1 <- et1 - st1
tt1
nam <- model_2$ga$final
nam

train_ga  <- data1_train[, c(1,2, which(colnames(data2) %in% nam))]
test_ga <- data1_test[, c(1,2, which(colnames(data2) %in% nam))]

model <- neuralnet(formula = f1, rep = 1, learningrate = 0.01, stepmax = 1e+05, data = train_ga, act.fct="logistic", hidden = c(1), linear.output=FALSE, err.fct = "ce", algorithm = "backprop")


nrow(test_ga)

f1 <- as.formula(paste("B + M ~", paste(nam, collapse = " + ")))
f2 <- as.formula(paste("M~", paste(nam, collapse = "+")))
f2


train2_ga <- train_ga
train2_ga$M <- as.factor(train2_ga$M)


test2_ga <- test_ga
test2_ga$M <- as.factor(test2_ga$M)

##

#Model Fitting

#Training our neural network 


f=as.formula(B + M~mean_radius+mean_texture+
               mean_perimeter+mean_area+mean_smoothness+mean_compactness+
               mean_concavity+mean_concave_points+mean_symmetry+
               mean_fractal_dimension+radius_SE+texture_SE+perimeter_SE+
               area_SE+smoothness_SE+compactness_SE+concavity_SE+
               concave_points_SE+symmetry_SE+fractal_dimension_SE+
               worst_radius+worst_texture+worst_perimeter+worst_area+
               worst_smoothness+worst_compactness+worst_concavity+
               worst_concave_points+worst_symmetry+worst_fractal_dimension)
f1
f3 <- as.formula(as.factor(M)~mean_radius+mean_texture+mean_perimeter+mean_area+mean_smoothness+mean_compactness+mean_concavity+mean_concave_points+mean_symmetry+mean_fractal_dimension+radius_SE+texture_SE+perimeter_SE+area_SE+smoothness_SE+compactness_SE+concavity_SE+concave_points_SE+symmetry_SE+fractal_dimension_SE+worst_radius+worst_texture+worst_perimeter+worst_area+worst_smoothness+worst_compactness+worst_concavity+worst_concave_points+worst_symmetry+worst_fractal_dimension)


set.seed(13)
st_rprop <- Sys.time()
model_rprop <- neuralnet(formula = f, data = data1_train, act.fct="logistic", 
                         hidden = 2, linear.output=FALSE, err.fct = "ce", 
                         algorithm = "rprop+")
et_rprop <- Sys.time()
tt_rprop <- et_rprop - st_rprop
tt_rprop


set.seed(13)
st_rpropw <- Sys.time()
model_rpropw <- neuralnet(formula = f , data = data1_train, act.fct="logistic", 
                          hidden = 2, linear.output=FALSE, err.fct = "ce", 
                          algorithm = "rprop-")
et_rpropw <- Sys.time()
tt_rpropw <- et_rpropw - st_rpropw
tt_rpropw

set.seed(12)
st_back <- Sys.time()
model_backprop <- neuralnet(formula = f, data = data1_train, rep = 1, learningrate = 0.01, 
                            stepmax = 1e+05,  act.fct="logistic", hidden = 2, linear.output=FALSE, 
                            err.fct = "ce", algorithm = "backprop")
et_back <- Sys.time()
tt_back <- et_back - st_back
tt_back



model_rprop.result <- compute(model_rprop, data1_test[,3:ncol(data1_test)])
model_rpropw.result <- compute(model_rpropw, data1_test[, 3 : ncol(data1_test)])
model_backprop.result <-  compute(model_backprop, data1_test[, 3 : ncol(data1_test)])

predicted_prob_rprop <- model_rprop.result$net.result
predicted_prob_rpropw <- model_rpropw.result$net.result
predicted_prob_backprop <- model_backprop.result$net.result

predicted_class_rprop <- max.col(predicted_prob_rprop)
predicted_class_rpropw <- max.col(predicted_prob_rpropw)
predicted_class_backprop <- max.col(predicted_prob_backprop)
observed_class <- max.col(data1_test[, 1 : 2])

conf_rprop <- confusionMatrix(predicted_class_rprop, observed_class)
conf_rpropw <- confusionMatrix(predicted_class_rpropw, observed_class)
conf_backprop <- confusionMatrix(predicted_class_backprop, observed_class)
conf_rprop

time1 <- rbind(tt_rprop, tt_rpropw, tt_back)
overall <- rbind(conf_rprop$overall, conf_rpropw$overall, conf_backprop$overall)
rownames(overall) <- c("rprop+", "rprop-", "backprop")
overall <- as.data.frame(cbind(overall, time1))
colnames(overall)[8] <- "time"
View(overall)

View(conf_backprop$table)
View(conf_rprop$table)
View(conf_rpropw$table)

View(cbind(conf_rprop$byClass, conf_rpropw$byClass, conf_backprop$byClass))


pred_rprop <- ROCR :: prediction(predictions = predicted_prob_rprop[,2], labels = test_ga[,2])
pred_rpropw <- ROCR :: prediction(predictions = predicted_prob_rpropw[,2], labels = test_ga[,2])


per_rprop <- performance(pred_rprop, measure = "tpr", x.measure = "fpr")
per_rpropw <- performance(pred_rpropw, measure = "tpr", x.measure = "fpr")



plot(per_rprop, main = "ROC curve", col = "blue", lwd = 1)
plot(per_rpropw, main = "ROC curve", col = "black", lwd = 1)



rec_op_char_rprop <- roc(data1_test[,2], predicted_prob_rprop[,2]) 
rec_op_char_rpropw <- roc(data1_test[,2], predicted_prob_rpropw[,2])
rec_op_char_backprop <- roc(data1_test[,2], predicted_prob_backprop[,2])

plot(1 - rec_op_char_rprop$specificities, 
     rec_op_char_rprop$sensitivities, type = "l", xlab = "1-specificity", 
     ylab = "sensitivity", main = "ROC Curve", lwd = 2, col = "black")
lines(1 - rec_op_char_rpropw$specificities, 
      rec_op_char_rpropw$sensitivities, lty = 2, lwd = 2, col = "blue")
lines(1 - rec_op_char_backprop$specificities, 
      rec_op_char_backprop$sensitivities, lty = 4, lwd = 2, col = "red")
abline(a = 0, b = 1, lwd = 2, lty = 2)
legend(0.7, 0.3, legend = c("rprop+", "rprop-", "Back-prop"), 
       lty = c(1, 2, 4), lwd = 2, col = c("black", "blue", "red"))

area <- rbind(rec_op_char_backprop$auc, 
              rec_op_char_rprop$auc,
              rec_op_char_rpropw$auc)
rownames(area) <- c("Back-propagation", "rprop+", "rprop-")
View(area)

#K-Fold Cross Validation

set.seed(2)
test_index <- createFolds(data1_train[,1], k = 10 )
test_index
test_index[[1]]

cross_val_2 = function(struct = 2, algo = "rprop+", lr = NULL, 
                       dataframe = data1_train,  
                       folds = 10, ind = test_index){
  
  #Initialise the error prop
  
  train_prop = NULL
  test_prop  = NULL

  for(i in 1 : folds) {
    data_train <- dataframe[-ind[[i]],]
    data_test <- dataframe[ind[[i]], ]
    
    #Training our neural network
    
    f=as.formula(B+M~mean_radius+mean_texture+mean_perimeter+mean_area+mean_smoothness+mean_compactness+
                   mean_concavity+mean_concave_points+mean_symmetry+mean_fractal_dimension+radius_SE+texture_SE+
                   perimeter_SE+area_SE+smoothness_SE+compactness_SE+concavity_SE+concave_points_SE+symmetry_SE+
                   fractal_dimension_SE+worst_radius+worst_texture+worst_perimeter+worst_area+worst_smoothness+
                   worst_compactness+worst_concavity+worst_concave_points+worst_symmetry+worst_fractal_dimension)
    
    set.seed(123)
    model <- neuralnet(formula = f, data = data_train, hidden=struct, 
                       learningrate = lr, algorithm = algo, act.fct="logistic", 
                       err.fct = "ce", linear.output=FALSE)
     #Computing The Predictions
    
    result_train <- compute(model, data_train[,3:ncol(data_train)])
    pred_train <- result_train$net.result
    result_test <- compute(model, data_test[,3:ncol(data_test)])
    pred_test <- result_test$net.result
    
    #Computation of efficiency from training data
    orig_diag_train <- max.col(data_train[,1:2])
    pred_diag_train <- max.col(pred_train)
    per_eff_train <- mean(orig_diag_train==pred_diag_train)*100
    train_prop[i] = per_eff_train
    
    #Computation of efficiecy from testing data
    orig_diag_test <- max.col(data_test[,1:2])
    pred_diag_test <- max.col(pred_test)
    per_eff_test <- mean(orig_diag_test==pred_diag_test)*100
    test_prop[i] = per_eff_test
    
  }
  l = list(train_prop, test_prop)
  return(sapply(l, mean))
}

x <- rep(c(2:10), 2)
y <- rep(c("rprop+", "rprop-"), each = 9)
cv_res <- map2( .x = x, .y = y, .f = cross_val_2, ind = test_index)
cv_res2 <- map(.x = c(2:10), .f = cross_val_2, algo = "backprop", 
               lr=0.01, ind = test_index)
cv_final <- list(cv_res, cv_res2)
v1<- as.data.frame(t(data.frame(cv_res)))
rownames(v1) <- NULL
v2 <- as.data.frame(t(data.frame(cv_res2)))
rownames(v2) <- NULL
cv <- rbind(v1, v2)
a <- rep(seq(2,10,1), 3)
b <- rep(c("rprop+", "rprop-", "backprop"), each = 9)
cv2 <- cbind(a, b, cv)
colnames(cv2) <- c("nodes", "Algorithm", "training error", "testing error")
cv3 <- gather(data = cv2, key = "error", value, -c(nodes, Algorithm))
cv4 <- spread(data = cv3, key = Algorithm, value = value)
cv4_train <- cv4[which(cv4$error=="training error"),]
cv4_test <- cv4[-which(cv4$error=="training error"),]

cv4_test[which(cv4_test[,3] == max(cv4_test[,3])),1]
cv4_test[which(cv4_test[,4] == max(cv4_test[,4])),1]
cv4_test[which(cv4_test[,5] == max(cv4_test[,5])),1]


matplot(cv4_test$nodes, cv4_test[,3 : 5], main = "Testing accuracy", 
        xlab = "Nodes", ylab = "Accuracy", type = "o", pch = 16, 
        col = c("red", "blue", "black"))
legend(2, 95.5, legend = c("backprop", "rprop-", "rprop+"),
       pch = 16, col = c("red", "blue", "black"), lty = c(1, 2, 3))

#Final Model

set.seed(13)
st_rprop2 <- Sys.time()
model_rprop2 <- neuralnet(formula = f, data = data1_train, act.fct="logistic", 
                          hidden = 8, linear.output=FALSE, err.fct = "ce", 
                          algorithm = "rprop+")
et_rprop2 <- Sys.time()
tt_rprop2 <- et_rprop2 - st_rprop2
tt_rprop2
plot(model_rprop2)

set.seed(13)
st_rpropw2 <- Sys.time()
model_rpropw2 <- neuralnet(formula = f , data = data1_train, act.fct="logistic", 
                           hidden = 3, linear.output=FALSE, err.fct = "ce", 
                           algorithm = "rprop-")
et_rpropw2 <- Sys.time()
tt_rpropw2 <- et_rpropw2 - st_rpropw2
tt_rpropw2
plot(model_rpropw2)
set.seed(12)
st_back2 <- Sys.time()
model_backprop2 <- neuralnet(formula = f, data = data1_train, rep = 1, 
                             learningrate = 0.01, stepmax = 1e+05,  
                             act.fct="logistic", hidden = 6, linear.output=FALSE, 
                             err.fct = "ce", algorithm = "backprop")
et_back2 <- Sys.time()
tt_back2 <- et_back2 - st_back2
tt_back2

model_rprop.result2 <- compute(model_rprop2, data1_test[,3:ncol(data1_test)])
model_rpropw.result2 <- compute(model_rpropw2, data1_test[, 3 : ncol(data1_test)])
model_backprop.result2 <-  compute(model_backprop2, data1_test[, 3 : ncol(data1_test)])

predicted_prob_rprop2 <- model_rprop.result2$net.result
predicted_prob_rpropw2 <- model_rpropw.result2$net.result
predicted_prob_backprop2 <- model_backprop.result2$net.result

predicted_class_rprop2 <- max.col(predicted_prob_rprop2)
predicted_class_rpropw2 <- max.col(predicted_prob_rpropw2)
predicted_class_backprop2 <- max.col(predicted_prob_backprop2)
observed_class2 <- max.col(data1_test[, 1 : 2])

conf_rprop2 <- confusionMatrix(predicted_class_rprop2, observed_class2)
conf_rpropw2 <- confusionMatrix(predicted_class_rpropw2, observed_class2)
conf_backprop2 <- confusionMatrix(predicted_class_backprop2, observed_class2)

time2 <- rbind(tt_rprop2, tt_rpropw2, tt_back2)
overall2 <- rbind(conf_rprop2$overall, conf_rpropw2$overall, conf_backprop2$overall)
rownames(overall2) <- c("rprop+", "rprop-", "backprop")
overall2 <- as.data.frame(cbind(overall2, time2))
colnames(overall2)[8] <- "time"
View(overall2)

View(conf_backprop2$table)
View(conf_rprop2$table)
View(conf_rpropw2$table)

View(cbind(conf_rprop2$byClass, conf_rpropw2$byClass, conf_backprop2$byClass))

conf_rprop2$table

rec_op_char_rprop2 <- roc(data1_test[,2], predicted_prob_rprop2[,2]) 
rec_op_char_rpropw2 <- roc(data1_test[,2], predicted_prob_rpropw2[,2])
rec_op_char_backprop2 <- roc(data1_test[,2], predicted_prob_backprop2[,2])

plot(1 - rec_op_char_rprop2$specificities, rec_op_char_rprop2$sensitivities, 
     type = "l", xlab = "1-specificity", ylab = "sensitivity", 
     main = "Revised ROC Curve", lwd = 2, col = "black")
lines(1 - rec_op_char_rpropw2$specificities, rec_op_char_rpropw2$sensitivities, 
      lty = 2, lwd = 2, col = "blue")
lines(1 - rec_op_char_backprop2$specificities, rec_op_char_backprop2$sensitivities, 
      lty = 4, lwd = 2 , col = "red")
abline(a = 0, b = 1, lwd = 2, lty = 2)
legend(0.7, 0.3, legend = c("rprop+", "rprop-", "backprop"), 
       lty=c(1, 2, 4), col =c("black", "blue", "red"))

area2 <- rbind(rec_op_char_backprop2$auc,
               rec_op_char_rprop2$auc,
               rec_op_char_rpropw2$auc)
rownames(area2) <- c("backprop", "rprop+", "rprop-")
View(area2)

par(mfrow = c(1,2))

pca_train <- prcomp(data1_train, retx = TRUE)
scores_train <- pca_train$x
train_new <- as.data.frame(cbind(data1_train$M, scores_train[,1:2]))
colnames(train_new)[1]="class"
plot(train_new$PC1, train_new$PC2, type = "p", pch = train_new$class+1, 
     col = train_new$class+1, 
     main = "Scatterplot of the first two Principal Components for the training set", 
     xlab = "PCA1", ylab = "PCA2")

pca_test <- prcomp(data1_test, retx = TRUE)
scores_test <- pca_test$x

test_new <- as.data.frame(cbind(data1_test$M, corrclass, scores_test[,1:2]))
colnames(train_new)[1]="class"
plot(scores_test[,1], scores_test[,2], type = "p", pch = data1_test$M+1, 
     col = data1_test$M+1, 
     main = "Scatterplot of the first two Principal Components for the testing set", 
     xlab = "PCA1", ylab = "PCA2")

corrclass <- ifelse(predicted_class_rprop != data1_test[,2]+1, "red", "black")
corrclass
plot(scores_test[,1], scores_test[,2], type = "p", pch = data1_test$M+1, col = corrclass, main = "Scatterplot showing the miss-classified testing samples by rprop+", xlab = "PCA1", ylab = "PCA2")

par(mfrow = c(1, 2))

plot(1 - rec_op_char_rprop$specificities, rec_op_char_rprop$sensitivities, 
     type = "l", xlab = "1-specificity", ylab = "sensitivity", main = "ROC Curve", 
     lwd = 2, col = "black")
lines(1 - rec_op_char_rpropw$specificities, rec_op_char_rpropw$sensitivities, 
      lty = 2, lwd = 2, col = "blue")
lines(1 - rec_op_char_backprop$specificities, rec_op_char_backprop$sensitivities, 
      lty = 4, lwd = 2, col = "red")
abline(a = 0, b = 1, lwd = 2, lty = 2)
legend(0.4, 0.3, legend = c("rprop+", "rprop-", "Back-prop"), lty = c(1, 2, 4), 
       lwd = 2, col = c("black", "blue", "red"))

plot(1 - rec_op_char_rprop2$specificities, rec_op_char_rprop2$sensitivities, 
     type = "l", xlab = "1-specificity", ylab = "sensitivity", 
     main = "Revised ROC Curve", lwd = 2, col = "black")
lines(1 - rec_op_char_rpropw2$specificities, rec_op_char_rpropw2$sensitivities, 
      lty = 2, lwd = 2, col = "blue")
lines(1 - rec_op_char_backprop2$specificities, rec_op_char_backprop2$sensitivities, 
      lty = 4, lwd = 2 , col = "red")
abline(a = 0, b = 1, lwd = 2, lty = 2)
legend(0.4, 0.3, legend = c("rprop+", "rprop-", "backprop"), lty=c(1, 2, 4), 
       col =c("black", "blue", "red"))
