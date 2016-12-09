#csc570 - data science essentials (Bernico) 
#Fall 2016 - Final Project
#matthew d murphy

#set environemet
#getwd()
#setwd("C:/Users/mdmurphy/Box Sync/csc570AL machine learning/data")

library(caret)

letters <- read.csv("letterdata.csv")
str(letters)


#delete observations with missing values/ none missing
letters<-na.omit(letters)

#set seed of random number generator
set.seed(123)

#shuffle df rowwise #necessary?
letters_s <- letters[sample(nrow(letters)),] #letters shuffled
#letters_s <- letters #or not?

#normalize the attribute values
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

#normalize
letters_norm <- as.data.frame(lapply(letters_s[2:17], normalize)) #letters normalized
head(letters_norm)


#z score scaling
#letters_z <- as.data.frame(scale(letters_s[2:17])) #letters z score scaled
#head(letters_z)

#z score scaled partitioning test/train split
#letters_z_all <- letters_z
#letters_all_labels <- letters_s[1:20000, 1]  
#letters_z_all$letters <- letters_all_labels
#head(letters_z_all)
#letters_z_train <- letters_z_all[1:16000, ]   #selects all columns of the # first 16000 rows
#letters_z_test  <- letters_z_all[16001:20000, ] #selects all columns of the # last 4000 rows
#head(letters_z_train)
#head(letters_z_test)

#split into train and tests sets
letters_train <- letters_norm[1:16000, ]   #selects all columns of the # first 16000 rows
letters_test  <- letters_norm[16001:20000, ] #selects all columns of the # last 4000 rows

#strip off labels
letters_train_labels <- letters_s[1:16000, 1]   
letters_test_labels  <- letters_s[16001:20000, 1] 

#add labels to the test and train sets
letters_train_all <- letters_train
letters_train_all$letters <- letters_train_labels
head(letters_train_all)

letters_test_all <- letters_test
letters_test_all$letters <- letters_test_labels
head(letters_test_all)

prop.table(table(letters_train_all$letter))
prop.table(table(letters_test_all$letter))

#conduct EDA
cor(letters_norm)
#[c("xbox", "ybox", "width", "height", "onpix","xbar","ybar","x2bar","y2bar","xybar","x2ybar","xy2bar","xedge","xedgey","yedge","yedgex")])

#pairs panel plots
#install.packages("psych") #one time install
library("psych")
pairs.panels(letters[c("xbox", "ybox", "width", "height", "onpix")])
pairs.panels(letters[c("xbar","ybar","x2bar","y2bar","xybar")])
pairs.panels(letters[c("x2ybar","xy2bar","xedge","xedgey","yedge","yedgex")])

#classification with RIPPER
library("RWeka")
model_JRip <- JRip(letters ~ ., data = letters_train_all)
model_JRip #number of rules ~373
summary(model_JRip) #accuracy 92.3% on train

#evaluate RIPPER model
model_JRip_pred <- predict(model_JRip,letters_test_all)
CrossTable(letters_test_all$letters, model_JRip_pred, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,dnn = c('actual', 'predicted'))
confusionMatrix(model_JRip_pred, letters_test_all$letters ) #accuracy 85.28% on test

library(randomForest)
#random forest classifier
rf_model <- randomForest(letters ~., data=letters_train_all)

#evaluate the random forest model
rf_model
importance(rf_model)
rf_model_pred <- predict(rf_model,letters_test,type="response")
confusionMatrix(rf_model_pred, letters_test_all$letters ) #accuracy 96.38%!!

#tuning the random forest model
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
grid_rf <- expand.grid(.mtry = c(2, 4, 6, 8, 16))
rf_model_tuned <- train(letters ~ ., data = letters_train_all, method = "rf",metric = "Kappa", trControl = ctrl,tuneGrid = grid_rf)

#evaluate rf model tuned
rf_model_tuned
rf_model_tuned_pred <- predict(rf_model_tuned,newdata = letters_test_all )
confusionMatrix(rf_model_tuned_pred, letters_test_all$letters )

#install.packages(class) #used for knn model
library(class)

#train k nearest neighbors model
knn_model_pred <- knn(train = letters_train, test = letters_test,cl = letters_train_labels, k = 1)

library(gmodels) # loads the gmodels package/CrossTable
#evaluate the knn model
CrossTable(x = letters_test_labels, y = knn_model_pred, prop.chisq=FALSE)

confusionMatrix(knn_model_pred, letters_test_all$letters )
mean(knn_model_pred == letters_test_all$letters ) #accuracy 95.075% k=5 /95.7% k=1

#tuning the knn model with caret
ctrl <- trainControl(method="repeatedcv",repeats = 100) #,classProbs=TRUE,summaryFunction = twoClassSummary)
knn_model_fitted <- train(letters ~ ., data = letters_train_all, method = "knn", trControl = ctrl, tuneLength = 15)
knn_model_fitted

#evaluate fitted knn model
knn_fitted_pred <- predict(knn_model_fitted,newdata = letters_test_all)
confusionMatrix(knn_fitted_pred, letters_test_all$letters )
mean(knn_fitted_pred == letters_test_all$letters ) #94.175% accuracy


#support vector machine SVM classifier
#install.packages("kernlab") #one time install
library(kernlab)

svm_model_linear <- ksvm(letters ~ ., data = letters_train_all, kernel = "vanilladot", C=1, cross=10) #linear kernel
svm_model_linear

svm_model_linear_pred <- predict(svm_model_linear, letters_test_all)
head(svm_model_linear_pred)

table(svm_model_linear_pred, letters_test_all$letter)
agreement <- svm_model_linear_pred == letters_test_all$letters

table(agreement)
prop.table(table(agreement)) #84.75% accuracy

#retrain SVM with kernel "rfbdot"
svm_model_rbf <- ksvm(letters ~ ., data = letters_train_all,kernel = "rbfdot", C=1, cross=10) #Gaussian radial basis kernel
svm_model_rbf_pred <- predict(svm_model_rbf,letters_test_all)

agreement_rbf <- svm_model_rbf_pred == letters_test_all$letter
table(agreement_rbf)
prop.table(table(agreement_rbf)) #96.55% accuracy/dropped to 93% with train = 16k

#details of SVM rbf model
attributes(svm_model_rbf)
alpha(svm_model_rbf)
alphaindex(svm_model_rbf)
b(svm_model_rbf)

#retrain SVM with kernel "polydot"
svm_model_poly <- ksvm(letters ~ ., data = letters_train_all,kernel = "polydot", C=1, cross=10) #polynomial kernel
svm_model_poly_pred <- predict(svm_model_poly,letters_test_all)

agreement_poly <- svm_model_poly_pred == letters_test_all$letter
table(agreement_poly)
prop.table(table(agreement_poly)) #84.9% accuracy/ 84.750

#svm models with bagging
library(caret)
str(svmBag)
svmBag$fit
svmBag$pred
svmBag$aggregate

#set bagging control structures
ctrl <- trainControl(method = "cv", number = 10)
bagctrl <- bagControl(fit = svmBag$fit, predict = svmBag$pred, aggregate = svmBag$aggregate)

#train bagged svm linear kernel model
svm_model_bag <- train(letters ~., data = letters_train_all, "svmLinear", C=1, trControl = ctrl, bagControl = bagctrl)
svm_model_bag  #~84.5% accuracy

#evaluate bagged svm linear kernel model
svm_model_bag_pred <- predict(svm_model_bag,letters_test_all)
agreement_svmbag <- svm_model_bag_pred == letters_test_all$letter
table(agreement_svmbag)
prop.table(table(agreement_svmbag))

#train bagged svm radial kernel model
svm_model_radial_bag <- train(letters ~., data = letters_train_all, "svmRadial", C=1, trControl = ctrl, bagControl = bagctrl)
svm_model_radial_bag #~93.5% accuracy / 92.95

#evaluate bagged svm radial kernel model
svm_model_radial_bag_pred <- predict(svm_model_radial_bag,letters_test_all)
agreement_svmRbag <- svm_model_radial_bag_pred == letters_test_all$letter
table(agreement_svmRbag)
prop.table(table(agreement_svmRbag))

#train bagged svm polynomial kernel model
svm_model_poly_bag <- train(letters ~., data = letters_train_all, "svmPoly", C=1, trControl = ctrl, bagControl = bagctrl)
svm_model_poly_bag #~96.6% accuracy

#evaluate bagged svm polynomial kernel model
svm_model_poly_bag_pred <- predict(svm_model_poly_bag,letters_test_all)
agreement_svmPbag <- svm_model_poly_bag_pred == letters_test_all$letter
table(agreement_svmPbag)
prop.table(table(agreement_svmPbag))

#multinomial logistic model
#install.packages("nnet")
library(nnet)

#train multinomial logistic model
model <- multinom(letters ~ ., data=letters_train_all, maxit=500, trace = F)
model
topModels <- varImp(model)
topModels

preds1 <- predict(model, type="probs", newdata = letters_test_all)
preds2 <- predict(model, type="class", newdata = letters_test_all)

head(preds1)
head(preds2)
head(letters_test_all$letters)

postResample(letters_test_all$letters,preds2) #accuracy 76.82%/77.3%

#library(nnet)
library(MASS)
#train single layer neural net
nnet_model <- train(letters ~., data = letters_train_all, method = "nnet", tuneLength = 2, trace=F, maxit=100)
nnet_model

#evaluate neural net model
nnet_model_pred <- predict(nnet_model,newdata=letters_test_all,type="raw")
agreement_nnet <- nnet_model_pred == letters_test_all$letter
table(agreement_nnet)
prop.table(table(agreement_nnet)) #accuracy 36.6%  

#retry with tuneLength = 4 gave accuracy of 59.6%  //update w/ tuneLength = 10... still running

###############regression tree model
#enhancing the model with regression trees
library(rpart.plot)
regtree_model <- rpart(letters ~ ., data=letters_train_all)
regtree_model
summary(regtree_model)

regtree_model_pred <- predict(regtree_model, letters_test_all, type ="class")
regtree_model_pred
summary(regtree_model_pred)

#evaluating model performance
table(letters_test_all$letters)

rtagreement <- regtree_model_pred == letters_test_all$letters
table(rtagreement)
prop.table(table(rtagreement))

misClasificError <- mean(regtree_model_pred != letters_test_all$letters)
print(paste('Accuracy',1-misClasificError)) #accuracy 86.924%

#plot regression trees visually
rpart.plot(regtree_model, digits = 3)
#alternate plot
rpart.plot(regtree_model, digits = 4, fallen.leaves = TRUE,type = 3, extra = 101)

#fine tuning the regression tree model with caret
cctrlR <- trainControl(method = "cv", number = 10, returnResamp = "all", search = "random")
regtree_model_tuned <- train(letters_train_all, letters_train_all$letters, 
                             method = "rpart", 
                             trControl = cctrlR,
                             tuneLength = 4)
regtree_model_tuned
regtree_model_tuned_pred <- predict(regtree_model_tuned, letters_test_all)

#evaluate the tuned regression tree model
summary(regtree_model_tuned_pred)
summary(letters_test_all$letters)

trtagreement <- regtree_model_tuned_pred == letters_test_all$letters
table(trtagreement)
prop.table(table(trtagreement))

misClasificError <- mean(regtree_model_tuned_pred != letters_test_all$letters)
print(paste('Accuracy',1-misClasificError)) #Accuracy 80.15%
##############

#save session
savehistory("script.txt")