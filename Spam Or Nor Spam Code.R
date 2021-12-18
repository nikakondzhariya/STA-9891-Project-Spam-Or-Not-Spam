rm(list = ls())    #delete objects
cat("\014")        #clear console

### REQUIRED LIBRARIES ###
library(readr)
library(tidyverse)
library(modelr)
library(glmnet)
library(randomForest)
library(reshape)
library(ggthemes)
library(gridExtra)
library(caTools)

set.seed(1)

### OPERATIONS WITH DATA ###

SpamBase = read_csv("/Users/nikakondzhariya/Desktop/Sta\ 9891\ Project/Data/spambase.csv", 
                    col_names = TRUE)
SpamBase=as.data.frame(SpamBase)
str(SpamBase)
dim(SpamBase)
summary(SpamBase)
table(SpamBase$class)

n_spam = sum(SpamBase$class == 1) #number of spam class outcomes 
n_spam
n_non_spam=sum(SpamBase$class == 0) #number of non_spam class outcomes
n_non_spam

hist(SpamBase$class)

# Create Y vector
Y=SpamBase$class
Y

# Create X matrix
#X = model.matrix(Res.Var.ND~.,TomsHardware)[,-1]
X=data.matrix(SpamBase%>%select(-class))
dim(X)

### TRAIN AND TEST: LASSO, ELASTIC-NET, RIDGE, RANDOM FORREST GENERATION WITH REQUIRED CALCULATIONS ###

# 1 Obtain n (number of observations) and p (number of predictors)
n=nrow(X)
p=ncol(X)

# 2 Define Modeling Parameters including train (n.train) and test (n.test) number of observations
#d.rate=0.9 # division rate for train and test data
d.repit=50 # repeat the ongoing tasks 50 times 
n.train=round(0.9*n)
n.test=n-n.train

# 3 Prepare the following zero-filled matrices that we are going to fill as values are obtained 

# Matrices for train and test R-squared 
Auc.train.table=matrix(0,d.repit,4)
colnames(Auc.train.table)=c("Lasso","Elastic-Net","Ridge","Random Forest")
Auc.test.table=matrix(0,d.repit,4)
colnames(Auc.test.table)=c("Lasso","Elastic-Net","Ridge","Random Forest")

# Matrix for the time it takes to cross-validate Lasso/Elastic-Net/Ridge regression
Time.cv=matrix(0,d.repit,4)
colnames(Time.cv)=c("Lasso","Elastic-Net","Ridge", "Random Forest")

# 4 Fit Lasso, Elastic-Net, Ridge and Random forest (50 times repition)

for (j in c(1:d.repit)) {
  cat("d.repit = ", j, "\n")
  
  shuffled_indexes =  sample(n)
  train            =  shuffled_indexes[1:n.train]
  test             =  shuffled_indexes[(1+n.train):n]
  X.train          =  X[train, ]
  y.train          =  Y[train]
  X.test           =  X[test, ]
  y.test           =  Y[test]
  
  # Fit Lasso, calculate time, AUC (both train and test), and estimated coefficients
  
  # Lasso
  
  time.start       =  Sys.time()
  Lasso = cv.glmnet(x = X.train, y=y.train, family = "binomial", alpha = 1, intercept = TRUE, standardize = FALSE, nfolds = 10, type.measure="auc")
  time.end         =  Sys.time()
  Time.cv [j,1] = time.end - time.start
  lasso_min = glmnet(x = X.train, y=y.train, lambda = Lasso$lambda.min, family = "binomial", alpha = 1, intercept = TRUE, standardize = FALSE)
  
  beta0.hat.lasso = lasso_min$a0
  beta.hat.lasso = as.vector(lasso_min$beta)
  
  seq_lasso=c(seq(0,1, by = 0.01))
  seq_len_lasso=length(seq_lasso)
  FPR_train_lasso=rep(0, seq_len_lasso)
  TPR_train_lasso=rep(0, seq_len_lasso)
  FPR_test_lasso=rep(0, seq_len_lasso)
  TPR_test_lasso=rep(0, seq_len_lasso)
  
  prob.train.lasso              =        exp(X.train %*% beta.hat.lasso +  beta0.hat.lasso  )/(1 + exp(X.train %*% beta.hat.lasso +  beta0.hat.lasso  ))
  prob.test.lasso              =        exp(X.test %*% beta.hat.lasso +  beta0.hat.lasso  )/(1 + exp(X.test %*% beta.hat.lasso +  beta0.hat.lasso  ))
  
  for (i in 1:seq_len_lasso){
    
    # training
    y.hat.train.lasso             =        ifelse(prob.train.lasso > seq_lasso[i], 1, 0) #table(y.hat.train, y.train)
    FP.train.lasso                =        sum(y.train[y.hat.train.lasso==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.train.lasso               =        sum(y.hat.train.lasso[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
    P.train.lasso                 =        sum(y.train==1) # total positives in the data
    N.train.lasso                 =        sum(y.train==0) # total negatives in the data
    FPR.train.lasso              =        FP.train.lasso/N.train.lasso # false positive rate = type 1 error = 1 - specificity
    TPR.train.lasso             =        TP.train.lasso/P.train.lasso # true positive rate = 1 - type 2 error = sensitivity
    typeI.err.train.lasso         =        FPR.train.lasso
    typeII.err.train.lasso        =        1 - TPR.train.lasso
    FPR_train_lasso[i]            =        typeI.err.train.lasso
    TPR_train_lasso[i]            =        1 - typeII.err.train.lasso
    
    
    # test
    y.hat.test.lasso              =        ifelse(prob.test.lasso > seq_lasso[i],1,0) #table(y.hat.test, y.test)  
    FP.test.lasso               =        sum(y.test[y.hat.test.lasso==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.test.lasso                 =        sum(y.hat.test.lasso[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
    P.test.lasso                  =        sum(y.test==1) # total positives in the data
    N.test.lasso                  =        sum(y.test==0) # total negatives in the data
    TN.test.lasso                 =        sum(y.hat.test.lasso[y.test==0] == 0)# negatives in the data that were predicted as negatives
    FPR.test.lasso                =        FP.test.lasso/N.test.lasso # false positive rate = type 1 error = 1 - specificity
    TPR.test.lasso                =        TP.test.lasso/P.test.lasso # true positive rate = 1 - type 2 error = sensitivity = recall
    typeI.err.test.lasso          =        FPR.test.lasso
    typeII.err.test.lasso        =        1 - TPR.test.lasso
    FPR_test_lasso[i]             =        typeI.err.test.lasso
    TPR_test_lasso[i]             =        1 - typeII.err.test.lasso
  }
  # train
  df.train.lasso = as.data.frame(FPR_train_lasso)
  df.train.lasso=df.train.lasso %>% mutate(TPR_train_lasso)%>%mutate(Set = "Train")%>% mutate(seq_lasso)
  colnames(df.train.lasso)=c("FPR","TPR", "Set", "Threshold")
  
  # test
  df.test.lasso = as.data.frame(FPR_test_lasso)
  df.test.lasso=df.test.lasso %>% mutate(TPR_test_lasso)%>%mutate(Set = "Test")%>% mutate(seq_lasso)
  colnames(df.test.lasso)=c("FPR","TPR", "Set", "Threshold")
  
  # test + train
  
  df_train_test.lasso=rbind(df.train.lasso,df.test.lasso)
  
  #Auc.train[i,1]=colAUC(prob.train.lasso, y.train, plotROC=FALSE)[1]
  Auc.train.table[j,1]=colAUC(prob.train.lasso, y.train, plotROC=FALSE)[1]
  Auc.train.table[j,1]
  #Auc.test[i,1]=colAUC(prob.test.lasso, y.test, plotROC=FALSE)[1]
  Auc.test.table[j,1]=colAUC(prob.test.lasso, y.test, plotROC=FALSE)[1]
  Auc.test.table[j,1] 
  
  # Elastic-net
  
  time.start       =  Sys.time()
  Elastic_net = cv.glmnet(x = X.train, y=y.train, family = "binomial", alpha = 0.5, intercept = TRUE, standardize = FALSE, nfolds = 10, type.measure="auc")
  time.end         =  Sys.time()
  Time.cv [j,2] = time.end - time.start
  elnet_min = glmnet(x = X.train, y=y.train, lambda = Elastic_net$lambda.min, family = "binomial", alpha = 0.5, intercept = TRUE, standardize = FALSE)
  
  beta0.hat.elnet = elnet_min$a0
  beta.hat.elnet = as.vector(elnet_min$beta)
  
  seq_elnet=c(seq(0,1, by = 0.01))
  seq_len_elnet=length(seq_elnet)
  FPR_train_elnet=rep(0, seq_len_elnet)
  TPR_train_elnet=rep(0, seq_len_elnet)
  FPR_test_elnet=rep(0, seq_len_elnet)
  TPR_test_elnet=rep(0, seq_len_elnet)
  
  prob.train.elnet              =        exp(X.train %*% beta.hat.elnet +  beta0.hat.elnet  )/(1 + exp(X.train %*% beta.hat.elnet +  beta0.hat.elnet  ))
  prob.test.elnet              =        exp(X.test %*% beta.hat.elnet +  beta0.hat.elnet  )/(1 + exp(X.test %*% beta.hat.elnet +  beta0.hat.elnet  ))
  
  for (i in 1:seq_len_elnet){
    
    # training
    y.hat.train.elnet             =        ifelse(prob.train.elnet > seq_elnet[i], 1, 0) #table(y.hat.train, y.train)
    FP.train.elnet                =        sum(y.train[y.hat.train.elnet==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.train.elnet               =        sum(y.hat.train.elnet[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
    P.train.elnet                 =        sum(y.train==1) # total positives in the data
    N.train.elnet                 =        sum(y.train==0) # total negatives in the data
    FPR.train.elnet              =        FP.train.elnet/N.train.elnet # false positive rate = type 1 error = 1 - specificity
    TPR.train.elnet             =        TP.train.elnet/P.train.elnet # true positive rate = 1 - type 2 error = sensitivity
    typeI.err.train.elnet         =        FPR.train.elnet
    typeII.err.train.elnet        =        1 - TPR.train.elnet
    FPR_train_elnet[i]            =        typeI.err.train.elnet
    TPR_train_elnet[i]            =        1 - typeII.err.train.elnet
    
    
    # test
    y.hat.test.elnet              =        ifelse(prob.test.elnet > seq_elnet[i],1,0) #table(y.hat.test, y.test)  
    FP.test.elnet               =        sum(y.test[y.hat.test.elnet==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.test.elnet                 =        sum(y.hat.test.elnet[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
    P.test.elnet                  =        sum(y.test==1) # total positives in the data
    N.test.elnet                  =        sum(y.test==0) # total negatives in the data
    TN.test.elnet                 =        sum(y.hat.test.elnet[y.test==0] == 0)# negatives in the data that were predicted as negatives
    FPR.test.elnet                =        FP.test.elnet/N.test.elnet # false positive rate = type 1 error = 1 - specificity
    TPR.test.elnet                =        TP.test.elnet/P.test.elnet # true positive rate = 1 - type 2 error = sensitivity = recall
    typeI.err.test.elnet          =        FPR.test.elnet
    typeII.err.test.elnet        =        1 - TPR.test.elnet
    FPR_test_elnet[i]             =        typeI.err.test.elnet
    TPR_test_elnet[i]             =        1 - typeII.err.test.elnet
  }
  # train
  df.train.elnet = as.data.frame(FPR_train_elnet)
  df.train.elnet=df.train.elnet %>% mutate(TPR_train_elnet)%>%mutate(Set = "Train")%>% mutate(seq_elnet)
  colnames(df.train.elnet)=c("FPR","TPR", "Set", "Threshold")
  
  # test
  df.test.elnet = as.data.frame(FPR_test_elnet)
  df.test.elnet=df.test.elnet %>% mutate(TPR_test_elnet)%>%mutate(Set = "Test")%>% mutate(seq_elnet)
  colnames(df.test.elnet)=c("FPR","TPR", "Set", "Threshold")
  
  # test + train
  
  df_train_test.elnet=rbind(df.train.elnet,df.test.elnet)
  
  #Auc.train[i,1]=colAUC(prob.train.elnet, y.train, plotROC=FALSE)[1]
  Auc.train.table[j,2]=colAUC(prob.train.elnet, y.train, plotROC=FALSE)[1]
  #Auc.test[i,1]=colAUC(prob.test.elnet, y.test, plotROC=FALSE)[1]
  Auc.test.table[j,2]=colAUC(prob.test.elnet, y.test, plotROC=FALSE)[1]
  
  # Ridge
  lam_ri_ri_ri=exp(seq(-4, 6, length.out = 100))
  time.start       =  Sys.time()
  Ridge = cv.glmnet(x = X.train, y=y.train, family = "binomial", alpha = 0, intercept = TRUE, standardize = FALSE, nfolds = 10, type.measure="auc", lambda=lam_ri_ri_ri)
  time.end         =  Sys.time()
  Time.cv [j,3] = time.end - time.start
  ridge_min = glmnet(x = X.train, y=y.train, lambda = Ridge$lambda.min, family = "binomial", alpha = 0, intercept = TRUE, standardize = FALSE)
  
  beta0.hat.ridge = ridge_min$a0
  beta.hat.ridge = as.vector(ridge_min$beta)
  
  seq_ridge=c(seq(0,1, by = 0.01))
  seq_len_ridge=length(seq_ridge)
  FPR_train_ridge=rep(0, seq_len_ridge)
  TPR_train_ridge=rep(0, seq_len_ridge)
  FPR_test_ridge=rep(0, seq_len_ridge)
  TPR_test_ridge=rep(0, seq_len_ridge)
  
  prob.train.ridge              =        exp(X.train %*% beta.hat.ridge +  beta0.hat.ridge  )/(1 + exp(X.train %*% beta.hat.ridge +  beta0.hat.ridge  ))
  prob.test.ridge              =        exp(X.test %*% beta.hat.ridge +  beta0.hat.ridge  )/(1 + exp(X.test %*% beta.hat.ridge +  beta0.hat.ridge  ))
  
  for (i in 1:seq_len_ridge){
    
    # training
    y.hat.train.ridge             =        ifelse(prob.train.ridge > seq_ridge[i], 1, 0) #table(y.hat.train, y.train)
    FP.train.ridge                =        sum(y.train[y.hat.train.ridge==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.train.ridge               =        sum(y.hat.train.ridge[y.train==1] == 1) # true positives = positives in the data that were predicted as positive
    P.train.ridge                 =        sum(y.train==1) # total positives in the data
    N.train.ridge                 =        sum(y.train==0) # total negatives in the data
    FPR.train.ridge              =        FP.train.ridge/N.train.ridge # false positive rate = type 1 error = 1 - specificity
    TPR.train.ridge             =        TP.train.ridge/P.train.ridge # true positive rate = 1 - type 2 error = sensitivity
    typeI.err.train.ridge         =        FPR.train.ridge
    typeII.err.train.ridge        =        1 - TPR.train.ridge
    FPR_train_ridge[i]            =        typeI.err.train.ridge
    TPR_train_ridge[i]            =        1 - typeII.err.train.ridge
    
    
    # test
    y.hat.test.ridge              =        ifelse(prob.test.ridge > seq_ridge[i],1,0) #table(y.hat.test, y.test)  
    FP.test.ridge               =        sum(y.test[y.hat.test.ridge==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.test.ridge                 =        sum(y.hat.test.ridge[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
    P.test.ridge                  =        sum(y.test==1) # total positives in the data
    N.test.ridge                  =        sum(y.test==0) # total negatives in the data
    TN.test.ridge                 =        sum(y.hat.test.ridge[y.test==0] == 0)# negatives in the data that were predicted as negatives
    FPR.test.ridge                =        FP.test.ridge/N.test.ridge # false positive rate = type 1 error = 1 - specificity
    TPR.test.ridge                =        TP.test.ridge/P.test.ridge # true positive rate = 1 - type 2 error = sensitivity = recall
    typeI.err.test.ridge          =        FPR.test.ridge
    typeII.err.test.ridge        =        1 - TPR.test.ridge
    FPR_test_ridge[i]             =        typeI.err.test.ridge
    TPR_test_ridge[i]             =        1 - typeII.err.test.ridge
  }
  # train
  df.train.ridge = as.data.frame(FPR_train_ridge)
  df.train.ridge=df.train.ridge %>% mutate(TPR_train_ridge)%>%mutate(Set = "Train")%>% mutate(seq_ridge)
  colnames(df.train.ridge)=c("FPR","TPR", "Set", "Threshold")
  
  # test
  df.test.ridge = as.data.frame(FPR_test_ridge)
  df.test.ridge=df.test.ridge %>% mutate(TPR_test_ridge)%>%mutate(Set = "Test")%>% mutate(seq_ridge)
  colnames(df.test.ridge)=c("FPR","TPR", "Set", "Threshold")
  
  # test + train
  
  df_train_test.ridge=rbind(df.train.ridge,df.test.ridge)
  
  #Auc.train[i,1]=colAUC(prob.train.ridge, y.train, plotROC=FALSE)[1]
  Auc.train.table[j,3]=colAUC(prob.train.ridge, y.train, plotROC=FALSE)[1]
  #Auc.test[i,1]=colAUC(prob.test.ridge, y.test, plotROC=FALSE)[1]
  Auc.test.table[j,3]=colAUC(prob.test.ridge, y.test, plotROC=FALSE)[1]
  
  # Random Forest 
  
  y.train.rf=as.factor(y.train)
  y.test.rf=as.factor(y.test)
  
  time.start       =  Sys.time()
  rf.fit     =    randomForest(X.train, y.train.rf, mtry = p/3, importance=TRUE)
  time.end         =  Sys.time()
  Time.cv [j,4] = time.end - time.start
  
  seq_rf=c(seq(0,1, by = 0.01))
  seq_len_rf=length(seq_rf)
  FPR_train_rf=rep(0, seq_len_rf)
  TPR_train_rf=rep(0, seq_len_rf)
  FPR_test_rf=rep(0, seq_len_rf)
  TPR_test_rf=rep(0, seq_len_rf)
  
  for (i in 1:seq_len_rf){
    
    #training
    p.hat.train.rf      =    predict(rf.fit, X.train, type = "prob")
    p.hat.train.rf      =    p.hat.train.rf[,2]
    y.hat.train.rf      =     rep("0",n.train)
    y.hat.train.rf[p.hat.train.rf>seq_rf[i]]  =     "1"
    FP.train.rf         =    sum(y.train.rf[y.hat.train.rf==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.train.rf         =    sum(y.hat.train.rf[y.train.rf==1] == 1) # true positives = positives in the data that were predicted as positive
    P.train.rf          =    sum(y.train.rf==1) # total positives in the data
    N.train.rf          =    sum(y.train.rf==0) # total negatives in the data
    FPR.train.rf        =    FP.train.rf/N.train.rf
    TPR.train.rf        =    TP.train.rf/P.train.rf
    typeI.err.train.rf         =        FPR.train.rf
    typeII.err.train.rf        =        1 - TPR.train.rf
    FPR_train_rf[i]            =        typeI.err.train.rf
    TPR_train_rf[i]            =        1 - typeII.err.train.rf
    
    #training
    p.hat.test.rf      =    predict(rf.fit, X.test, type = "prob")
    p.hat.test.rf      =    p.hat.test.rf[,2]
    y.hat.test.rf      =     rep("0",n.test)
    y.hat.test.rf[p.hat.test.rf>seq_rf[i]]  =     "1"
    FP.test.rf         =    sum(y.test.rf[y.hat.test.rf==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.test.rf         =    sum(y.hat.test.rf[y.test.rf==1] == 1) # true positives = positives in the data that were predicted as positive
    P.test.rf          =    sum(y.test.rf==1) # total positives in the data
    N.test.rf          =    sum(y.test.rf==0) # total negatives in the data
    FPR.test.rf        =    FP.test.rf/N.test.rf
    TPR.test.rf        =    TP.test.rf/P.test.rf
    typeI.err.test.rf         =        FPR.test.rf
    typeII.err.test.rf        =        1 - TPR.test.rf
    FPR_test_rf[i]            =        typeI.err.test.rf
    TPR_test_rf[i]            =        1 - typeII.err.test.rf
    
    
  }
  
  Auc.train.table[j,4]=colAUC(p.hat.train.rf, y.train.rf, plotROC=FALSE)[1]
  Auc.test.table[j,4]=colAUC(p.hat.test.rf, y.test.rf, plotROC=FALSE)[1]
  
}

# Mean AUCs and Times
Lasso_trainAuc_mean  =  mean(Auc.train.table[,1])
Lasso_testAuc_mean   =  mean(Auc.test.table[,1])
Elnet_trainAuc_mean  =  mean(Auc.train.table[,2])
Elnet_testAuc_mean   =  mean(Auc.test.table[,2])
Ridge_trainAuc_mean  =  mean(Auc.train.table[,3])
Ridge_testAuc_mean   =  mean(Auc.test.table[,3])
RF_trainAuc_mean     =  mean(Auc.train.table[,4])
RF_testAuc_mean      =  mean(Auc.test.table[,4])

# NEED THIS TO ANSWER 3c -- Record the time it takes to cross-validate ridge/lasso/elastic-net logistic regression.
# We will out average time that it takes to cross-validate each regression
Lasso_time_mean      =  mean(Time.cv[,1])
Elnet_time_mean      =  mean(Time.cv[,2])
Ridge_time_mean      =  mean(Time.cv[,3])
RF_time_mean         =  mean(Time.cv[,4])

average_time_to_tune=rbind(Lasso_time_mean,Elnet_time_mean,Ridge_time_mean)


# NEED THIS TO ASNWER 3b -- Side-by-Side Boxplots of AUC 
par(mfrow=c(1,2))
a=boxplot(Auc.train.table[,1], Auc.train.table[,2],Auc.train.table[,3],Auc.train.table[,4],
          main = "Auc for Train Data",
          names = c("Lasso", "Elastic-Net", "Ridge", "Random Forest"),
          col=c("blue","yellow", "green","red"),
          ylim=c(0.92,1))


b=boxplot(Auc.test.table[,1], Auc.test.table[,2],Auc.test.table[,3],Auc.test.table[,4],
          main = "Auc for Test Data",
          names = c("Lasso", "Elastic-Net", "Ridge", "Random Forest"),
          col=c("blue","yellow", "green","red"),
          ylim=c(0.92,1))


# NEED THIS TO ANSWER 3c -- For one on the 50 samples, create 10-fold CV curves for Lasso, Elastic-Net,Ridge
lam=exp(seq(-10, -2, length.out = 100))
lam_ri=exp(seq(-11, 6, length.out = 100))

cv.la.1 = cv.glmnet(x = X.train, y=y.train, family = "binomial", alpha = 1, intercept = TRUE, standardize = FALSE, nfolds = 10, type.measure="auc", lambda=lam)
cv.el.1 = cv.glmnet(x = X.train, y=y.train, family = "binomial", alpha = 0.5, intercept = TRUE, standardize = FALSE, nfolds = 10, type.measure="auc", lambda=lam)
cv.ri.1 = cv.glmnet(x = X.train, y=y.train, family = "binomial", alpha = 0, intercept = TRUE, standardize = FALSE, nfolds = 10, type.measure="auc", lambda=lam_ri)


par(mfrow=c(1,3))
plot(cv.la.1, ylim=c(0.6,1))
title('Lasso', line = 2.5)
plot(cv.el.1, ylim=c(0.6,1))
title('Elastic-Net', line = 2.5)
plot(cv.ri.1, ylim=c(0.6,1))
title('Ridge', line = 2.5)

# NEED THIS TO ANSWER 3c -- For one on the 50 samples, create 10-fold CV curves for Lasso, Elastic-Net,Ridge
#cv.la.1 = cv.glmnet(x = X.train, y=y.train, family = "binomial", alpha = 1, intercept = TRUE, standardize = FALSE, nfolds = 10, type.measure="auc")
#cv.el.1 = cv.glmnet(x = X.train, y=y.train, family = "binomial", alpha = 0.5, intercept = TRUE, standardize = FALSE, nfolds = 10, type.measure="auc")
#cv.ri.1 = cv.glmnet(x = X.train, y=y.train, family = "binomial", alpha = 0, intercept = TRUE, standardize = FALSE, nfolds = 10, type.measure="auc")

#?cv.glmnet
#par(mfrow=c(1,3))
#plot(cv.la.1, ylim=c(0.6,1))
#title('Lasso', line = 2.5)
#plot(cv.el.1, ylim=c(0.6,1))
#title('Elastic-Net', line = 2.5)
#plot(cv.ri.1, ylim=c(0.6,1))
#title('Ridge', line = 2.5)
#cv.la.1=cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
#cv.el.1=cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10)
#cv.ri.1=cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)

#par(mfrow=c(1,3))
#plot(cv.la.1)
#title('Lasso', line = 2.5)
#plot(cv.el.1)
#title('Elastic-Net', line = 2.5)
#plot(cv.ri.1)
#title('Ridge', line = 2.5)


### FULL DATASET: LASSO, ELASTIC-NET, RIDGE, RANDOM FORREST GENERATION WITH REQUIRED CALCULATIONS ###

#Lasso 
lambda_la=exp(seq(-5, -2, length.out = 100))
time.start.la=Sys.time()
cv.fit.la = cv.glmnet(X, Y, family = "binomial", alpha = 1, intercept = TRUE, standardize = FALSE, nfolds = 10, type.measure="auc", lambda=lambda_la)
fit.la = glmnet(X, Y, lambda = cv.fit.la$lambda.min, family = "binomial", alpha = 1, intercept = TRUE, standardize = FALSE)
time.end.la=Sys.time()
Time.la=time.end.la - time.start.la

# Coefficients Lasso
betaS.la=data.frame(c(1:p), as.vector(fit.la$beta))
colnames(betaS.la)     =     c( "feature", "value")

# Elastic-Net
time.start.el=Sys.time()
cv.fit.el = cv.glmnet(X, Y, family = "binomial", alpha = 0.5, intercept = TRUE, standardize = FALSE, nfolds = 10, type.measure="auc", lambda=lambda_la)
fit.el = glmnet(X, Y, lambda = cv.fit.el$lambda.min, family = "binomial", alpha = 0.5, intercept = TRUE, standardize = FALSE)
#Y.hat.el=predict(fit.el, newx = X, type = "response") # ASK QUESTION ABOUT IT
time.end.el=Sys.time()
Time.el=time.end.el - time.start.el

# Coefficients Elastic-Net
betaS.el=data.frame(c(1:p), as.vector(fit.el$beta))
colnames(betaS.el)     =     c( "feature", "value")

lam_ri_ri=exp(seq(-4, 6, length.out = 100))
# Ridge
time.start.ri=Sys.time()
cv.fit.ri = cv.glmnet(X, Y, family = "binomial", alpha = 0, intercept = TRUE, standardize = FALSE, nfolds = 10, type.measure="auc", lambda=lam_ri_ri)
fit.ri = glmnet(X, Y, lambda = cv.fit.ri$lambda.min, family = "binomial", alpha = 0, intercept = TRUE, standardize = FALSE)
#Y.hat.ri=predict(fit.ri, newx = X, type = "response") # ASK QUESTION ABOUT IT
time.end.ri=Sys.time()
Time.ri=time.end.ri - time.start.ri


# Coefficients Ridge
betaS.ri=data.frame(c(1:p), as.vector(fit.ri$beta))
colnames(betaS.ri)     =     c( "feature", "value")

# Random Forrest
Y.rf=as.factor(Y)
time.start.rf=Sys.time()
rf.wh = randomForest(X, Y.rf, mtry = p/3, importance=TRUE)
time.end.rf=Sys.time()
Time.rf=time.end.rf - time.start.rf

# Coefficients Random Forrest
betaS.rf=data.frame(c(1:p), as.vector(rf.wh$importance[1:p]))
colnames(betaS.rf)     =     c( "feature", "value")

# NEED THIS TO ANSWER 5D - Result for time 
Time.for.4m=rbind(Time.la,Time.el,Time.ri,Time.rf)
colnames(Time.for.4m)=c("Time Elapsed")

# NEED THIS TO ANSWER 5D - Median of test AUCs + 90% Confidence interval for test AUCs
# MEDIAN
median_testAUC_la = median(Auc.test.table[,1])
median_testAUC_el = median(Auc.test.table[,2])
median_testAUC_ri = median(Auc.test.table[,3])
median_testAUC_rf = median(Auc.test.table[,4])

#Result for MEDIAN
median_testAUC    = rbind(median_testAUC_la, median_testAUC_el, median_testAUC_ri, median_testAUC_rf)

# 90% CI
interval.la=quantile(Auc.test.table[,1], c(0.05,0.95))
interval.en=quantile(Auc.test.table[,2], c(0.05,0.95))
interval.ri=quantile(Auc.test.table[,3], c(0.05,0.95))
interval.rf=quantile(Auc.test.table[,4], c(0.05,0.95))

#Result for 90% CI
Inteval_90_testAUC=rbind(interval.la,interval.en,interval.ri,interval.rf)

### BAR-PLOTS OF THE ESTIMATED COEFFICIENTS (LASS, ELASTIC-NET, RIDGE), THE IMPORTANCE OF RF PARAMETERS ###

# we need to change the order of factor levels by specifying the order explicitly.
betaS.el$feature=factor(betaS.el$feature, levels = betaS.el$feature[order(betaS.el$value, decreasing = TRUE)])
betaS.rf$feature=factor(betaS.rf$feature,levels = betaS.el$feature[order(betaS.el$value, decreasing = TRUE)])
betaS.la$feature=factor(betaS.la$feature, levels = betaS.el$feature[order(betaS.el$value, decreasing = TRUE)])
betaS.ri$feature=factor(betaS.ri$feature, levels = betaS.el$feature[order(betaS.el$value, decreasing = TRUE)])

# Let's use elastic-net estimated coefficients to create an order based on largest to smallest coefficients, and 
# use this order to present bar-plots of the estimated coefficients of all 4 models

# Lasso Plot
laPlot =  ggplot(betaS.la, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")+
  labs(x = element_blank(), y = "Coefficients", title = expression(Lasso))+
  ylim(-1.5,1.5)

# Elastic-Net Plot
elPlot =  ggplot(betaS.el, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")+
  labs(x = element_blank(), y = "Coefficients", title = expression(Elastic-Net))+
  ylim(-1.5,1.5)

# Ridge 
riPlot =  ggplot(betaS.ri, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")+
  labs(x = element_blank(), y = "Coefficients", title = expression(Ridge))+
  ylim(-1.5,1.5)

# Random Forrest
rfPlot=ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  labs(x = element_blank(), y = "Importance", title = expression(Random.Forest))

Coef.Plot=grid.arrange(laPlot,elPlot, riPlot, rfPlot, nrow = 4)



