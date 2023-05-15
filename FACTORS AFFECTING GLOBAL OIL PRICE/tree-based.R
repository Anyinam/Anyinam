library(tree)
library(ISLR)
library(tidyverse)
library(magrittr)

# 1. TREES --------------------------------------------------------------------------------


# 1.1 CLASSIFICATION  TREES---------------------------------------------------------------
data("Carseats")
df = Carseats
df$High = as.factor(ifelse(Carseats$Sales<=8, "No", "Yes"))
df %<>% select(-Sales)

# fit a classification tree
fit_tr = tree(High~., data=df)
summary(fit_tr)
plot(fit_tr, col="grey")
text(fit_tr, pretty=0, cex = 0.7)

# change stopping criterion
fit_tr = tree(High~., data=df, control = tree.control(nobs=nrow(df), mincut=50))
summary(fit_tr)
plot(fit_tr, col="grey")
text(fit_tr, pretty=0, cex = 0.7)


# use gini index
fit_tr = tree(High~., data=df, split="gini")
summary(fit_tr)
plot(fit_tr, col="grey")
text(fit_tr, pretty=0, cex = 0.7)


# split train test
set.seed(1)
idx_tr = sample(nrow(df), 0.7*nrow(df))
df_tr = df[idx_tr, ]
df_te = df[-idx_tr,]


# fit a classification tree on training data
fit_tr = tree(High~., data=df_tr)
summary(fit_tr)
pred_tr = predict(fit_tr, df_te, type="class")

table(pred_tr, df_te$High)
mean(pred_tr != df_te$High)
1 - mean(pred_tr != df_te$High)


# tree pruning
fit_cvtr = cv.tree(fit_tr, FUN = prune.misclass)
# show the sequence of tree
fit_cvtr

par(mfrow = c(1,2))
plot(fit_cvtr$size, fit_cvtr$dev/nrow(df_te), type = "b", xlab="Size", ylab="Err")
plot(fit_cvtr$k, fit_cvtr$dev/nrow(df_te), type="b", col=3, xlab="k", ylab="Err")


fit_pr = prune.misclass(fit_tr, k = fit_cvtr$k[which.min(fit_cvtr$dev)])
par(mfrow=c(1,1))
plot(fit_pr, col="grey")
text(fit_pr, pretty=0, cex = 0.7)

pred_pr = predict(fit_pr, df_te, type="class")
table(pred_pr, df_te$High)
mean(pred_pr != df_te$High)
1 - mean(pred_pr != df_te$High)



# 1.2 REGRESSION TREES -------------------------------------------------------------

df2 = MASS::Boston

set.seed(1)
idx_tr = sample(nrow(df2), 0.7*nrow(df2))
df2_tr = df2[idx_tr, ]
df2_te = df2[-idx_tr,]

# fit a regression tree
fit_tr = tree(medv~., data=df2_tr)
summary(fit_tr)
plot(fit_tr, col="grey")
text(fit_tr, pretty=0, cex = 0.7)

pred_tr = predict(fit_tr, df2_te)

mean((pred_tr - df2_te$medv)^2)
var(df2_te$medv)

# tree pruning
fit_cvtr = cv.tree(fit_tr)
# show the sequence of tree
fit_cvtr

par(mfrow = c(1,2))
plot(fit_cvtr$size, fit_cvtr$dev/nrow(df2_te), type = "b", xlab="Size", ylab="Err")
plot(fit_cvtr$k, fit_cvtr$dev/nrow(df2_te), type="b", col=3, xlab="k", ylab="Err")


fit_pr = prune.tree(fit_tr, best = 11)
par(mfrow=c(1,1))
plot(fit_pr, col="grey")
text(fit_pr, pretty=0, cex = 0.7)

par(mfrow=c(1,1))
pred_pr = predict(fit_pr, df2_te)
mean((pred_pr - df2_te$medv)^2)
plot(pred_pr, df2_te$medv, col="#00000060", xlab="predicted", ylab="test")
abline(0,1, col=2)



# 2. BAGGING, RANDOM FORESTS ---------------------------------------------------------------

library(randomForest)



# bagging estimator: random forest with all the predictors
set.seed(1)
fit_bag = randomForest(medv~., data = df2_tr, mtry = ncol(df2_tr)-1, importance = T)

pred_bag = predict(fit_bag, df2_te)
mean((pred_bag - df2_te$medv)^2)
plot(pred_bag, df2_te$medv, col="#00000060", xlab="predicted", ylab="test")
abline(0,1, col=2)


# random forest
fit_rf = randomForest(medv~., data = df2_tr, mtry = 5, importance = T, ntree=1e3 )

pred_rf = predict(fit_rf, df2_te)
mean((pred_rf - df2_te$medv)^2)
plot(pred_rf, df2_te$medv, col="#00000060", xlab="predicted", ylab="test")
abline(0,1, col=2)


importance(fit_rf)
varImpPlot(fit_rf)


# get oob error
fit_rf$mse


# 3. BOOSTING ---------------------------------------------------------------------------
install.packages("gbm")
install.packages("fastAdaboost")
install.packages("adabag")
install.packages("xgboost")
install.packages("reticulate")


library(gbm)
library(fastAdaboost)
library(xgboost)
library(reticulate)
library(caret)


# 3.1 gradient boosting ---------------------------------------------------------------
# i.e. boosted regression / classification trees
fit_gb = gbm(medv ~., data=df2_tr, distribution = "gaussian", 
             n.trees = 200, interaction.depth = 4, bag.fraction = 1, cv.folds = 10)
summary(fit_gb)



# partial dependence plot
par(mfrow = c(1,2))
plot(fit_gb, i="rm")
plot(fit_gb, i="lstat")

pred_gb = predict(fit_gb, df_te)

mean((pred_gb - df_te$medv)^2)
plot(pred_gb, df_te$medv, col="#00000060", xlab="predicted", ylab="test")
abline(0,1, col=2)


# get training and cv error curves, choose optimal number of trees
gbm.perf(fit_gb, oobag.curve = F, method = "cv")
fit_gb$cv.error
plot(fit_gb$cv.error, type="l")




# 3.2 adaboost ----------------------------------------------------------------------


# adaboost, method 1 - using adabag, slow but more control
fit_ab = boosting(High ~., boos=F, data=df_tr, mfinal= 50, control = rpart.control(maxdepth = 2))
names(fit_ab)
# automatically includes classification error and confusion matrix
pred_ab = predict(fit_ab, df_te)
pred_ab$error
pred_ab$confusion


# adaboost, method 2 - using fastAdaboost, fast but no control
fit_ab2 = adaboost(High ~., nIter=50, data=df_tr)
pred_ab2 = predict(fit_ab2, df_te)
pred_ab2$error
pred_ab2$prob
pred_ab2$class

# adaboost, FORBIDDEN METHOD - use python's sklearn

# reference: https://cran.r-project.org/web/packages/reticulate/vignettes/calling_python.html
# sklearn adaboost: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html?highlight=adaboost#sklearn.ensemble.AdaBoostClassifier
skl_ab = import("sklearn.ensemble") # equivalent of import sklearn.ensemble as skl_ab
skl_cv = import("sklearn.model_selection")$cross_val_score

X_tr = dummyVars(High ~ ., data=df_tr) %>% predict(df_tr)
y_tr = df_tr$High %>% as.numeric(.)-1 #class2ind(df_tr$High)[,2]
X_te = dummyVars(High ~ ., data=df_te) %>% predict(df_te)
y_te = df_te$High %>% as.numeric(.)-1

fit_ab = skl_ab$AdaBoostClassifier()
fit_ab$fit(X_tr, y_tr)

fit_ab$predict(X_te)
pred_ab = ifelse(fit_ab$predict(X_te), "Yes", "No") %>% as.factor

mean(pred_ab != df_te$High)

cv_ab = numeric(20)
err_ab = numeric(20)
set.seed(1)
for(i in 1:20){
  # set estimator
  n_est = 10L*i
  fit_ab = skl_ab$AdaBoostClassifier(n_estimators = n_est)
  fit_ab$fit(X_tr, y_tr)
  # cross validation score
  cv_ab[i] = 1 - mean(skl_cv(fit_ab, X_tr, y_tr))
  # test error
  fit_ab$predict(X_te)
  pred_ab = ifelse(fit_ab$predict(X_te), "Yes", "No") %>% as.factor
  err_ab[i] = mean(pred_ab != df_te$High)
}

plot(10L*1:20, err_ab, type="b", xlab="number of trees", ylab="error", col=3)
points(10L*1:20, cv_ab, type="b")
legend("topright", legend = c("CV", "Test"), col=c(1, 3), lty=1, lwd=2, cex = 0.7)
which.min(cv_ab)
# what are your conclusions?



# 3.3 XGBOOST --------------------------------------------------------
# read the docs: https://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html
fit_xg = xgboost(data = X_tr, label = y_tr, nrounds = 50, 
                 objective = "binary:logistic", eval_metric = "error")

pred_xg = ifelse(predict(fit_xg, X_te)> 0.5, 1, 0)
mean(pred_xg != y_te)


# monitor performance on validation/test set
err_xg_tr = fit_xg$evaluation_log$train_error
err_xg_te = numeric(50)

for(j in 1:50){
   pred_j = ifelse(predict(fit_xg, X_te, ntreelimit = j)> 0.5, 1, 0)
   err_xg_te[j] = mean(pred_j != y_te)
}

plot(1:50, err_xg_te, type="b", xlab="number of trees", ylab="error", col=3, ylim = c(0,0.3), cex=0.5)
points(1:50, err_xg_tr, type="b", cex=0.5)
legend("topright", legend = c("Train", "Test"), col=c(1, 3), lty=1, lwd=2, cex = 0.7)
which.min(cv_ab)

# can we choose the number of trees from this plot?

# feature importance
imp_matrix = xgb.importance(model = fit_xg)
print(imp_matrix)
xgb.plot.importance(importance_matrix = imp_matrix)


# cross validation for tuning parameter selection
# caret pkg documentation: https://topepo.github.io/caret/index.html
# in particular consider section 7



fitControl = trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 10)

fitControl = trainControl(
  method = "cv",
  number = 5, 
  search="random")

# what is repeated cv?
# why can it be beficial? (Hint. think of variance)

# go parallel
library(doParallel)
cl = makePSOCKcluster(5)
registerDoParallel(cl)
stopCluster(cl)

tune_grid = 
  expand.grid(
    nrounds = c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
    eta = 0.3,
    max_depth=5,
    subsample = 1,
    colsample_bytree = 1,
    min_child_weight = 5,
    gamma = c(0.1, 0.2, 0.5, 0.75, 1)
  )
set.seed(1)
fit_xg_cv = train(High ~ ., data = df_tr, 
                 method = "xgbTree", 
                 trControl = fitControl,
                 verbose = FALSE, 
                 tuneGrid = tune_grid,
                 objective = "binary:logistic", 
                 eval_metric = "error")
fit_xg_cv
trellis.par.set(caretTheme())
plot(fit_xg_cv)  

pred_xg_cv = predict(fit_xg_cv, df_te, type="raw")


mean(pred_xg_cv != df_te$High)

