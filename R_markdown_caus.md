Causal Analysis and Variable Importance
================
Jon Stewart

Often, learning about the generators of a process is equally important as being able to predict an outcome. Being able to understand what variables matter in terms of driving a process is incredibly important. This allows for optimization in many fields such as marketing, and most sciences. There are often mathematical models, such as differential equations, which are meant to both describe the most important features and explain how changes in those parameters effect the result. Unfortunately, for many problems, there is no underlying descriptive equation that can be put into this form. There may only be data, and this may be noisy, highly nonlinear, very high dimensional and have other problems that make approaching variable explanation from a traditional mathematical modeling standpoint ill posed. This type of problem is suited to machine learning, and there is significant research done in terms of understanding both variable importance and estimating causality measures.

The most straightforward way to show causality is through a randomized test, such as AB testing. If results are significantly different between the two versions, then the variable that is different between the two groups can be said to have caused the differences between those two groups. In cases where explicit testing is not an option, such as situations when one only has access to historical data, there are several options in terms of estimating causality for predictive variables. These include, among others,

-   Kernel Methods based upon the Hilbert-Schmidt conditional information criteria
-   Decision Tree based and Random Forest based methods.
-   Bayesian Networks and Directed Acyclic Graphs.

This project focuses upon Decision tree and Random Forest based methods for determining causality. Kernel methods are very good when there is less data available and when there are a small number of potential causal influences. Tree based methods are flexible, and allow for easier use with higher dimensional datasets. One of the particularly nice features of random forests, or, more generally, any method that uses the random subspace bootstrapping, is that they give feature importance estimates. This is slightly different than causality. Feature importance describes a given variable's importance in determining the accuracy of a prediction. It does not describe the degree to which a given variable is responsible for an outcome. This is because the variables with high feature importance may simply also respond to some causal variable. From a pure prediction standpoint, it might not really matter. Most models, such as traditional deep learning models, do not have any sense of causality, but rely on high order, nonlinear correlations between data in order to make predictions, and they can be remarkably accurate in terms of predictions on challenging data. From an explanatory standpoint however, if the goal is to understand or better optimize the system, being able to identify causal variables is very important.

Even among decision tree methods, there are two different general approaches. The first is to use information theoretic principles, specifically minimum description length. The variables that best describe a given dependent variable when splitting decisions are made on the candidate predictive variables are the ones that lead to a tree with the least depth. It is essentially a compression test. There are arguments for and against this approach. This report focuses upon a different tree based method, called causal trees and causal forests, which more closely aligns with Judea Pearl's do-calculus framework for testing causality. Because this method uses random forests, it is typically much more accurate in terms of prediction than Bayesian network approaches to causality. Since there's obviously a need to be able to predict a variable accurately in order to be able to well describe which variables are causing a given outcome, and random forests typically give a much more accurate prediction than Bayesian networks, this allows for better interpretation of results. Below is a quick R example of a variable importance measure and a causal forest estimate, applied to the well-known pima diabetes dataset, where the goal is to predict the likelihood of developing type II diabetes, given the descriptor variables.

Note that, unlike standard train/test/validation splits, which tend to be around 60/30/10, for Causal inference, there is a a 50/50 split. It's still critical to avoid bias, so half the data set is used to fit the tree, and half is used for inference. In the R package, this is done internally, so there is no reason to split the data.

``` r
pima_diabetes <- as.data.frame(read.csv("~/pima_diabetes.csv", header=FALSE))
X = pima_diabetes[,1:8]
Y = pima_diabetes[,9]
sample_size <- floor(0.5 * nrow(pima_diabetes))


set.seed(123)
train_ind <- sample(seq_len(nrow(pima_diabetes)), size = sample_size)

X_train = pima_diabetes[train_ind,1:8]
X_test = pima_diabetes[-train_ind,1:8]

Y_train = pima_diabetes[train_ind,9]
Y_test = pima_diabetes[-train_ind,9]
```

``` r
rf_fit <- randomForest(X, Y, ntree = 250, importance=TRUE)
```

``` r
kable(as.data.frame(rf_fit$importance))
```

|     |    %IncMSE|  IncNodePurity|
|-----|----------:|--------------:|
| V1  |  0.0169371|       12.78895|
| V2  |  0.0680146|       42.86186|
| V3  |  0.0033212|       12.98563|
| V4  |  0.0038979|       10.34316|
| V5  |  0.0063121|       11.28474|
| V6  |  0.0252671|       25.55034|
| V7  |  0.0060298|       18.59060|
| V8  |  0.0258596|       21.65890|

From the above table, it can be seen that the variable **V2**, is the most significant of the variables. Again, one cannot yet say that **V2** causes Y, as there may be variables that both **V2** and **Y** respond to. **V2** is a continuous variable. This means that instead of a binary treatment, this essentially becomes a regression problem, with all covariates other than **V2** being fixed.

Here is where the causal forest implementation comes in. Image there are three general parameters. **Y**, the response variable, **X**, the vector of predictor variables, and some variable **Z**, which is the variable being tested for causality.

One way to think of decision trees is as a clustering algorithm with respect to the target variable. That is, observations that end up in the same leaf are 'nearby'. Variations in X parameters that matter with respect to the response variable are split upon as a part of fitting the decision tree. Since all other relevant variables are fixed due to clustering, **Note that this is a strong assumption of the model in that the other X variables must capture all possible nuisance parameters. There are workarounds however, that allow for slightly more flexibility in real world situations where this may not be guaranteed.** any differences in the response variable within each leaf must be due to the unfit variable, **Z**. This is a powerful way to test for causality, assuming there is not a question on the direction of causality, which this technique does not answer. Although not the focus of this small project, individual leaf estimates can be used to test for heterogeneity of effects, to see how different subgroups, **X**, of the data respond to the **Z** variable with respect to **Y**. This is the counterfactual reasoning needed to make claims about causality. That is, \(P(Y|do(x))\). Since the learning algorithm used is a random forest model, it is certainly feasible to see how variable importance and causal estimation interact, allowing one to better understand which variables are nuisance variables, and which are responsible for actually driving the process under examination. I chose to show them as separate algorithms here, but causal forests can do both using the same model fit.

Below, the causal forest model is fit, with **V2** as the variable being tested for causality.

``` r
Z <- X[,2]
X1 = X[,-2]

tau.forest <- causal_forest(X1, Y, Z)

tau.hat.oob = predict(tau.forest)
hist(tau.hat.oob$predictions, main = 'Out of bag predictions')
```

![](R_markdown_caus_files/figure-markdown_github/unnamed-chunk-5-1.png)

The above histogram shows the estimated treatment effects using out of bag prediction. Now, let us see the predicted treatment effect as a point value estimate, with a standard error estimate. This is the another great feature of causal trees. The estimates of the causal effect have error that is asymptotically gaussian. This allows for using traditional statistical tools to evaluate confidence intervals, p-values, and other measures of interest. The general causal forest concept can also be extended to Bayesian inference.

``` r
average_partial_effect(tau.forest)
```

    ##     estimate      std.err 
    ## 0.0054633461 0.0005781662

From the above point estimates, we can safely assume that, although **V2** is significant, it does not have a causal effect on the probability of diabetes. That is, **V2** covaries with the probability.

The above report is a toy problem essentially, as this is a very well studied data set, but this should give a better idea regarding the ability of causal forests to identify the important drivers of processes. Additionally, the general concept can be extended to multiple time series, where where Granger causality variants have trouble, or other scenarios, with only minor modifications.