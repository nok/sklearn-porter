# Estimator Notes

## Class labels

- The ported classifiers return the labels of the classes like scikit-learn (`classes_`), e.g. `-1` and `1`, `1.5` or `'setosa'`. The labels are stored in the ported code as a constant (`labels` in C and Go, `LABELS` in the other languages), also by the `exported` template.
- `predict_index` (`predictIndex` in Java, JavaScript and PHP and `PredictIndex` in C# and Go) returns the index of the predicted class instead, which is also the position of its probability in the result of `predict_proba`.

## Multilabel and multi-output classification

- The classifiers `RidgeClassifier`, `RidgeClassifierCV` and `MLPClassifier` also support multilabel classification, i.e. if they were fitted on a label indicator matrix, the ported `predict` returns one binary value per label (without `predict_index`).
- The classifiers `DecisionTreeClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier` and `KNeighborsClassifier` also support multi-output classification, i.e. if they were fitted on a target with several columns, the ported `predict` returns the label of one class per output. Like in scikit-learn, the ported `predict_proba` returns the class probabilities of each output.

## DecisionTreeClassifier, RandomForestClassifier and ExtraTreesClassifier

- Like scikit-learn, the ported tree classifiers return the class fractions of the leaves (`tree_.value`, weighted by `sample_weight` and `class_weight`) as probabilities, and the forests average them over all trees in the same order. So the probabilities are identical to those of scikit-learn.

## BernoulliNB

- [BernoulliNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html) with feature or class probabilities of 0 or 1 (e.g. with `alpha=0`) can't be ported and raises a `NotSupportedYetError`, because scikit-learn itself computes undefined (`NaN`) joint log likelihoods with them.

## MultinomialNB and ComplementNB

- [MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) and [ComplementNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html) with infinite feature log probabilities (e.g. with `alpha=0` and features without counts in a class or, with ComplementNB, in all other classes) can't be ported and raise a `NotSupportedYetError`, because scikit-learn itself computes undefined (`NaN`) joint log likelihoods with them. The same applies to a MultinomialNB with class probabilities of 0 (e.g. with `class_prior` or classes without samples after `partial_fit`).
- Like scikit-learn, the ported ComplementNB uses the class priors only if it was fitted on a single class, otherwise `class_prior` and `fit_prior` don't change its predictions.
- The features are usually word counts (e.g. of a [`CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)) or tf-idf values (e.g. of a [`TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)). The vectorizer isn't ported, so you have to compute the features of the ported estimator yourself, and with a large vocabulary the dense model data gets large (see the [limitations](limitations.md)).

## Linear models

- The estimators with cross-validation ([RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html), [LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html), [ElasticNetCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html), [LogisticRegressionCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html) and [RidgeClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html)) are ported with their final coefficients (`coef_` and `intercept_`), which scikit-learn uses for the predictions. Usually these are the coefficients of the estimator refitted on all samples with the best parameters; a `LogisticRegressionCV` with `refit=False` uses the means of the best coefficients of all folds.
- The ported [BayesianRidge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html) and [ARDRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html) return the mean of the predictive distribution like `predict()`. The standard deviation (`predict(X, return_std=True)`) isn't ported.

## GradientBoostingClassifier and GradientBoostingRegressor

- The ported [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) and [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) start with the constant raw predictions of the initial estimator (`init_`) and add the values of the leaves of all trees, which are multiplied by the learning rate in advance. They add the same values in the same order as scikit-learn, so the raw predictions and the predictions are the same; the probabilities can differ in the last digits (see the [limitations](limitations.md)).
- The default initial estimator and `init='zero'` are supported. An `init` estimator whose predictions depend on the features (e.g. a `LogisticRegression`) or a `DummyClassifier(strategy='stratified')` can't be ported and raises a `NotSupportedYetError`.
- A multiclass GradientBoostingClassifier has one tree per class and stage (e.g. 1000 trees with 10 classes and the default `n_estimators=100`), so the generated code can get large. Use the `exported` template for many trees, especially with Java.

## HistGradientBoostingClassifier and HistGradientBoostingRegressor

- The ported [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) and [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) start with the baseline prediction and add the values of the leaves of the trees of all iterations (which already contain the learning rate) in the same order as scikit-learn. They compare the raw features with the thresholds in double precision like scikit-learn, so the raw predictions are the same; the probabilities and the predictions of the losses `poisson` and `gamma` (the exponential of the raw predictions) can differ in the last digits (see the [limitations](limitations.md)).
- All losses are supported: `squared_error`, `absolute_error`, `gamma`, `poisson` and `quantile` of the regressor and `log_loss` of the classifier. A binary classifier predicts the negative class if the raw prediction is exactly 0 like scikit-learn (unlike the GradientBoostingClassifier, which predicts the positive class).
- Missing values (NaN) go to the same child of each split as in scikit-learn (`missing_go_to_left`).
- Categorical features (`categorical_features`, e.g. the columns with the dtype `category` of a pandas DataFrame) are supported if their categories are numbers. The ported estimator expects the features in their original order and the categories as values, e.g. `2.5`, not as their codes, because scikit-learn encodes them internally with an `OrdinalEncoder`. Like in scikit-learn, missing values and unknown categories go to the child of the missing values and negative values are ordinary categories. Categories which aren't numbers (e.g. strings) raise a `NotSupportedYetError`, so encode them as numbers before training.
- The templates `attached` and `exported` replace each categorical split by a chain of numeric splits, which checks the categories one after another (two splits per category). The template `combined` compares the features with the categories directly.
- The trees are read from the internal attribute `_predictors` of scikit-learn, because there is no public attribute for them.
- A multiclass HistGradientBoostingClassifier has one tree per class and iteration (e.g. 1000 trees with 10 classes and the default `max_iter=100`), so the generated code can get large. Use the `exported` template for many trees, especially with Java.

## IsolationForest

- The ported [IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) supports only `predict`, which returns `1` for inliers and `-1` for outliers like scikit-learn. The scores (`score_samples` and `decision_function`) aren't ported.
- A sample is an outlier if its score is lower than `offset_`, which scikit-learn derives from `contamination` (`-0.5` with `contamination='auto'`). To change the threshold, fit the estimator with another `contamination` or set `offset_` before you port it. The template `exported` saves the offset in the JSON file (`offset`) next to the trees.
- Samples whose score is (almost) equal to the offset can be predicted differently than by scikit-learn, because the power function of each language (and of NumPy, depending on the CPU) can round the last digit differently. With a float `contamination` the offset is a percentile of the scores of the training samples, so some training samples can be such samples.
- [PyOD](https://github.com/yzhao062/pyod) wraps the IsolationForest of scikit-learn, so you can port its fitted estimator (`detector_`). PyOD uses its own threshold and inverted scores, so its labels can differ from the ported `predict`.

## KNeighborsClassifier and KNeighborsRegressor

- [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) and [KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) can be ported with the weights `uniform` and `distance` and the metrics `minkowski` (e.g. `euclidean` or `manhattan`), `chebyshev`, `cosine` and `hamming`. Other metrics (e.g. `canberra` or a callable), the weighted Minkowski distance (`metric_params={'w': ...}`) and callable weights raise a `NotSupportedYetError`. With `weights='distance'` the neighbors vote with the inverse of their distances; like in scikit-learn, only the neighbors with the distance 0 vote (with equal weights) if there are any.
- The ported [KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) predicts the mean (or with `weights='distance'` the weighted mean) of the targets of the nearest neighbors, also of several outputs. It stores all training samples and their targets, so the ported code gets as large as the training data.
- [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) can predict different classes and probabilities (and [KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) different values) than scikit-learn if several training samples have the same distance as the `n_neighbors`-th nearest neighbor (a tie). The ported estimator picks the first of these samples in the order of the training data. scikit-learn picks them depending on its internal search: the `algorithm` (`brute`, `kd_tree` or `ball_tree`, chosen automatically with `'auto'`), the `leaf_size` and the order of the computations. So the predictions can differ for samples with ties and match for all other samples. Ties are common with the metrics `hamming`, `chebyshev` and `manhattan` and with few features or discrete feature values (e.g. integers). They can't be detected when the estimator is ported, because they depend on the input samples. Use `test()` with representative samples to check how often the predictions match.

## SVC, NuSVC, SVR, NuSVR and OneClassSVM

- [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and [NuSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html) with `break_ties=True` can't be ported and raise a `NotSupportedYetError`.
- The ported [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), [NuSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html), [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html), [NuSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html) and [OneClassSVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html) compute the decision values like libsvm: the polynomial kernel with repeated multiplications (instead of a power function) and the sums over the support vectors in the same order. scikit-learn computes the dot products with BLAS and, depending on the CPU and the compiler, with fused multiply-add instructions, so the decision values (and the predicted values of the regressors) can still differ in the last digits and samples on the decision boundary can get different predictions.
- The SVMs can be ported with the kernels `linear`, `poly`, `rbf` and `sigmoid` and the templates `attached` and `exported`. The kernel `precomputed` and callable kernels raise a `NotSupportedYetError`.
- The ported [OneClassSVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html) supports only `predict`, which returns `1` for inliers (a positive decision value) and `-1` for outliers like scikit-learn. The scores (`score_samples` and `decision_function`) aren't ported.
- A [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) without support vectors (e.g. with a large `epsilon`) predicts its intercept for all samples, also the ported one.

## Transformers

- The model data of the transformers (e.g. the mean of a feature) must be finite numbers. A transformer with missing or infinite values in its model data (e.g. a [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) fitted on a feature with missing values only) raises a `NotSupportedYetError`.
- scikit-learn transforms samples of the type `float32` with single precision, the ported transformers always use double precision. So pass samples of the type `float64` to compare the results.

## StandardScaler, MinMaxScaler, MaxAbsScaler and RobustScaler

- The ported scalers compute the transformed features with the same operations in the same order as scikit-learn, so the transformed features are identical: the [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) and the [RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html) subtract the mean (or the center) and divide by the scale, the [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) multiplies by the scale and adds the minimum (without fused multiply-add) and the [MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html) divides by the maximum absolute value.
- Without centering (`with_mean=False` or `with_centering=False`) or scaling (`with_std=False` or `with_scaling=False`) the scalers subtract zero or divide by one, which doesn't change any value.
- The [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) with `clip=True` clips the transformed features to `feature_range` like scikit-learn; the bounds are numbers in the code, also with the template `exported`.
- Missing values (`NaN`) remain missing values like in scikit-learn.

## Binarizer

- The ported [Binarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html) returns `1.0` for features greater than the threshold and `0.0` otherwise. The threshold is stored once per feature in the model data.
- scikit-learn rejects samples with missing values (`NaN`), the ported transformer returns `0.0` for them.

## SimpleImputer

- The ported [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) replaces the missing values of each feature by its statistic (the mean, the median, the most frequent value or the constant `fill_value`), so the transformed features are identical.
- The missing values (`missing_values`) can be `NaN` (default) or a number. Other missing values (e.g. `None` or `pd.NA`) and statistics which aren't numbers (e.g. a string as `fill_value`) raise a `NotSupportedYetError`.
- Features without any observed values in the fit are dropped by scikit-learn, which changes the number of features, so they raise a `NotSupportedYetError`. Pass `keep_empty_features=True` to keep them. `add_indicator=True` (additional features) isn't supported either.
