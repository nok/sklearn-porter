# Estimator Notes

## Class labels

- The ported classifiers return the labels of the classes like scikit-learn (`classes_`), e.g. `-1` and `1`, `1.5` or `'setosa'`. The labels are stored in the ported code as a constant (`labels` in C and Go, `LABELS` in the other languages), also by the `exported` template.
- `predict_index` (`predictIndex` in Java, JavaScript and PHP and `PredictIndex` in C# and Go) returns the index of the predicted class instead, which is also the position of its probability in the result of `predict_proba`.

## Multilabel and multi-output classification

- The classifiers `RidgeClassifier` and `MLPClassifier` also support multilabel classification, i.e. if they were fitted on a label indicator matrix, the ported `predict` returns one binary value per label (without `predict_index`).
- The classifiers `DecisionTreeClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier` and `KNeighborsClassifier` also support multi-output classification, i.e. if they were fitted on a target with several columns, the ported `predict` returns the label of one class per output. Like in scikit-learn, the ported `predict_proba` returns the class probabilities of each output.

## BernoulliNB

- [BernoulliNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html) with feature or class probabilities of 0 or 1 (e.g. with `alpha=0`) can't be ported and raises a `NotSupportedYetError`, because scikit-learn itself computes undefined (`NaN`) joint log likelihoods with them.

## MultinomialNB and ComplementNB

- [MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) and [ComplementNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html) with infinite feature log probabilities (e.g. with `alpha=0` and features without counts in a class or, with ComplementNB, in all other classes) can't be ported and raise a `NotSupportedYetError`, because scikit-learn itself computes undefined (`NaN`) joint log likelihoods with them. The same applies to a MultinomialNB with class probabilities of 0 (e.g. with `class_prior` or classes without samples after `partial_fit`).
- Like scikit-learn, the ported ComplementNB uses the class priors only if it was fitted on a single class, otherwise `class_prior` and `fit_prior` don't change its predictions.
- The features are usually word counts (e.g. of a [`CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)) or tf-idf values (e.g. of a [`TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)). The vectorizer isn't ported, so you have to compute the features of the ported estimator yourself, and with a large vocabulary the dense model data gets large (see the [limitations](limitations.md)).

## GradientBoostingClassifier and GradientBoostingRegressor

- The ported [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) and [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) start with the constant raw predictions of the initial estimator (`init_`) and add the values of the leaves of all trees, which are multiplied by the learning rate in advance. They add the same values in the same order as scikit-learn, so the raw predictions and the predictions are the same; the probabilities can differ in the last digits (see the [limitations](limitations.md)).
- The default initial estimator and `init='zero'` are supported. An `init` estimator whose predictions depend on the features (e.g. a `LogisticRegression`) or a `DummyClassifier(strategy='stratified')` can't be ported and raises a `NotSupportedYetError`.
- A multiclass GradientBoostingClassifier has one tree per class and stage (e.g. 1000 trees with 10 classes and the default `n_estimators=100`), so the generated code can get large. Use the `exported` template for many trees, especially with Java.
- [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) and [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) use another structure of trees and aren't supported yet.

## IsolationForest

- The ported [IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) supports only `predict`, which returns `1` for inliers and `-1` for outliers like scikit-learn. The scores (`score_samples` and `decision_function`) aren't ported.
- A sample is an outlier if its score is lower than `offset_`, which scikit-learn derives from `contamination` (`-0.5` with `contamination='auto'`). To change the threshold, fit the estimator with another `contamination` or set `offset_` before you port it. The template `exported` saves the offset in the JSON file (`offset`) next to the trees.
- Samples whose score is (almost) equal to the offset can be predicted differently than by scikit-learn, because the power function of each language (and of NumPy, depending on the CPU) can round the last digit differently. With a float `contamination` the offset is a percentile of the scores of the training samples, so some training samples can be such samples.
- [PyOD](https://github.com/yzhao062/pyod) wraps the IsolationForest of scikit-learn, so you can port its fitted estimator (`detector_`). PyOD uses its own threshold and inverted scores, so its labels can differ from the ported `predict`.

## KNeighborsClassifier

- [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) can be ported with the weights `uniform` and `distance` and the metrics `minkowski` (e.g. `euclidean` or `manhattan`), `chebyshev`, `cosine` and `hamming`. Other metrics (e.g. `canberra` or a callable), the weighted Minkowski distance (`metric_params={'w': ...}`) and callable weights raise a `NotSupportedYetError`. With `weights='distance'` the neighbors vote with the inverse of their distances; like in scikit-learn, only the neighbors with the distance 0 vote (with equal weights) if there are any.
- [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) can predict different classes and probabilities than scikit-learn if several training samples have the same distance as the `n_neighbors`-th nearest neighbor (a tie). The ported estimator picks the first of these samples in the order of the training data. scikit-learn picks them depending on its internal search: the `algorithm` (`brute`, `kd_tree` or `ball_tree`, chosen automatically with `'auto'`), the `leaf_size` and the order of the computations. So `predict` and `predict_proba` can differ for samples with ties and match for all other samples. Ties are common with the metrics `hamming`, `chebyshev` and `manhattan` and with few features or discrete feature values (e.g. integers). They can't be detected when the estimator is ported, because they depend on the input samples. Use `test()` with representative samples to check how often the predictions match.

## SVC and NuSVC

- [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and [NuSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html) with `break_ties=True` can't be ported and raise a `NotSupportedYetError`.
- The ported [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and [NuSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html) compute the decision values like libsvm: the polynomial kernel with repeated multiplications (instead of a power function) and the sums over the support vectors in the same order. scikit-learn computes the dot products with BLAS and, depending on the CPU and the compiler, with fused multiply-add instructions, so the decision values can still differ in the last digits and samples on the decision boundary can get different predictions.
