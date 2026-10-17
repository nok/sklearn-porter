# Estimator Notes

## Class labels

- The ported classifiers return the labels of the classes like scikit-learn (`classes_`), e.g. `-1` and `1`, `1.5` or `'setosa'`. The labels are stored in the ported code as a constant (`labels` in C and Go, `LABELS` in the other languages), also by the `exported` template.
- `predict_index` (`predictIndex` in Java, JavaScript and PHP and `PredictIndex` in Go) returns the index of the predicted class instead, which is also the position of its probability in the result of `predict_proba`.

## Multilabel and multi-output classification

- The classifiers `RidgeClassifier` and `MLPClassifier` also support multilabel classification, i.e. if they were fitted on a label indicator matrix, the ported `predict` returns one binary value per label (without `predict_index`).
- The classifiers `DecisionTreeClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier` and `KNeighborsClassifier` also support multi-output classification, i.e. if they were fitted on a target with several columns, the ported `predict` returns the label of one class per output. Like in scikit-learn, the ported `predict_proba` returns the class probabilities of each output.

## BernoulliNB

- [BernoulliNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html) with feature or class probabilities of 0 or 1 (e.g. with `alpha=0`) can't be ported and raises a `NotSupportedYetError`, because scikit-learn itself computes undefined (`NaN`) joint log likelihoods with them.

## KNeighborsClassifier

- [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) can only be ported with `weights='uniform'` and the metrics `minkowski` (e.g. `euclidean` or `manhattan`), `chebyshev`, `cosine` and `hamming`. Other metrics (e.g. `canberra` or a callable) and the weighted Minkowski distance (`metric_params={'w': ...}`) raise a `NotSupportedYetError`.
- [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) can predict different classes and probabilities than scikit-learn if several training samples have the same distance as the `n_neighbors`-th nearest neighbor (a tie). The ported estimator picks the first of these samples in the order of the training data. scikit-learn picks them depending on its internal search: the `algorithm` (`brute`, `kd_tree` or `ball_tree`, chosen automatically with `'auto'`), the `leaf_size` and the order of the computations. So `predict` and `predict_proba` can differ for samples with ties and match for all other samples. Ties are common with the metrics `hamming`, `chebyshev` and `manhattan` and with few features or discrete feature values (e.g. integers). They can't be detected when the estimator is ported, because they depend on the input samples. Use `test()` with representative samples to check how often the predictions match.

## SVC and NuSVC

- [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and [NuSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html) with `break_ties=True` can't be ported and raise a `NotSupportedYetError`.
