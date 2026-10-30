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

## ColumnTransformer and FeatureUnion

- The ported [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) transforms the selected features (columns) of each part with the ported transformers of the part and concatenates the transformed features in the order of scikit-learn (`output_indices_`), the [FeatureUnion](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html) transforms all features with each part. A part can be a supported transformer, a pipeline of supported transformers or `'passthrough'`; dropped parts (`'drop'`), the remainder (`'drop'`, `'passthrough'` or a transformer) and the weights of the parts (`transformer_weights`) are supported. The columns can be selected by positions, slices, masks, names or callables (the fitted column transformer resolves them).
- A composite transformer with an unsupported transformer in a part raises a `NotSupportedYetError`. The ported transformer reads the features as numbers, so the columns must contain numbers (no strings).

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
- Features without any observed values in the fit are dropped like in scikit-learn (except with `keep_empty_features=True`). With `add_indicator=True` the indicators of the missing values (`1.0` or `0.0`) of the features with missing values in the fit follow the imputed features like in scikit-learn.

## FunctionTransformer

- The ported [FunctionTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html) supports the identity (`func=None`) and the functions `np.abs`, `np.sqrt`, `np.square`, `np.exp`, `np.expm1`, `np.log`, `np.log1p`, `np.negative`, `np.reciprocal`, `np.floor`, `np.ceil`, `np.sign` and `np.clip`. Other functions and `kw_args` of other functions raise a `NotSupportedYetError`.
- `np.clip` needs numbers as bounds in `kw_args`: `a_min` and `a_max` (one of them can be `None`) or `min` and/or `max`. The lower bound must not be greater than the upper bound, arrays as bounds aren't supported. With one bound NumPy uses `np.maximum` or `np.minimum`, which treat `-0.0` as lower than `0.0` (e.g. on aarch64), the ported transformer does the same. On other CPUs the sign of a zero at the bound can differ.
- The identity and the functions `np.abs`, `np.sqrt`, `np.square`, `np.negative`, `np.reciprocal`, `np.floor`, `np.ceil`, `np.sign` and `np.clip` return identical values. NumPy computes `np.exp` and `np.log` (and their variants) with SIMD instructions depending on the CPU, so their results can differ by a few ULP from those of the target languages. C# and Ruby have no `log1p` and `expm1`, so the ported transformer computes them with an accurate formula, which can differ by a few ULP as well.

## Normalizer

- The ported [Normalizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html) divides the features of each sample by its norm (or by one, if the norm is close to zero) like scikit-learn.
- With `norm='max'` and `norm='l1'` the transformed features are identical: the ported transformer sums the absolute values in the order of NumPy (pairwise summation of a contiguous row, i.e. of samples in C order; NumPy sums the samples of an array in Fortran order in another order).
- With `norm='l2'` (default) scikit-learn sums the squares with `np.einsum`, whose order depends on the CPU, so the transformed features can differ by a few ULP.

## KBinsDiscretizer

- The ported [KBinsDiscretizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html) counts the inner edges of the bins of each feature which are lower than or equal to the feature like scikit-learn, so the transformed features are identical.
- With `encode='onehot'` or `encode='onehot-dense'` each bin is a transformed feature, which is `1.0` for the bin of the feature and `0.0` otherwise. The ported transformer always returns all transformed features (also for the sparse matrix of `encode='onehot'`).

## OrdinalEncoder

- The ported [OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) returns the position of the feature in the categories of each feature like scikit-learn. Only numbers are supported as categories; strings and infrequent categories (`max_categories` or `min_frequency`) raise a `NotSupportedYetError`.
- Missing values (`NaN`) seen in the fit get `encoded_missing_value`. Unknown categories get `unknown_value` with `handle_unknown='use_encoded_value'`, scikit-learn raises an error for them by default, the ported transformer returns `NaN` instead.

## OneHotEncoder

- The ported [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) returns one transformed feature per category of each feature (without the dropped categories of `drop`), which is `1.0` if the feature is the category and `0.0` otherwise, like scikit-learn. The ported transformer always returns all transformed features (also for the sparse matrix of `sparse_output=True`).
- Only numbers are supported as categories; strings and infrequent categories (`max_categories` or `min_frequency`) raise a `NotSupportedYetError`. Missing values (`NaN`) seen in the fit are a category.
- Unknown categories get only zeros with `handle_unknown='ignore'` (or `'infrequent_if_exist'` and `'warn'` without infrequent categories), scikit-learn raises an error for them by default, the ported transformer returns zeros as well.

## PolynomialFeatures

- The ported [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) multiplies the features of each transformed feature in the order of scikit-learn (each term of degree `d` is a term of degree `d - 1` multiplied by a feature), so the transformed features are identical. `degree` can be the maximal degree or a tuple of the minimal and the maximal degree, `interaction_only` and `include_bias` are supported.
- The number of transformed features grows with the power of the degree, with many features the generated code (especially of the template `combined`) can get very large.
- scikit-learn multiplies the features of sparse matrices in another order, so the transformed features of sparse samples (e.g. of a `StandardScaler(with_mean=False)` fitted on sparse data) can differ by a few ULP from degree 3 on. Transformed features of zeros and ones (e.g. of a `OneHotEncoder`) are identical.

## TargetEncoder

- The ported [TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html) returns the encodings of the categories (`encodings_`) like `transform` of scikit-learn, so the transformed features are identical. With several classes (`target_type_` is `multiclass`) there is one transformed feature per feature and class.
- `fit_transform` of scikit-learn encodes the samples of the fit with a cross fitting scheme (different from `transform`); the ported transformer corresponds to `transform`, which a pipeline uses for predictions.
- Missing values (`NaN`) seen in the fit get their encoding, unknown categories (also unseen missing values) get the mean of the target (`target_mean_`) like in scikit-learn. Only numbers are supported as categories; strings raise a `NotSupportedYetError`.

## PowerTransformer

- The ported [PowerTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html) supports both methods with the formulas of SciPy (`scipy.stats.yeojohnson` and `scipy.special.boxcox`, also their special cases for the parameters close to `0` and `2` and the large values of Box-Cox) and standardizes the transformed features like scikit-learn (`standardize=True`).
- C and PHP use the same `log`, `exp`, `log1p` and `expm1` as NumPy and SciPy (of the C library), the functions of the other languages can differ by a few ULP. NumPy computes some of them with SIMD instructions depending on the CPU as well. C# and Ruby have no `log1p` and `expm1`, so the ported transformer computes them with an accurate formula.

## QuantileTransformer

- The ported [QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html) supports only `output_distribution='uniform'`: it interpolates each feature between its quantiles in both directions with the operations of `np.interp` like scikit-learn, so the transformed features are identical. The normal distribution needs the quantile function of SciPy (`scipy.stats.norm.ppf`), so it raises a `NotSupportedYetError`.
- Each feature has up to `n_quantiles` (default: 1000) quantiles, so the model data and especially the template `combined` can get very large (e.g. the size of a method in Java is limited); use the template `exported` or fewer quantiles.
- scikit-learn transforms only the non-zero values of sparse matrices (`ignore_implicit_zeros`), the ported transformer transforms all values like those of dense samples.

## CountVectorizer

- The ported [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) reads documents instead of numbers: one document from the arguments (joined by spaces) or one document per line from the standard input. `make()` and `test()` take a list of documents (strings or bytes, which are decoded with `encoding` like scikit-learn) or a single document. Line breaks and null characters of the documents are replaced by spaces, because they separate the tokens like spaces.
- Like scikit-learn, the ported vectorizer converts the document to lowercase (`lowercase=True`), splits it into tokens of two or more word characters (the default `token_pattern`), removes the stop words (`stop_words`), joins the n-grams of the remaining tokens with spaces (`ngram_range`) and counts the terms of the vocabulary (or sets them to one with `binary=True`). The vocabulary is the fitted `vocabulary_` or the fixed `vocabulary`, which can be ported without a fit. The parameters `max_df`, `min_df` and `max_features` only change the vocabulary of the fit. Only the stop words which can change the counts are ported.
- Other analyzers (`analyzer='char'`, `'char_wb'` or a callable), a `preprocessor`, a `tokenizer`, another `token_pattern`, `strip_accents` and `input='filename'` or `'file'` raise a `NotSupportedYetError`.
- The counts of documents with ASCII characters only are identical in all languages. Outside of ASCII the languages use their own Unicode support: the word characters are the letters, numbers and the underscore (`[\p{L}\p{N}_]` like `\w` of Python), so without the conversion to lowercase (`lowercase=False`) the counts are identical (except in C). The conversion to lowercase of Java and JavaScript is identical to Python, C# and Go convert some characters differently (e.g. the final sigma of Greek and the dotted capital I of Turkish), Ruby the final sigma. PHP converts all characters with the extension `mbstring`, otherwise only ASCII characters. C treats all characters outside of ASCII as word characters and converts only ASCII characters to lowercase.
- The ported vectorizer returns dense arrays, so the size of the vocabulary determines the number of transformed features. A large vocabulary can exceed the 64 KB bytecode limit of a method in Java with the templates `attached` and `combined` (see the limitations).

## HashingVectorizer

- The ported [HashingVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html) splits the documents into terms like the [CountVectorizer](#countvectorizer) (with the same parameters, limitations and Unicode support), but instead of a vocabulary it maps each term to a feature with the signed 32-bit MurmurHash3 (x86_32, seed 0) of its UTF-8 bytes like scikit-learn: the index is the absolute value of the hash modulo `n_features`, the value is `1` or (with `alternate_sign=True`) the sign of the hash. Terms with the same index (collisions) are summed. Finally the features are normalized (`norm`, unless the norm is zero), so the transformed features are identical.
- With `binary=True` each feature of a term is `1` like in scikit-learn, also if the signs of colliding terms sum up to zero.
- The vectorizer is stateless: it can be ported without a fit, its parameters are validated like in `fit` of scikit-learn. The model data consists of the stop words only, so the ported vectorizer is small, but it returns dense arrays of `n_features` values (`2**20` by default), so choose a small `n_features` (e.g. `2**10` to `2**16`) for the ported vectorizer and the estimator.

## TfidfTransformer

- The ported [TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html) transforms the counts of the terms of a document (e.g. of a `CountVectorizer`) into tf-idf features like scikit-learn: it applies `1 + log(tf)` (with `sublinear_tf=True`), multiplies the counts by `idf_` and divides them by their norm (`l1` or `l2`). It weights the counts with `idf_`, if the fit created it (`use_idf=True`), also if `use_idf` was changed after the fit, like scikit-learn.
- The ported transformer handles the counts like a sparse matrix of scikit-learn: zeros (also negative zeros) remain zeros (`0.0`), the logarithm is only computed of counts which aren't zero, and samples whose norm is zero (e.g. if the squares of very small counts are zero) aren't normalized. It sums the counts one after another like scikit-learn, so the transformed features are identical. Only with `sublinear_tf=True` they can differ by a few ULP, because NumPy computes the logarithm with SIMD instructions depending on the CPU.
- Without smoothing (`smooth_idf=False`) the idf of a feature which has no counts in the samples of the fit is infinite, which raises a `NotSupportedYetError` (see above).
- The ported transformer returns dense arrays, so the number of features (e.g. the size of the vocabulary) determines the size of the transformed features.

## TfidfVectorizer

- The ported [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) counts the terms of a document like the [CountVectorizer](#countvectorizer) (with the same parameters, limitations and Unicode support) and weights the counts like the [TfidfTransformer](#tfidftransformer): `1 + log(tf)` (with `sublinear_tf=True`), the multiplication by `idf_` and the division by the norm (`l1` or `l2`). With `binary=True` only the counts are binary, the idf and the norm are still applied like in scikit-learn.
- Like `transform` of scikit-learn, the ported vectorizer uses the parameters `norm`, `use_idf` and `sublinear_tf` of the fit (the internal `TfidfTransformer`), also if they were changed after the fit. Unlike a `CountVectorizer` it needs a fit, also with a fixed `vocabulary`.
- The weighted counts are summed one after another like scikit-learn, so the transformed features are identical. Only with `sublinear_tf=True` they can differ by a few ULP (the logarithm of NumPy). With `dtype=np.float32` scikit-learn computes the features with single precision, the ported vectorizer always with double precision.

## Feature selectors

- The ported feature selectors ([GenericUnivariateSelect](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html), [RFE](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html), [RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html), [SelectFdr](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html), [SelectFpr](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html), [SelectFromModel](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html), [SelectFwe](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFwe.html), [SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html), [SelectPercentile](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html), [SequentialFeatureSelector](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html), [VarianceThreshold](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html)) return the selected features (the mask of `get_support()`) unchanged, so the transformed features are identical (also negative zeros, missing and infinite values). The selection is part of the fit: the score functions (e.g. `f_classif` or `chi2`), the estimators of `SelectFromModel`, `RFE`, `RFECV` and `SequentialFeatureSelector` and the cross-validation aren't ported.
- `SelectFromModel` uses the importances of its estimator at the time of porting (also with `prefit=True` without a fit of the selector, like scikit-learn).
- A selector without any selected feature (scikit-learn returns no features with a warning) raises a `NotSupportedYetError`.
- Like `transform` of scikit-learn the ported selectors reject samples with missing or infinite values in any feature (also in a feature which isn't selected), unless the selector allows missing values (the tag `allow_nan`, e.g. of a `VarianceThreshold` or of a `SelectFromModel`, `RFE`, `RFECV` or `SequentialFeatureSelector` with an estimator which allows them, like a decision tree); then they return them unchanged. The ported selector aborts with an error (C and Go exit the program, the other languages throw an exception), so `make()` raises an `ExecutionFailed` error.
