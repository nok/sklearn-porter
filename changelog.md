# Changelog

All notable changes to this project will be documented in this file.


## [Unreleased]

### Added

- Add [changelog.md](changelog.md) for the next releases ([#ee44ac9](https://github.com/nok/sklearn-porter/commit/ee44ac92618bf48e3aff6fbb65591b6f87c88826)).
- Add [changelog.md](changelog.md) and [readme.md](readme.md) to each build of a release ([#ee44ac9](https://github.com/nok/sklearn-porter/commit/ee44ac92618bf48e3aff6fbb65591b6f87c88826)).
- Add for each target programming language a new new command-line argument (e.g. `--java`, `--c` or `--go`) ([#41b93a0](https://github.com/nok/sklearn-porter/commit/41b93a0bff44dd045e711a08a53fe8c75d8d460a)).
- Add argument `--class_name` and `--method_name` to define the class and method name in the final output directly ([#6f2a1d9](https://github.com/nok/sklearn-porter/commit/6f2a1d97b5cddb6232a4fcf0d469cf167a019fdf)). 
- Add pipe functionality and the related command-line argument (`--pipe` or `-p`) ([#8a57746](https://github.com/nok/sklearn-porter/commit/8a57746e4e97b137032fa7401e37792d496c0aa2)).
- Add test class `Go` in `tests/language/Go.py` to test all implementations for the target programming language Go ([#1d0b5d6](https://github.com/nok/sklearn-porter/commit/1d0b5d6a2bf1a5604ae283cc728e3a83fb17a6ea)).
- Add Go compiling (`go build -o brain brain.go`) and execution (`./brain`) command ([#5d24f57](https://github.com/nok/sklearn-porter/commit/5d24f57ec50e9935dac8389e243deda7b09659d7)).
- Add initial Web Workers features in JavaScript templates ([#87d3236](https://github.com/nok/sklearn-porter/commit/87d32365d06ba01cce7667b03f9a4265a1312dad)) and a create seperate example ([#187efac](https://github.com/nok/sklearn-porter/commit/187efac3fa045e177a1980244bef302a462fcf4e)). 
- Add the feature to read the estimator from a used [Pipeline](http://scikit-learn.org/stable/modules/generated/pipeline.Pipeline.html) ([#b92edff](https://github.com/nok/sklearn-porter/commit/b92edfff278a997d03f6bca65ea99d0bd02f8ba3), issue: [#18](https://github.com/nok/sklearn-porter/issues/18)).
- Add a new class argument (`num_format=lambda x: str(x)`) to change the default representation of floating-point values ([#7f9fac8](https://github.com/nok/sklearn-porter/commit/7f9fac8eb35371e9374b4cf73519f83dbcb66632)).
- Use estimator name as default class name (e.g. `MLPClasifier`, `KNeighborsClassifier`, `SVC`, ...) ([#710a854](https://github.com/nok/sklearn-porter/commit/710a854072bf19054cc2c46eff661241ffa92d65)). 
- Add new estimator:
    - Go:
        - `tree.DecisionTreeClassifier` ([#fe59710](https://github.com/nok/sklearn-porter/commit/fe59710a72c6a4bf5fb1d0acc0a35eba3dda950e))
    - JavaScript:
        - `naive_bayes.BernoulliNB` ([#9784d6b](https://github.com/nok/sklearn-porter/commit/9784d6b8752fbb15b57345a5a08138618e3b676e))
    - PHP:
        - `ensemble.RandomForestClassifier` ([#faac38d](https://github.com/nok/sklearn-porter/commit/faac38d60f04c40641935b25c4b6dce33e96b4ac))
        - `ensemble.ExtraTreesClassifier` ([#2a29321](https://github.com/nok/sklearn-porter/commit/2a2932114e9313ae1e54b9369adcae00a4cce813))
    - Ruby:
        - `svm.SVC` ([#3ef1646](https://github.com/nok/sklearn-porter/commit/3ef16464515e539e2c4bd6dd718e9d097e95e131))
        - `svm.NuSVC` ([#0a39aaf](https://github.com/nok/sklearn-porter/commit/0a39aaf9349830130f92c09a8e9af77fed5bacac))
        - `tree.DecisionTreeClassifier` ([#a404c4f](https://github.com/nok/sklearn-porter/commit/a404c4f383a62d98ac543c617234c0a907b8267a))
        - `ensemble.RandomForestClassifier` ([#3775501](https://github.com/nok/sklearn-porter/commit/3775501b77436c0b5b5132e11893d0c4add0cb7b))
        - `ensemble.ExtraTreesClassifier` ([#81b9914](https://github.com/nok/sklearn-porter/commit/81b99149116f00a790e0df33d60e381cafc89bf2))

### Changed
 
- Use human-readable placeholders (e.g. `'{class_name}.{method_name}'`) instead of index-based placeholders (e.g. `'{0}.{1}'`) in all main templates of all estimators ([#de02795](https://github.com/nok/sklearn-porter/commit/de02795f3628ccad9d5e85940d37b866e2e7443e)).
- Change the order of optional and required arguments in the `--help` text ([#54d9973](https://github.com/nok/sklearn-porter/commit/54d99736f5fe144350e990621ba4d145776eecdd)).
- Change the default representation of floating-point values from `repr(x)` to `str(x)` ([#7f9fac8](https://github.com/nok/sklearn-porter/commit/7f9fac8eb35371e9374b4cf73519f83dbcb66632)).
- Use the method name `integrity_score(X)` instead of `predict_test(X)` to avoid misconceptions for the integrity test ([#715ec7d](https://github.com/nok/sklearn-porter/commit/715ec7dee0e2d98cb2917d48a2522683240d084a)). 
- Separate the model data from the algorithm:
    - `tree.DecisionTreeClassifier` ([#f669aab](https://github.com/nok/sklearn-porter/commit/f669aab7e15971ea2071c5f9df096b924ae0dbcf), [#bba6296](https://github.com/nok/sklearn-porter/commit/bba629602d46780467efbc0e8f74d7880131593b), [#b727186](https://github.com/nok/sklearn-porter/commit/b7271867c755f3372886b07b76d763f2f2911eff), [#e2740fd](https://github.com/nok/sklearn-porter/commit/e2740fd07f43c02f3514b3834a765d43c640efaa), [#5c9da8a](https://github.com/nok/sklearn-porter/commit/5c9da8a58ec2143398444bd3afcc16806dfdc86b))
    - `neighbors.KNeighborsClassifier` ([#59a0e91](https://github.com/nok/sklearn-porter/commit/59a0e9114daeeb7d81a975c3adfa0ad27be3a426), [#1ac5d8a](https://github.com/nok/sklearn-porter/commit/29412ab55d8ebcdb7914974121c03d64660e5f94))
    - `neural_network.MLPClassifier` ([#635da46](https://github.com/nok/sklearn-porter/commit/635da46dbf29a80d51a16f3bbc28a5ba87eacdd7), [#7d31668](https://github.com/nok/sklearn-porter/commit/7d3166894229f70aafe6a6c9e2e7dbd091589c15), [#78296e2](https://github.com/nok/sklearn-porter/commit/78296e2d893d882240ebb8f54ada07d28ab9fc49), [#4cdcfde](https://github.com/nok/sklearn-porter/commit/4cdcfde6a34e131b8ab7088af880eb081fd8f3dd), [#7820508](https://github.com/nok/sklearn-porter/commit/7820508aad7f1ccf39529023c22b3427471bde68), [#7820508](https://github.com/nok/sklearn-porter/commit/7820508aad7f1ccf39529023c22b3427471bde68))
    - `neural_network.MLPRegressor` ([#60d9d42](https://github.com/nok/sklearn-porter/commit/60d9d42a0fd7860097f37dd3be5808b8be136cda), [#e4a8169](https://github.com/nok/sklearn-porter/commit/e4a8169d8cd1a5ecbb0821e792fcbfd932364fd5))
    - `naive_bayes.GaussianNB` ([#1ac5d8a](https://github.com/nok/sklearn-porter/commit/1ac5d8a3e5137e7d308c8c0f6529ae4c70a54abe))
    - `naive_bayes.BernoulliNB` ([#ff82bb8](https://github.com/nok/sklearn-porter/commit/ff82bb880ce4ae95af0f95e90bc3e681e4f261b8), [#3c57a06](https://github.com/nok/sklearn-porter/commit/3c57a06a733cdd8e9a74cb41c4087064161ad0d5))
    - `svm.SVC` and `svm.NuSVC` ([#4745d8b](https://github.com/nok/sklearn-porter/commit/4745d8b0dd09addf7b6e6affba8954b4d7da6ecb), [#5f77e4d](https://github.com/nok/sklearn-porter/commit/5f77e4dba1ce4f84478ada2652227922471a4d9f), [#59831da](https://github.com/nok/sklearn-porter/commit/59831dab24d4f6f43daec61ae277139ed1bf921c), [#cd8f52e](https://github.com/nok/sklearn-porter/commit/cd8f52e33cad7c1e909b858333a9132e4b03a4a7), [#c483d25](https://github.com/nok/sklearn-porter/commit/c483d259dc4fb1b8beada4ef9c7c11f4b1d5aff6))
    - `svm.LinearSVC` ([#bb617c7](https://github.com/nok/sklearn-porter/commit/bb617c741ea80dde8da97121ec253fd3ee8f4810))
 
### Removed

- Hide the command-line argument `--language` and `-l` for the choice of the target programming language ([#fc14a3b](https://github.com/nok/sklearn-porter/commit/fc14a3b55d6319d3940c9c11d168b015b972f96d)). 

### Fixed

- Fix inaccuracies in `neural_network.MLPRegressor` and `neural_network.MLPClassifier` occurred by the transpiled tanh and identity activation functions ([#6696410](https://github.com/nok/sklearn-porter/commit/66964103083d04eedbd51cd83487808d43073350)).
- Fix installation problems with pip and Python 3 ([#2935828](https://github.com/nok/sklearn-porter/commit/2935828735fb1a8141c32f5f772172c12877c42d), issue: [#17](https://github.com/nok/sklearn-porter/issues/17))