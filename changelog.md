# Changelog

All notable changes to this project will be documented in this file.


## [Unreleased]

### Added

- Add [changelog.md](changelog.md) for the next releases ([#ee44ac9](https://github.com/nok/sklearn-porter/commit/ee44ac92618bf48e3aff6fbb65591b6f87c88826)).
- Add [changelog.md](changelog.md) and [readme.md](readme.md) to each build of a release ([#ee44ac9](https://github.com/nok/sklearn-porter/commit/ee44ac92618bf48e3aff6fbb65591b6f87c88826)).
- Add for each target programming language a new new command-line argument (e.g. `--java`, `--c` or `--go`) ([#41b93a0](https://github.com/nok/sklearn-porter/commit/41b93a0bff44dd045e711a08a53fe8c75d8d460a)).
- Add pipe functionality and the related command-line argument (`--pipe` or `-p`) ([#8a57746](https://github.com/nok/sklearn-porter/commit/8a57746e4e97b137032fa7401e37792d496c0aa2)).
- Add test class `Go` in `tests/language/Go.py` to test all implementations for the target programming language Go ([#1d0b5d6](https://github.com/nok/sklearn-porter/commit/1d0b5d6a2bf1a5604ae283cc728e3a83fb17a6ea)). 
- Add possibility to read the estimator from a used [Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) ([#b92edff](https://github.com/nok/sklearn-porter/commit/b92edfff278a997d03f6bca65ea99d0bd02f8ba3), issue: [#18](https://github.com/nok/sklearn-porter/issues/18)).
- Add new estimator:
    - Go:
        - sklearn.tree.DecisionTreeClassifier ([#fe59710](https://github.com/nok/sklearn-porter/commit/fe59710a72c6a4bf5fb1d0acc0a35eba3dda950e))
    - JavaScript:
        - sklearn.naive_bayes.BernoulliNB ([#9784d6b](https://github.com/nok/sklearn-porter/commit/9784d6b8752fbb15b57345a5a08138618e3b676e))
    - PHP:
        - sklearn.ensemble.RandomForestClassifier ([#faac38d](https://github.com/nok/sklearn-porter/commit/faac38d60f04c40641935b25c4b6dce33e96b4ac))
        - sklearn.ensemble.ExtraTreesClassifier ([#2a29321](https://github.com/nok/sklearn-porter/commit/2a2932114e9313ae1e54b9369adcae00a4cce813))
    - Ruby:
        - sklearn.svm.SVC ([#3ef1646](https://github.com/nok/sklearn-porter/commit/3ef16464515e539e2c4bd6dd718e9d097e95e131))
        - sklearn.svm.NuSVC ([#0a39aaf](https://github.com/nok/sklearn-porter/commit/0a39aaf9349830130f92c09a8e9af77fed5bacac))
        - sklearn.tree.DecisionTreeClassifier ([#a404c4f](https://github.com/nok/sklearn-porter/commit/a404c4f383a62d98ac543c617234c0a907b8267a))
        - sklearn.ensemble.RandomForestClassifier ([#3775501](https://github.com/nok/sklearn-porter/commit/3775501b77436c0b5b5132e11893d0c4add0cb7b))
        - sklearn.ensemble.ExtraTreesClassifier ([#81b9914](https://github.com/nok/sklearn-porter/commit/81b99149116f00a790e0df33d60e381cafc89bf2))

### Changed
 
- Use human-readable placeholders (e.g. `'{class_name}.{method_name}'`) instead of index-based placeholders (e.g. `'{0}.{1}'`) in all main templates of all estimators ([#de02795](https://github.com/nok/sklearn-porter/commit/de02795f3628ccad9d5e85940d37b866e2e7443e)).
 
### Removed

- Hide the command-line argument `--language` and `-l` for the choice of the target programming language ([#fc14a3b](https://github.com/nok/sklearn-porter/commit/fc14a3b55d6319d3940c9c11d168b015b972f96d)). 

### Fixed

- Fix inaccuracies in sklearn.neural_network.MLPRegressor and sklearn.neural_network.MLPClassifier occurred by the transpiled tanh and identity activation functions ([#6696410](https://github.com/nok/sklearn-porter/commit/66964103083d04eedbd51cd83487808d43073350)).
- Fix installation problems with pip and Python 3 ([#2935828](https://github.com/nok/sklearn-porter/commit/2935828735fb1a8141c32f5f772172c12877c42d), issue: [#17](https://github.com/nok/sklearn-porter/issues/17))