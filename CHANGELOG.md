# Changelog

All notable changes to this project will be documented in this file.


## [1.0.0](https://github.com/nok/sklearn-porter/compare/v0.7.4...v1.0.0) (2026-10-17)


### ⚠ BREAKING CHANGES

* `predict` of the ported classifiers returns the label of the class instead of its index, the index is returned by `predict_index`. The return type of `predict` in C, Go and Java depends on the type of the labels.
* The JSON files of the exported tree estimators have a new format, so the model data has to be exported again.
* AdaBoostClassifier is ported with the SAMME algorithm. Serialized estimators of old scikit-learn versions can't be loaded.
* Python < 3.11 and scikit-learn < 1.8 are no longer supported.

### Features

* **c:** port AdaBoostClassifier to C ([#150](https://github.com/nok/sklearn-porter/issues/150)) ([c7f0dac](https://github.com/nok/sklearn-porter/commit/c7f0dacc02b7f829ce8f7e669f111402be970325))
* **c:** port BernoulliNB to C ([#129](https://github.com/nok/sklearn-porter/issues/129)) ([834bebc](https://github.com/nok/sklearn-porter/commit/834bebc7564bbe09dd313839fca48e78d1a74f1c))
* **c:** port GaussianNB to C ([#128](https://github.com/nok/sklearn-porter/issues/128)) ([554a97f](https://github.com/nok/sklearn-porter/commit/554a97fe9cdb3a1c6743d41c8466b2378a947a81))
* **c:** port KNeighborsClassifier to C ([#138](https://github.com/nok/sklearn-porter/issues/138)) ([875e697](https://github.com/nok/sklearn-porter/commit/875e6971665c8542f88a73cf90203e477aa7f5de))
* **c:** port LinearSVC to C with the exported template ([#139](https://github.com/nok/sklearn-porter/issues/139)) ([27f75df](https://github.com/nok/sklearn-porter/commit/27f75df2402f1fca3d35597df9f56c65f292c1c0))
* **c:** port MLPClassifier to C ([#133](https://github.com/nok/sklearn-porter/issues/133)) ([5d62b36](https://github.com/nok/sklearn-porter/commit/5d62b36b1c4dde8f5135e18ad76fe9a26f185c46))
* **c:** port MLPRegressor to C ([#151](https://github.com/nok/sklearn-porter/issues/151)) ([07ee1d7](https://github.com/nok/sklearn-porter/commit/07ee1d75b9d7a7fbf7007022d0748cd5bb9faa4f))
* **c:** port RandomForestClassifier and ExtraTreesClassifier to C ([#145](https://github.com/nok/sklearn-porter/issues/145)) ([ece3ba2](https://github.com/nok/sklearn-porter/commit/ece3ba206a52cb2cf58acd8b4ff97a2949b1f253))
* **c:** port SVC and NuSVC to C with the exported template ([#140](https://github.com/nok/sklearn-porter/issues/140)) ([e957fd4](https://github.com/nok/sklearn-porter/commit/e957fd40919bd0b553bc01c222c893b6fac826fc))
* **c:** port the DecisionTreeClassifier with the exported template ([#121](https://github.com/nok/sklearn-porter/issues/121)) ([f11f663](https://github.com/nok/sklearn-porter/commit/f11f6636f01082c1debb8bc02f18c81514bd0fb2))
* **go:** port AdaBoostClassifier to Go ([#146](https://github.com/nok/sklearn-porter/issues/146)) ([25ae6b2](https://github.com/nok/sklearn-porter/commit/25ae6b2ee7c0359f9336125f79c1ebe924bcb62b))
* **go:** port BernoulliNB to Go ([#126](https://github.com/nok/sklearn-porter/issues/126)) ([1a43e32](https://github.com/nok/sklearn-porter/commit/1a43e3215e364d2b935b25fea7e891de2a49834a))
* **go:** port GaussianNB to Go ([#122](https://github.com/nok/sklearn-porter/issues/122)) ([3f04718](https://github.com/nok/sklearn-porter/commit/3f047186d1cee16043d86fb05be4fea172ae2d19))
* **go:** port MLPClassifier to Go ([#130](https://github.com/nok/sklearn-porter/issues/130)) ([7afa171](https://github.com/nok/sklearn-porter/commit/7afa1719b3c6430f4c2e590eeb49f0ae7cc5955a))
* **go:** port MLPRegressor to Go ([#134](https://github.com/nok/sklearn-porter/issues/134)) ([520fc82](https://github.com/nok/sklearn-porter/commit/520fc82e829406ecb1a7ebe406a3c74d9cce74dc))
* **go:** port RandomForestClassifier and ExtraTreesClassifier to Go ([#141](https://github.com/nok/sklearn-porter/issues/141)) ([d1d3e31](https://github.com/nok/sklearn-porter/commit/d1d3e31794ea2054669dde25c4c2c7414ec7d2ec))
* **java:** port AdaBoostClassifier to Java ([#149](https://github.com/nok/sklearn-porter/issues/149)) ([ad1ed29](https://github.com/nok/sklearn-porter/commit/ad1ed294cccf4db31bd3d86389d27d801aa8c567))
* **java:** port MLPRegressor to Java ([#135](https://github.com/nok/sklearn-porter/issues/135)) ([6a267a8](https://github.com/nok/sklearn-porter/commit/6a267a8cb9132967b07d8b7b5ed58b5707930741))
* **java:** port RandomForestClassifier and ExtraTreesClassifier to Java with the attached template ([#142](https://github.com/nok/sklearn-porter/issues/142)) ([5a4deb2](https://github.com/nok/sklearn-porter/commit/5a4deb2dfeb9f1490e7279818551da1e0279342a))
* **php:** port AdaBoostClassifier to PHP ([#147](https://github.com/nok/sklearn-porter/issues/147)) ([c177d91](https://github.com/nok/sklearn-porter/commit/c177d915e0917eb50980b2112cae8590958c8364))
* **php:** port BernoulliNB to PHP ([#123](https://github.com/nok/sklearn-porter/issues/123)) ([7aa82cc](https://github.com/nok/sklearn-porter/commit/7aa82cc154fa88af22f6804e09dbdee458fa4809))
* **php:** port GaussianNB to PHP ([#124](https://github.com/nok/sklearn-porter/issues/124)) ([728ccea](https://github.com/nok/sklearn-porter/commit/728ccea9a7a7cfd00f7c540e370883f70e29e251))
* **php:** port MLPClassifier to PHP ([#131](https://github.com/nok/sklearn-porter/issues/131)) ([7273702](https://github.com/nok/sklearn-porter/commit/72737028951b71a4f6198e8abc505743647a9b63))
* **php:** port MLPRegressor to PHP ([#136](https://github.com/nok/sklearn-porter/issues/136)) ([927aca9](https://github.com/nok/sklearn-porter/commit/927aca9699b31f0bc4cfa5a93a469f7ffe41eee5))
* **php:** port RandomForestClassifier and ExtraTreesClassifier to PHP with the combined template ([#143](https://github.com/nok/sklearn-porter/issues/143)) ([4072596](https://github.com/nok/sklearn-porter/commit/407259699027643cabf7393a35d24df6f08d63e5))
* port DecisionTreeRegressor ([#153](https://github.com/nok/sklearn-porter/issues/153)) ([71fc099](https://github.com/nok/sklearn-porter/commit/71fc09970a497a4f475c23e893529d549d9df036))
* port KNeighborsClassifier with distance weights ([#189](https://github.com/nok/sklearn-porter/issues/189)) ([f7f2d73](https://github.com/nok/sklearn-porter/commit/f7f2d73de6fe17df2dd2158bf332367926cc90c0))
* port LinearRegression, Ridge, Lasso and ElasticNet ([#159](https://github.com/nok/sklearn-porter/issues/159)) ([8b8c239](https://github.com/nok/sklearn-porter/commit/8b8c239a7fc2f1d3d1f5a62264160a427052589c))
* port LogisticRegression ([#158](https://github.com/nok/sklearn-porter/issues/158)) ([4fa81e9](https://github.com/nok/sklearn-porter/commit/4fa81e9774f15dbd1f150eea085c821acbb83ed6))
* port RandomForestRegressor and ExtraTreesRegressor ([#154](https://github.com/nok/sklearn-porter/issues/154)) ([159d241](https://github.com/nok/sklearn-porter/commit/159d2412131fbc13aa62c0c8c9a6e0821d614595))
* port SGDClassifier, RidgeClassifier and Perceptron ([#160](https://github.com/nok/sklearn-porter/issues/160)) ([1e1356e](https://github.com/nok/sklearn-porter/commit/1e1356ee9266a55bfe7328f01fc2a3348df41e82))
* predict batches of samples via the standard input ([#190](https://github.com/nok/sklearn-porter/issues/190)) ([1da5a10](https://github.com/nok/sklearn-porter/commit/1da5a108b96ab58568eb728f4f08344c6bfdd1ef))
* return the class labels from the ported classifiers ([#187](https://github.com/nok/sklearn-porter/issues/187)) ([20d1fb8](https://github.com/nok/sklearn-porter/commit/20d1fb8675db98203afea0034d76d37f93a46165))
* **ruby:** port AdaBoostClassifier to Ruby ([#148](https://github.com/nok/sklearn-porter/issues/148)) ([235c27c](https://github.com/nok/sklearn-porter/commit/235c27cd18f85711856a9329c994cd6ab9363da2))
* **ruby:** port BernoulliNB to Ruby ([#127](https://github.com/nok/sklearn-porter/issues/127)) ([539736e](https://github.com/nok/sklearn-porter/commit/539736e943d9b021f6d5f357e02aad997adaacc1))
* **ruby:** port GaussianNB to Ruby ([#125](https://github.com/nok/sklearn-porter/issues/125)) ([db4492d](https://github.com/nok/sklearn-porter/commit/db4492deb6e14ef7a22032744414b8479067c945))
* **ruby:** port MLPClassifier to Ruby ([#132](https://github.com/nok/sklearn-porter/issues/132)) ([602d6bc](https://github.com/nok/sklearn-porter/commit/602d6bc76db3d41a7bbe31c39348d9f2371aeab5))
* **ruby:** port MLPRegressor to Ruby ([#137](https://github.com/nok/sklearn-porter/issues/137)) ([3cf0038](https://github.com/nok/sklearn-porter/commit/3cf00380a693e026cfc61900a44ba82a838e125b))
* **ruby:** port RandomForestClassifier and ExtraTreesClassifier to Ruby ([#144](https://github.com/nok/sklearn-porter/issues/144)) ([7a1beb5](https://github.com/nok/sklearn-porter/commit/7a1beb5d59a0fae68d699eb6eadcb6587a1581f7))
* store only the splits and leaf values of ported trees ([#186](https://github.com/nok/sklearn-porter/issues/186)) ([dfa8fcd](https://github.com/nok/sklearn-porter/commit/dfa8fcd233cb06333e5b39005c262c97b613833a))
* support multilabel RidgeClassifier and MLPClassifier ([#168](https://github.com/nok/sklearn-porter/issues/168)) ([fcb691f](https://github.com/nok/sklearn-porter/commit/fcb691fdb38d5f955f29d9e7404a6063000815cd))
* support scikit-learn &gt;= 1.8 ([#102](https://github.com/nok/sklearn-porter/issues/102)) ([4a62cc0](https://github.com/nok/sklearn-porter/commit/4a62cc027e8284864fe8c8bcaca40f7544ef626d))


### Bug Fixes

* apply the output activation exp of MLPRegressor with the loss poisson ([#177](https://github.com/nok/sklearn-porter/issues/177)) ([854ddd2](https://github.com/nok/sklearn-porter/commit/854ddd2d2bb89e072b0e67ae3ee12bcfce34d931))
* binarize the features of BernoulliNB with the threshold of the estimator ([#167](https://github.com/nok/sklearn-porter/issues/167)) ([472f5b1](https://github.com/nok/sklearn-porter/commit/472f5b164caf04f2000ac328404398e50823d586))
* compute the class probabilities of the ported Ruby decision trees in floating point ([#170](https://github.com/nok/sklearn-porter/issues/170)) ([2aa64cd](https://github.com/nok/sklearn-porter/commit/2aa64cd494dc03641676ea60c59541c75de4d8de))
* correct the exported MLPRegressor in JavaScript and test regressors ([#120](https://github.com/nok/sklearn-porter/issues/120)) ([0424b64](https://github.com/nok/sklearn-porter/commit/0424b643776635aee2da8b54f81316e5fd3961cd))
* correct the probabilities of the ported classifiers ([#118](https://github.com/nok/sklearn-porter/issues/118)) ([29698cc](https://github.com/nok/sklearn-porter/commit/29698cc17fcc0c3a7b54174e510bc710d3955690))
* decide the splits of ported trees like scikit-learn ([#119](https://github.com/nok/sklearn-porter/issues/119)) ([9cbf1d9](https://github.com/nok/sklearn-porter/commit/9cbf1d94bd1cfffb238a8a295c1b81f8e70b5678))
* declare the properties of the ported PHP estimators explicitly ([#165](https://github.com/nok/sklearn-porter/issues/165)) ([1d0b8cb](https://github.com/nok/sklearn-porter/commit/1d0b8cb6a1a2943bc92ceeb3b1abcb3cde114e02))
* derive the layer sizes of the ported MLPs from the fitted weights ([#178](https://github.com/nok/sklearn-porter/issues/178)) ([a987a8a](https://github.com/nok/sklearn-porter/commit/a987a8a26e5bb9d662e7b5e4588b851204f1d877))
* **docker:** use PYTHON_VERSION for the Python of the tests ([#115](https://github.com/nok/sklearn-porter/issues/115)) ([b10306d](https://github.com/nok/sklearn-porter/commit/b10306d3972f05093ec7709cbc22d4432ea34f07))
* import encoding/json in the exported LinearSVC for Go ([#191](https://github.com/nok/sklearn-porter/issues/191)) ([cc5b93c](https://github.com/nok/sklearn-porter/commit/cc5b93cded546bf3790fc5ded6e1ec67e3bea178))
* **php:** pass negative feature values to the ported estimator ([#104](https://github.com/nok/sklearn-porter/issues/104)) ([e1a9394](https://github.com/nok/sklearn-porter/commit/e1a9394e38cefe78141a264a40dd7fe20a8fd6a8))
* pick tied neighbors of the ported KNN in training order in all languages ([#180](https://github.com/nok/sklearn-porter/issues/180)) ([9f161ec](https://github.com/nok/sklearn-porter/commit/9f161ecf6036b56c8b2d64e3f2d961ee23b6223f))
* port estimators with sparse matrices as dense arrays ([#173](https://github.com/nok/sklearn-porter/issues/173)) ([52c3c89](https://github.com/nok/sklearn-porter/commit/52c3c8925a505ef467bb0ef47b337d55e30ab803))
* port infinite thresholds of trees and reject infinite BernoulliNB log probabilities ([#175](https://github.com/nok/sklearn-porter/issues/175)) ([c197fd7](https://github.com/nok/sklearn-porter/commit/c197fd7ff6f0f57b3efa617814abf8815af3f032))
* port multi-output DecisionTree, RandomForest, ExtraTrees and KNeighbors classifiers ([#169](https://github.com/nok/sklearn-porter/issues/169)) ([9d2e212](https://github.com/nok/sklearn-porter/commit/9d2e21288418c334e74cfce61d9f81fd24f19fcd))
* port support vector machines with double precision and without QualityWarning ([#163](https://github.com/nok/sklearn-porter/issues/163)) ([05a08e2](https://github.com/nok/sklearn-porter/commit/05a08e2470f06f59b738572f6914c265376e14fc))
* port the metric of KNeighborsClassifier and support chebyshev, cosine and hamming ([#166](https://github.com/nok/sklearn-porter/issues/166)) ([b0008b9](https://github.com/nok/sklearn-porter/commit/b0008b9bcfb214c7d3f97550d3266f16eb4dbc17))
* port the unrounded class fractions of weighted tree classifiers ([#171](https://github.com/nok/sklearn-porter/issues/171)) ([184de3c](https://github.com/nok/sklearn-porter/commit/184de3cb7a2ec573618725689f7ae111514b04ce))
* quote the paths in the shell commands of make() and test() ([#172](https://github.com/nok/sklearn-porter/issues/172)) ([72b4ca5](https://github.com/nok/sklearn-porter/commit/72b4ca5969874ad428e4b1c8b1321a12d6d11611))
* raise loading warnings in the CLI only while loading the estimator ([#162](https://github.com/nok/sklearn-porter/issues/162)) ([3350ee8](https://github.com/nok/sklearn-porter/commit/3350ee80d495c15c7f597da185022686aad7d4d3))
* reduce the memory usage of the ported Java estimators ([#185](https://github.com/nok/sklearn-porter/issues/185)) ([ba4028a](https://github.com/nok/sklearn-porter/commit/ba4028ac8ba4a5baee4d3088a9e692779a5b6c75))
* return new arrays from the ported MLPs in Java and JS ([#183](https://github.com/nok/sklearn-porter/issues/183)) ([c9417ce](https://github.com/nok/sklearn-porter/commit/c9417ce6522008b8c12ffad42a59dd18beb513f1))
* route missing values of the ported trees like scikit-learn ([#174](https://github.com/nok/sklearn-porter/issues/174)) ([b5972bf](https://github.com/nok/sklearn-porter/commit/b5972bfcc318d8e8d5147e3f7417149c51bdc7eb))
* run make() and test() without a shell and only warn on Windows ([#184](https://github.com/nok/sklearn-porter/issues/184)) ([0f763e2](https://github.com/nok/sklearn-porter/commit/0f763e2376232f8b8e5e6d42219a9ed0ff5ade39))
* run the ported estimators in threads and report failed executions ([#116](https://github.com/nok/sklearn-porter/issues/116)) ([f8bf14c](https://github.com/nok/sklearn-porter/commit/f8bf14ca4a4b0f64bc926fcb25c0530d71ce772e))
* **svc:** export gamma as number instead of string ([#103](https://github.com/nok/sklearn-porter/issues/103)) ([19f70e1](https://github.com/nok/sklearn-porter/commit/19f70e1b9499b63edbc03170a9b74338a8d6c403))
* use the fitted trees of the ported forest classifiers ([#179](https://github.com/nok/sklearn-porter/issues/179)) ([3967b91](https://github.com/nok/sklearn-porter/commit/3967b91928192d6d65e5ae5f41b38bbe700fd9e7))


### Documentation

* document the differences of KNeighborsClassifier with tied neighbors ([#176](https://github.com/nok/sklearn-porter/issues/176)) ([d891fb1](https://github.com/nok/sklearn-porter/commit/d891fb1a7a6cf00dbabd34209ad2e3e1248f92b9))
* improve README and move development docs to CONTRIBUTING.md ([#156](https://github.com/nok/sklearn-porter/issues/156)) ([7d94754](https://github.com/nok/sklearn-porter/commit/7d947545aac1a55cc5ecfa739a0b88474a4f8c2a))
* improve the wording and the versions in the readme ([#117](https://github.com/nok/sklearn-porter/issues/117)) ([4963775](https://github.com/nok/sklearn-porter/commit/4963775748a8de159f001ee85b2947e113301584))
* keep the save call of the readme usage example on one line ([#182](https://github.com/nok/sklearn-porter/issues/182)) ([1ab733c](https://github.com/nok/sklearn-porter/commit/1ab733c5e9e6c2b001992217f08a4016a1aaa784))
* move the notes, limitations and known issues of the readme to separate pages ([#181](https://github.com/nok/sklearn-porter/issues/181)) ([1257f4b](https://github.com/nok/sklearn-porter/commit/1257f4be4d21dd9c49817ff62db2097fe0cc2e0e))
* simplify and restructure README ([#161](https://github.com/nok/sklearn-porter/issues/161)) ([af1f5c5](https://github.com/nok/sklearn-porter/commit/af1f5c5101e2df9d3213297fa661c0290288f177))
* split README known issues into limitations and known issues ([#164](https://github.com/nok/sklearn-porter/issues/164)) ([a69aae5](https://github.com/nok/sklearn-porter/commit/a69aae5c6d0820ff8860412f560b7f78faeac327))
* update readme and add AGENTS.md ([#108](https://github.com/nok/sklearn-porter/issues/108)) ([e3c2bf5](https://github.com/nok/sklearn-porter/commit/e3c2bf5d5a785967bfe32753d64b851ec9415617))


### Build System

* **deps-dev:** update uv-build requirement to &gt;=0.12.24,&lt;0.14 ([d9f4336](https://github.com/nok/sklearn-porter/commit/d9f4336be886cf9699c43eb0eb08435cd9db2fe7))
* migrate from setup.py and conda to uv and pyproject.toml ([#101](https://github.com/nok/sklearn-porter/issues/101)) ([b9407c1](https://github.com/nok/sklearn-porter/commit/b9407c1dfa8b024b6112b8b6fcd9607c1143ff02))


### Continuous Integration

* add GitHub Actions workflows ([#106](https://github.com/nok/sklearn-porter/issues/106)) ([0d51dae](https://github.com/nok/sklearn-porter/commit/0d51dae5173cdd83c0227505fb8620375b2ca790))
* bump actions/checkout from 5 to 7 in the github-actions group ([#111](https://github.com/nok/sklearn-porter/issues/111)) ([ca28530](https://github.com/nok/sklearn-porter/commit/ca28530123e854a97fa5177e844369e11f8c21f3))
* release with release-please via TestPyPI to PyPI ([#107](https://github.com/nok/sklearn-porter/issues/107)) ([1976a9b](https://github.com/nok/sklearn-porter/commit/1976a9b10a48640c1a344cfa77030dc8476f5f0e))
* remove Travis CI ([#100](https://github.com/nok/sklearn-porter/issues/100)) ([0f6a7ca](https://github.com/nok/sklearn-porter/commit/0f6a7caf542f99f3179a236e6b38111439fb0baf))
* show build and CI changes in the changelog ([#114](https://github.com/nok/sklearn-porter/issues/114)) ([d59f19b](https://github.com/nok/sklearn-porter/commit/d59f19b9f262f3f631f05f12dccb55406234e04f))

## 0.7.3

### Fixed

- Fix argument name in generated method name ([#49](https://github.com/nok/sklearn-porter/commit/c8e7dbbf1d0ab660f4c3587d50111adf6a18b4e9))
- Fix parameter initialization of MLPClassifier/MLPRegressor ([#33](https://github.com/nok/sklearn-porter/commit/672e0188b47e3b7624c6b9cba2e66ef3dc9145a1))


## 0.7.2
## 0.7.1

### Fixed

- Fix installation issues with the centralised meta information and the build process
- Fix missed package and add `six` to the `requirements.txt` file


## 0.7.0

### Fixed

- Fix indices format in RandomForestClassifier ([#41](https://github.com/nok/sklearn-porter/pull/41), thanks [@apasanen](https://github.com/apasanen))

### Added

- Add Python 3.7 with Xenial to CI for testing
- Add [PyTest](https://docs.pytest.org) for extended testing (see [pytest.ini](pytest.ini))
- Add useful Makefile tasks with dependency handling (see [Makefile](Makefile)):
    - `install.environment` to install a `conda` environment
    - `install.requirements` to install all `pip` requirements
    - `make link` to install `porter` (cli) to the command line
    - `make open.examples` to start a local jupyter server
    - `make stop.examples` to stop the started jupyter server
    - `make test` to run all unittests in [tests](tests)
    - `make lint` to run pylint over [sklearn_porter](sklearn_porter)
    - `make jupytext` to generate the notebooks from Python sources

### Changed

- Core:
    - Merge all meta information to `sklearn_porter/pypi.json`
- CI and requirements:
    - Remove `conda` from CI
    - Clear and split `pip` requirements into three parts:
        - [requirements.txt](requirements.txt) for library requirements
        - [requirements.examples.txt](requirements.examples.txt) for examples requirements
        - [requirements.development.txt](requirements.development.txt) for development requirements
    - Update extended requriements (e.g. GCC, PHP, ...)
- Utils:
    - Refactor class [sklearn_porter.utils.Shell](sklearn_porter.utils.Shell) to remove redundancy
    - Refactor CLI with a bunch of improvements (grouped arguments, use of console scripts)
- Testing:
    - Change web server port for testing from `8080` to `8713` (because of collusion with the default port of Jenkins)
    - Remove class [tests.utils.DependencyChecker](https://github.com/nok/sklearn-porter/blob/release/0.6.2/tests/utils/DependencyChecker.py), instead use [sklearn_porter.utils.Environment](https://github.com/nok/sklearn-porter/blob/release/0.7.0/sklearn_porter/utils/Environment.py)
    - Remove class [tests.utils.Timer](https://github.com/nok/sklearn-porter/blob/release/0.6.2/tests/utils/Timer.py), instead use `--durations=0` of PyTest


## 0.6.2

### Fixed

- Fix getter of transpiled estimator and remove filename manipulations ([#b5efe78](https://github.com/nok/sklearn-porter/commit/b5efe78de9dd721f9135f65c525a3e8ab8b06f79))
- Fix and add binary to string conversion in Python 3.x ([#bdcfb1f](https://github.com/nok/sklearn-porter/commit/bdcfb1ff4c076485a5fc3b00beaf81becec0717b))
- Fix and close open template file(s) ([#b5efe78](https://github.com/nok/sklearn-porter/commit/b5efe78de9dd721f9135f65c525a3e8ab8b06f79))


## 0.6.1

### Added

- Add new estimator:
    - Go:
        - `ensemble.RandomForestClassifier` ([PR #24](https://github.com/nok/sklearn-porter/pull/24) by [tpotega](https://github.com/tpotega))

### Fixed

- Fix wrong method signature of the DecisionTreeClassifier in C ([Issue #23](https://github.com/nok/sklearn-porter/issues/23) by [sanjivsoni17](https://github.com/sanjivsoni17))


## 0.6.0

### Added

- Add [changelog.md](changelog.md) for the next releases ([#ee44ac9](https://github.com/nok/sklearn-porter/commit/ee44ac92618bf48e3aff6fbb65591b6f87c88826)).
- Add [changelog.md](changelog.md) and [readme.md](readme.md) to each build of a release ([#ee44ac9](https://github.com/nok/sklearn-porter/commit/ee44ac92618bf48e3aff6fbb65591b6f87c88826)).
- Add for each target programming language a new new command-line argument (e.g. `--java`, `--c` or `--go`) ([#41b93a0](https://github.com/nok/sklearn-porter/commit/41b93a0bff44dd045e711a08a53fe8c75d8d460a)).
- Add argument `--class_name` and `--method_name` to define the class and method name in the final output directly ([#6f2a1d9](https://github.com/nok/sklearn-porter/commit/6f2a1d97b5cddb6232a4fcf0d469cf167a019fdf)).
- Add pipe functionality and the related command-line argument (`--pipe` or `-p`) ([#8a57746](https://github.com/nok/sklearn-porter/commit/8a57746e4e97b137032fa7401e37792d496c0aa2)).
- Add `--export` argument to dump the model data and use the specific templates ([#0669645](https://github.com/nok/sklearn-porter/commit/0669645acdfa8cbe39a8446f95f671c580cc2026)).
- Add `--checksum` argument to append the computed md5 checksum at the end of the dumped model data file ([#cd12827](https://github.com/nok/sklearn-porter/commit/cd12827cb136d0711c555fe321f72d6aa8326143)).
- Add `--data` argument to kust export the model data ([#fad499a](https://github.com/nok/sklearn-porter/commit/fad499a4448f55d8c344f1088cca54a5c267c3e9)).
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
- Add support to export and import model data:
    - Java:
        - `svm.SVC` [notebook](examples/estimator/classifier/SVC/java/basics_imported.ipynb) ([#e753252](https://github.com/nok/sklearn-porter/commit/e75325255a91bb9ca00a4e34c5393c6b05a41d6f))
        - `svm.NuSVC` [notebook](examples/estimator/classifier/NuSVC/java/basics_imported.ipynb) ([#e753252](https://github.com/nok/sklearn-porter/commit/e75325255a91bb9ca00a4e34c5393c6b05a41d6f))
        - `svm.LinearSVC` [notebook](examples/estimator/classifier/LinearSVC/java/basics_imported.ipynb) ([#562ed8c](https://github.com/nok/sklearn-porter/commit/562ed8c77cc3b9e1fa3f9b5219ec842c290219e6))
        - `ensemble.AdaBoostClassifier` [notebook](examples/estimator/classifier/AdaBoostClassifier/java/basics_imported.ipynb) ([#79d846f](https://github.com/nok/sklearn-porter/commit/79d846f71af4a37de4655d86ab5e42d11311d5ed))
        - `ensemble.RandomForestClassifier` [notebook](examples/estimator/classifier/RandomForestClassifier/java/basics_imported.ipynb) ([#9979a94](https://github.com/nok/sklearn-porter/commit/9979a948f51fef527b792babe06103c2c0dd5db4))
        - `ensemble.ExtraTreesClassifier` [notebook](examples/estimator/classifier/ExtraTreesClassifier/java/basics_imported.ipynb) ([#9979a94](https://github.com/nok/sklearn-porter/commit/9979a948f51fef527b792babe06103c2c0dd5db4))
        - `tree.DecisionTreeClassifier` [notebook](examples/estimator/classifier/DecisionTreeClassifier/java/basics_imported.ipynb) ([#9979a94](https://github.com/nok/sklearn-porter/commit/9979a948f51fef527b792babe06103c2c0dd5db4))
        - `neighbors.KNeighborsClassifier` [notebook](examples/estimator/classifier/KNeighborsClassifier/java/basics_imported.ipynb) ([#1b81d3f](https://github.com/nok/sklearn-porter/commit/1b81d3f0e9cbe52203bbfde4cf73df17eacf4fdd#diff-86bc174b477eabd59f582e0e2882dcb3R54))
        - `naive_bayes.GaussianNB` [notebook](examples/estimator/classifier/GaussianNB/java/basics_imported.ipynb) ([#b68c5df](https://github.com/nok/sklearn-porter/commit/b68c5df241fec0b654d919066dca0663d909bde1))
        - `naive_bayes.BernoulliNB` [notebook](examples/estimator/classifier/BernoulliNB/java/basics_imported.ipynb) ([#7eae57d](https://github.com/nok/sklearn-porter/commit/7eae57df5b64a11cfc520e057649d711022653cd))
        - `neural_network.MLPClassifier` [notebook](examples/estimator/classifier/MLPClassifier/java/basics_imported.ipynb) ([#b988f57](https://github.com/nok/sklearn-porter/commit/b988f575db10133d41b1c80773bfe814b7271900), [#d8ff774](https://github.com/nok/sklearn-porter/commit/d8ff77400c20ce41e318403bb12d2d65eeb3d309))
    - JavaScript:
        - `ensemble.AdaBoostClassifier` [notebook](examples/estimator/classifier/AdaBoostClassifier/js/basics_imported.ipynb) ([#9fdcc49](https://github.com/nok/sklearn-porter/commit/9fdcc49355617668c3bf451a8d5828880245a4ae), [#dd70ac7](https://github.com/nok/sklearn-porter/commit/dd70ac7e63cecad9028e9dea32efb7e1d2e43237), [#2401fba](https://github.com/nok/sklearn-porter/commit/2401fba9938880bf420de0253e489c22e7b040e4))
        - `neighbors.KNeighborsClassifier` [notebook](examples/estimator/classifier/KNeighborsClassifier/js/basics_imported.ipynb) ([#f915b70](https://github.com/nok/sklearn-porter/commit/f915b70568b4212e7fb8d616058a927f98b186da))
        - `neural_network.MLPClassifier` [notebook](examples/estimator/classifier/MLPClassifier/js/basics_imported.ipynb) ([#ce23148](https://github.com/nok/sklearn-porter/commit/ce2314893d013a1c4ac2fd9965d1d61fe324834c))
- Add new recipes:
    - JavaScript:
        - [MLPClassifier Web Workers](examples/recipes/use_javascript_web_workers) ([#187efac](https://github.com/nok/sklearn-porter/commit/187efac3fa045e177a1980244bef302a462fcf4e))

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
- Fix dynamic class name in the `MLPClassifier` template ([#b988f57](https://github.com/nok/sklearn-porter/commit/b988f575db10133d41b1c80773bfe814b7271900#diff-9eeccb9a3d9e520cb41df490846b7c88L22))
