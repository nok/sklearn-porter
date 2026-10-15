# Known Issues

- The ported classifiers return the index of the predicted class instead of its label. The predictions only match those of scikit-learn if the labels are `0, 1, …, n_classes - 1`, so encode other labels (e.g. `1..3`, `-1/+1` or strings) with a [`LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) before training.
