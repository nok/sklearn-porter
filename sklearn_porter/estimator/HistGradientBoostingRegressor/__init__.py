from collections.abc import Callable
from textwrap import indent

import numpy as np
from jinja2 import Environment
from loguru import logger as L

# scikit-learn
from sklearn.ensemble import (
    HistGradientBoostingClassifier as HistGradientBoostingClassifierClass,
)
from sklearn.ensemble import (
    HistGradientBoostingRegressor as HistGradientBoostingRegressorClass,
)

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorBase import EstimatorBase
from sklearn_porter.estimator.GradientBoostingRegressor import (
    GradientBoostingRegressor,
)


class CategoricalSplit(float):
    """
    The threshold of a categorical split of a tree of the `combined` template.

    The `combined` template compiles a categorical split into a comparison
    with its categories (see `HistGradientBoostingRegressor._create_branch`).
    The threshold is the float `0.0` for the checks of all thresholds (e.g.
    for infinite thresholds) and stores the categories of the split.

    Parameters
    ----------
    categories : list of float
        The sorted categories which don't go to the child of the missing
        values (NaN) and unknown categories.
    missing_go_to_left : bool
        Whether the missing values and the unknown categories go to the left
        child.
    """

    categories: list
    missing_go_to_left: bool

    def __new__(cls, categories: list, missing_go_to_left: bool):
        split = super().__new__(cls, 0.0)
        split.categories = categories
        split.missing_go_to_left = missing_go_to_left
        return split


def _in_bitset(bitset: np.ndarray, value: int) -> bool:
    """Check whether a value is in a bitset of scikit-learn (32-bit words)."""
    return bool((int(bitset[value // 32]) >> (value % 32)) & 1)


def _get_categories(estimator) -> tuple[np.ndarray, dict]:
    """
    Get the original columns of the features and the categories of the
    categorical features of a fitted estimator.

    scikit-learn encodes the categorical features with an `OrdinalEncoder`
    and moves them to the front of the features (`_preprocessor`), so the
    trees refer to the features in this order and to the categories by
    their indices.

    Parameters
    ----------
    estimator : HistGradientBoostingClassifier or HistGradientBoostingRegressor
        The fitted estimator.

    Returns
    -------
    The original column of each feature of the trees and the categories
    of each categorical feature of the trees.

    Raises
    ------
    NotSupportedYetError
        If a categorical feature has categories which aren't finite numbers.
    """
    est = estimator  # alias
    n_features = est.n_features_in_
    is_categorical = est.is_categorical_
    if is_categorical is None:
        return np.arange(n_features), {}
    is_categorical = np.asarray(is_categorical, dtype=bool)
    columns = np.concatenate(
        (np.flatnonzero(is_categorical), np.flatnonzero(~is_categorical))
    )
    encoder = est._preprocessor.named_transformers_['encoder']
    categories = {}
    for feature, values in enumerate(encoder.categories_):
        try:
            values = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            msg = (
                'The categories of a categorical feature must be numbers, '
                'because the ported estimators only support numeric features.'
            )
            raise exception.NotSupportedYetError(msg) from None
        # The encoder puts the missing values (NaN) at the end:
        if len(values) and np.isnan(values[-1]):
            values = values[:-1]
        if not np.isfinite(values).all():
            msg = 'The categories of a categorical feature must be finite.'
            raise exception.NotSupportedYetError(msg)
        categories[feature] = values
    return columns, categories


def _get_tree_data(
    predictor,
    columns: np.ndarray,
    categories: dict,
    known_bitsets: np.ndarray,
    known_indices: np.ndarray,
) -> tuple[dict, dict]:
    """
    Get the model data of a fitted tree (`TreePredictor`) in the compact
    format of the other trees (see `sklearn_porter.utils.get_tree_data()`).

    The trees compare the raw features with the thresholds (`x <= t`) in
    double precision like scikit-learn. A categorical split sends a sample
    to the child of the missing values (NaN), unless its category is one
    of the categories of the other child. The templates `attached` and
    `exported` replace a categorical split by a chain of numeric splits,
    which checks these categories one after another: `x <= c'` and `x <= c`,
    where `c'` is the next smaller double of the category `c`, are true for
    `x == c` only. The template `combined` compares the features with the
    categories directly (see `CategoricalSplit`).

    Parameters
    ----------
    predictor : TreePredictor
        The fitted tree.
    columns : np.ndarray
        The original column of each feature of the tree.
    categories : dict
        The categories of each categorical feature of the tree.
    known_bitsets : np.ndarray
        The known categories of the categorical features (bitsets).
    known_indices : np.ndarray
        The index of the bitset of each feature.

    Returns
    -------
    The model data of the tree for the templates `attached` and `exported`
    and for the template `combined`.
    """
    nodes = predictor.nodes
    is_leaf = nodes['is_leaf'].astype(bool)
    values = nodes['value'][is_leaf].astype(float).tolist()
    if is_leaf.all():  # a single leaf
        tree = dict(
            lefts=[-1],
            rights=[-1],
            thresholds=[0.0],
            indices=[0],
            missing_go_to_left=[0],
            values=values,
        )
        return tree, tree

    # The references of all nodes, splits by `i` and leaves by `-i - 1`:
    refs = np.where(is_leaf, -np.cumsum(is_leaf), np.cumsum(~is_leaf) - 1)
    splits = nodes[~is_leaf]
    tree = dict(
        lefts=refs[splits['left']].tolist(),
        rights=refs[splits['right']].tolist(),
        thresholds=splits['num_threshold'].astype(float).tolist(),
        indices=columns[splits['feature_idx']].tolist(),
        missing_go_to_left=splits['missing_go_to_left'].astype(int).tolist(),
        values=values,
    )
    combined = {key: list(value) for key, value in tree.items()}

    for i, split in enumerate(splits):
        if not split['is_categorical']:
            continue
        feature = int(split['feature_idx'])
        if split['bitset_idx'] >= len(predictor.raw_left_cat_bitsets):
            msg = 'The categorical split of a tree has no categories.'
            raise exception.NotSupportedYetError(msg)
        # The same checks as scikit-learn, missing values (NaN) and unknown
        # categories go to the child of the missing values:
        left_bitset = predictor.raw_left_cat_bitsets[split['bitset_idx']]
        known_bitset = known_bitsets[known_indices[feature]]
        lefts, rights = [], []
        for code, category in enumerate(categories[feature].tolist()):
            if _in_bitset(left_bitset, code):
                lefts.append(category)
            elif _in_bitset(known_bitset, code):
                rights.append(category)
        missing_go_to_left = bool(split['missing_go_to_left'])
        if missing_go_to_left:
            others, child, missing = rights, tree['rights'][i], tree['lefts'][i]
        else:
            others, child, missing = lefts, tree['lefts'][i], tree['rights'][i]
        others = sorted(others)

        # Without other categories all samples go to the child of the
        # missing values, so both children of the split are this child:
        if not others:
            for data in (tree, combined):
                data['lefts'][i] = data['rights'][i] = missing
                data['thresholds'][i] = 0.0
                data['missing_go_to_left'][i] = 0
            continue
        combined['thresholds'][i] = CategoricalSplit(others, missing_go_to_left)
        combined['missing_go_to_left'][i] = 0

        # The chain of splits, which checks the categories one after another.
        # It starts with the split itself and appends the other splits:
        start = len(tree['lefts'])
        chain = [i, *range(start, start + 2 * len(others) - 1), missing]
        for k, category in enumerate(others):
            below, equal, following = chain[2 * k : 2 * k + 3]
            below_category = float(np.nextafter(category, -np.inf))
            column = int(columns[feature])
            _set_split(tree, below, missing, equal, below_category, column)
            _set_split(tree, equal, child, following, category, column)
    return tree, combined


def _set_split(
    tree: dict, node: int, left: int, right: int, threshold: float, index: int
):
    """Set (or append) a numeric split of a tree in the compact format."""
    split = dict(
        lefts=left,
        rights=right,
        thresholds=threshold,
        indices=index,
        # All comparisons with missing values (NaN) are false, so they go to
        # the right child and finally to the child of the missing values:
        missing_go_to_left=0,
    )
    for key, value in split.items():
        if node == len(tree[key]):
            tree[key].append(value)
        else:
            tree[key][node] = value


def get_hist_trees(
    estimator: HistGradientBoostingClassifierClass
    | HistGradientBoostingRegressorClass,
) -> tuple[list, list, list]:
    """
    Get the model data of the trees and the initial raw predictions of a
    fitted histogram-based gradient boosting estimator.

    scikit-learn starts with the baseline prediction and adds the values of
    the leaves of all trees iteration by iteration (with one tree per class
    of multiclass classifiers). The values of the leaves already contain
    the learning rate.

    Parameters
    ----------
    estimator : HistGradientBoostingClassifier or HistGradientBoostingRegressor
        The fitted estimator.

    Returns
    -------
    The model data of the trees for the templates `attached` and `exported`,
    the model data of the trees for the template `combined` and the initial
    raw predictions (one per tree of an iteration).
    """
    est = estimator  # alias
    columns, categories = _get_categories(est)
    known_bitsets, known_indices = (
        est._bin_mapper.make_known_categories_bitsets()
    )
    trees, combined = [], []
    for predictors in est._predictors:
        for predictor in predictors:
            tree, tree_combined = _get_tree_data(
                predictor, columns, categories, known_bitsets, known_indices
            )
            trees.append(tree)
            combined.append(tree_combined)
    inits = np.ravel(est._baseline_prediction).astype(float).tolist()
    return trees, combined, inits


class HistGradientBoostingRegressor(GradientBoostingRegressor, EstimatorBase):
    """Extract model data and port a HistGradientBoostingRegressor regressor."""

    SKLEARN_URL = 'sklearn.ensemble.HistGradientBoostingRegressor.html'

    def __init__(self, estimator: HistGradientBoostingRegressorClass):
        EstimatorBase.__init__(self, estimator)
        L.info(f'Create specific estimator `{self.estimator_name}`.')
        est = estimator  # alias

        # Is the estimator fitted?
        if not getattr(est, '_predictors', None):
            raise exception.NotFittedEstimatorError(self.estimator_name)

        trees, self._combined_trees, inits = get_hist_trees(est)
        self.meta_info = dict(
            n_estimators=len(trees),
            n_features=est.n_features_in_,
            n_outputs=1,  # the number of values of each leaf
            # The losses `poisson` and `gamma` predict the exponential of
            # the raw predictions (log link):
            is_log_link=est.loss in ('poisson', 'gamma'),
        )
        L.info(f'Meta info (keys): {self.meta_info.keys()}')
        L.opt(lazy=True).debug(f'Meta info: {self.meta_info}')

        self.model_data = dict(estimators=trees, inits=inits)
        L.info(f'Model data (keys): {self.model_data.keys()}')
        L.opt(lazy=True).debug(f'Model data: {self.model_data}')

    def _get_trees(self, template: enum.Template) -> list:
        """
        Get the model data of the trees which are ported with a template.

        The template `combined` gets the trees with the categorical splits,
        the other templates the trees with the chains of numeric splits.

        Parameters
        ----------
        template : Template
            The required template.

        Returns
        -------
        The model data of all trees.
        """
        if template == enum.Template.COMBINED:
            return self._combined_trees
        return self.model_data['estimators']

    def _create_branch(
        self,
        tpls: Environment,
        language: enum.Language,
        converter: Callable[[object], str],
        left_nodes: list,
        right_nodes: list,
        threshold: list,
        value: list,
        features: list,
        missing_go_to_left: list,
        node: int,
        depth: int,
    ):
        """
        The ported single tree as nested if/else statements.

        A categorical split compares the feature with its categories, the
        numeric splits are created like the splits of the other trees.

        Parameters
        ----------
        tpls : Environment
            All relevant templates.
        language
            The required language.
        converter
            The number converter.
        left_nodes : list
            The left children of the splits.
        right_nodes : list
            The right children of the splits.
        threshold : list
            The decision thresholds of the splits.
        value : list
            The values of the leaves.
        features : list
            The feature indices of the splits.
        missing_go_to_left : list
            Whether samples with missing values (NaN) go to the left child.
        node : int
            The current node, a split `i` or a leaf `-i - 1`.
        depth : int
            The tree depth.

        Returns
        -------
        A single branch of the tree.
        """
        args = (
            tpls,
            language,
            converter,
            left_nodes,
            right_nodes,
            threshold,
            value,
            features,
            missing_go_to_left,
        )
        split = threshold[node] if node >= 0 else None
        if not isinstance(split, CategoricalSplit):
            return super()._create_branch(*args, node, depth)

        out_indent = tpls.get_template('indent').render()
        val_a = f'features[{features[node]}]'
        if language is enum.Language.PHP:
            val_a = '$' + val_a
        # The missing values (NaN) are unequal to all categories:
        categories = list(map(converter, split.categories))
        if split.missing_go_to_left:
            op, join = '!=', f' && {val_a} != '
        else:
            op, join = '==', f' || {val_a} == '
        tpl_if = tpls.get_template('if')
        out_if = tpl_if.render(a=val_a, op=op, b=join.join(categories))

        out = '\n' + indent(out_if, depth * out_indent)
        out += self._create_branch(*args, left_nodes[node], depth + 1)
        out += '\n'
        out_else = tpls.get_template('else').render()
        out += indent(out_else, depth * out_indent)
        out += self._create_branch(*args, right_nodes[node], depth + 1)
        out += '\n'
        out_endif = tpls.get_template('endif').render()
        out += indent(out_endif, depth * out_indent)
        return out
