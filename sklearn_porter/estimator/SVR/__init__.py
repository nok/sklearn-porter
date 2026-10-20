from collections.abc import Callable
from copy import deepcopy
from json import dumps

import numpy as np
from loguru import logger as L

# scikit-learn
from sklearn.svm import SVR as SVRClass
from sklearn.svm import NuSVR as NuSVRClass
from sklearn.svm import OneClassSVM as OneClassSVMClass

# sklearn-porter
from sklearn_porter import enums as enum
from sklearn_porter import exceptions as exception
from sklearn_porter.estimator.EstimatorApiABC import EstimatorApiABC
from sklearn_porter.estimator.EstimatorBase import EstimatorBase


class SVR(EstimatorBase, EstimatorApiABC):
    """Extract model data and port a SVR regressor."""

    SKLEARN_URL = 'sklearn.svm.SVR.html'

    DEFAULT_LANGUAGE = enum.Language.JS
    DEFAULT_TEMPLATE = enum.Template.ATTACHED
    DEFAULT_METHOD = enum.Method.PREDICT

    SUPPORT = {
        enum.Language.C: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.CS: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.GO: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.JAVA: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.JS: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.PHP: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
        enum.Language.RUBY: {
            enum.Template.ATTACHED: {enum.Method.PREDICT},
            enum.Template.EXPORTED: {enum.Method.PREDICT},
        },
    }

    # Outlier detectors (OneClassSVM) predict 1 (inlier) or -1 (outlier)
    # instead of the decision value:
    IS_OUTLIER_DETECTOR = False

    def __init__(self, estimator: SVRClass | NuSVRClass | OneClassSVMClass):
        super().__init__(estimator)
        L.info(f'Create specific estimator `{self.estimator_name}`.')
        est = estimator  # alias
        params = est.get_params()

        # Is the estimator fitted?
        try:
            est.support_vectors_
        except AttributeError:
            raise exception.NotFittedEstimatorError(
                self.estimator_name
            ) from None

        # Check kernel type:
        supported_kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        if params['kernel'] not in supported_kernels:
            msg = f'The passed kernel `{params["kernel"]}` is not supported.'
            raise exception.NotSupportedYetError(msg)

        # Use the computed gamma value (e.g. of 'scale' or 'auto'):
        gamma = float(est._gamma)

        # libsvm computes the decision value as the sum of the products of
        # the dual coefficients and the kernels of the support vectors minus
        # `rho`, which is the negated intercept:
        vectors = np.asarray(est.support_vectors_)
        coeffs = np.asarray(est._dual_coef_)[0]
        inter = float(est._intercept_[0])
        if len(vectors) == 0:
            # A regressor without support vectors (e.g. with a large
            # `epsilon`) predicts the intercept. A zero vector with the
            # coefficient 0 keeps the arrays of the ported code non-empty:
            n_features = est.n_features_in_  # ty: ignore[unresolved-attribute]
            vectors = np.zeros((1, n_features))
            coeffs = np.zeros(1)

        self.model_data = dict(
            vectors=vectors.tolist(),
            coeffs=coeffs.tolist(),
            inter=inter,
            kernel=params['kernel'],
            gamma=gamma,
            coef0=float(params['coef0']),
            degree=int(params['degree']),
        )
        L.info(f'Model data (keys): {self.model_data.keys()}')
        L.opt(lazy=True).debug(f'Model data: {self.model_data}')

        self.meta_info = dict(
            n_features=vectors.shape[1],
            n_vectors=len(vectors),
            is_outlier_detector=self.IS_OUTLIER_DETECTOR,
        )
        L.info(f'Meta info (keys): {self.meta_info.keys()}')
        L.opt(lazy=True).debug(f'Meta info: {self.meta_info}')

    def port(
        self,
        language: enum.Language,
        template: enum.Template,
        class_name: str,
        converter: Callable[[object], str],
        to_json: bool = False,
    ) -> str | tuple[str, str]:
        """
        Port an estimator.

        Parameters
        ----------
        language : Language
            The required language.
        template : Template
            The required template.
        class_name : str
            Change the default class name which will be used in the generated
            output. By default the class name of the passed estimator will be
            used, e.g. `SVR`.
        converter : Callable
            Change the default converter of all floating numbers from the model
            data. By default a simple string cast `str(value)` will be used.
        to_json : bool (default: False)
            Return the result as JSON string.

        Returns
        -------
        The ported estimator.
        """
        # Placeholders:
        plas = deepcopy(self.placeholders)  # alias
        plas.update(
            dict(
                class_name=class_name,
                to_json=to_json,
            )
        )
        plas.update(self.meta_info)

        # Templates:
        tpls = self._load_templates(language.value.KEY)

        # Make 'exported' variant:
        if template == enum.Template.EXPORTED:
            tpl_class = tpls.get_template('exported.class')
            out_class = tpl_class.render(**plas)
            model_data = dumps(self.model_data, separators=(',', ':'))
            return out_class, model_data

        # Make 'attached' variant:
        tpl_double = tpls.get_template('double').render()
        tpl_arr_1 = tpls.get_template('arr[]')
        tpl_arr_2 = tpls.get_template('arr[][]')
        tpl_in_brackets = tpls.get_template('in_brackets')

        vectors = self.model_data['vectors']
        vectors_str = tpl_arr_2.render(
            type=tpl_double,
            name='vectors',
            values=', '.join(
                tpl_in_brackets.render(value=', '.join(map(converter, v)))
                for v in vectors
            ),
            n=len(vectors),
            m=len(vectors[0]),
        )
        coeffs = self.model_data['coeffs']
        coeffs_str = tpl_arr_1.render(
            type=tpl_double,
            name='coeffs',
            values=', '.join(map(converter, coeffs)),
            n=len(coeffs),
        )

        # The kernel is a character in C and a string in the other languages:
        kernel = self.model_data['kernel']
        if language == enum.Language.C:
            kernel = kernel[0]

        plas.update(
            dict(
                vectors=vectors_str,
                coeffs=coeffs_str,
                inter=converter(self.model_data['inter']),
                kernel=kernel,
                gamma=converter(self.model_data['gamma']),
                coef0=converter(self.model_data['coef0']),
                degree=str(self.model_data['degree']),
            )
        )

        tpl_class = tpls.get_template('attached.class')
        return tpl_class.render(**plas)
