# -*- coding: utf-8 -*-

from typing import Union, Optional, Callable
from logging import Logger, ERROR
from pathlib import Path

from sklearn.tree.tree import DecisionTreeClassifier \
    as DecisionTreeClassifierClass

from sklearn_porter.EstimatorInterApiABC import EstimatorInterApiABC
from sklearn_porter.utils import get_logger


class DecisionTreeClassifier(EstimatorInterApiABC):
    """
    Port a DecisionTreeClassifier.

    See also
    --------
    http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """
    estimator = None  # type: DecisionTreeClassifierClass

    def __init__(
            self,
            estimator: DecisionTreeClassifierClass,
            logger: Union[Logger, int] = ERROR
    ):
        self.logger = get_logger(__name__, logger=logger)

        self.estimator = est = estimator  # alias
        self.default_class_name = estimator.__class__.__name__
        self.logger.info('Start extracting model data from `%s`.',
                         self.default_class_name)

    def port(
            self,
            method: str = 'predict',
            to: Union[str] = 'java',
            with_num_format: Callable[[object], str] = lambda x: str(x),
            with_class_name: Optional[str] = None,
            with_method_name: Optional[str] = None
    ) -> str:
        lang = to  # alias

        # Load special templates from files:
        temps_dir = Path(__file__).parent / 'templates' / lang
        temps_paths = set(temps_dir.glob('*.txt'))
        file_temps = {path.stem: path.read_text() for path in temps_paths}

        # Load standard templates from package:
        # from sklearn_porter.language.java import TEMPLATES as lang_temps
        package = 'sklearn_porter.language.' + lang
        name = 'TEMPLATES'
        lang_temps = getattr(__import__(package, fromlist=[name]), name)

        temps = {**file_temps, **lang_temps}
        print(temps.keys())

        return str(self.estimator)
