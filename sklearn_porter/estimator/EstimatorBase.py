# -*- coding: utf-8 -*-

from pathlib import Path

from sklearn.base import BaseEstimator


class EstimatorBase:

    estimator = None  # type: BaseEstimator

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator
        self.estimator_name = estimator.__class__.__qualname__

    def load_templates(self, language: str) -> dict:
        temps = {}

        # 1. Load special templates from files:
        file_dir = Path(__file__).parent
        temps_dir = file_dir / self.estimator_name / 'templates' / language
        if temps_dir.exists():
            temps_paths = set(temps_dir.glob('*.txt'))
            temps.update({path.stem: path.read_text() for path in temps_paths})

        # 2. Load basic templates from package:
        # from sklearn_porter.language.java import TEMPLATES as lang_temps
        package = 'sklearn_porter.language.' + language
        name = 'TEMPLATES'
        lang_temps = getattr(__import__(package, fromlist=[name]), name)
        if isinstance(lang_temps, dict):
            temps.update(lang_temps)

        return temps
