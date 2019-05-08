# -*- coding: utf-8 -*-

from pathlib import Path


class Templater:

    @staticmethod
    def load_templates(estimator_name: str, language: str) -> dict:
        temps = {}

        # 1. Load special templates from files:
        temps_dir = Path(__file__).parent / estimator_name / 'templates' / language
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
