from importlib import import_module
from pathlib import Path
from subprocess import CompletedProcess, run

import numpy as np
import pytest

# scikit-learn
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# sklearn-porter
from sklearn_porter import Transformer
from tests.commons import DATASETS, LANGUAGES, REGRESSION_DATASETS, Dataset
from tests.utils import dataset_generate_x, dataset_uniform_x, fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

ESTIMATOR_MODULE = import_module('sklearn_porter.Base')

PARAMS = {
    'default': dict(),
    'without_mean': dict(with_mean=False),
    'without_std': dict(with_std=False),
    'without_mean_and_std': dict(with_mean=False, with_std=False),
}


def _samples(x: np.ndarray, orig_est: StandardScaler) -> np.ndarray:
    """Samples with missing values, the means and negative zeros."""
    samples = np.vstack(
        (
            dataset_uniform_x(x),
            dataset_generate_x(x, 100),
            np.full((1, x.shape[1]), -0.0),
        )
    )
    if orig_est.mean_ is not None:
        samples = np.vstack((samples, orig_est.mean_))
    samples[::7, 0] = np.nan
    return samples


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
@pytest.mark.parametrize(
    'dataset', DATASETS + REGRESSION_DATASETS, ids=lambda x: x.name
)
@pytest.mark.parametrize('params', PARAMS.values(), ids=PARAMS.keys())
def test_transformer_standard_scaler(
    tmp_root_dir: Path,
    params: dict,
    dataset: Dataset,
    language: str,
    template: str,
):
    """Test and compare the transformed features of a StandardScaler."""
    x = dataset.data
    orig_est = StandardScaler(**params).fit(x)
    est = Transformer(orig_est, language=language, template=template)
    test_x = _samples(x, orig_est)

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'transformer_standard_scaler'),
            ('params', '-'.join(map(str, params.values())) or 'default'),
            ('dataset', dataset.name),
            ('language', language),
            ('template', template),
        ],
    )
    x_tr = est.make(test_x, directory=tmp_dir, delete_created_files=False)
    # The ported transformer only subtracts and divides like scikit-learn,
    # so the transformed features are identical (also the missing values
    # and the signs of the zeros):
    x_true = orig_est.transform(test_x)
    np.testing.assert_array_equal(x_tr, x_true)
    np.testing.assert_array_equal(np.signbit(x_tr), np.signbit(x_true))
    assert est.test(test_x) == 1.0


@pytest.mark.parametrize('template', ['attached', 'combined', 'exported'])
@pytest.mark.parametrize('language', LANGUAGES)
def test_transformer_with_samples_as_arguments(
    tmp_root_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    language: str,
    template: str,
):
    """Test the features of one sample as arguments of the transformer."""
    cmds = []

    def run_and_record(cmd: list[str], **kwargs) -> CompletedProcess:
        cmds.append(cmd)
        return run(cmd, **kwargs)

    monkeypatch.setattr(ESTIMATOR_MODULE, 'run', run_and_record)

    x = load_iris().data
    orig_est = StandardScaler().fit(x)
    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'transformer_with_samples_as_arguments'),
            ('language', language),
            ('template', template),
        ],
    )
    est = Transformer(orig_est, language=language, template=template)
    x_tr = est.make(x[::15], directory=tmp_dir, delete_created_files=False)
    assert len(cmds) == 1  # one process transforms all samples
    cmd = cmds[0]

    for features, values in zip(x[::15].tolist(), x_tr, strict=True):
        args = list(map(str, features))
        proc = run(
            cmd + args, capture_output=True, encoding='utf-8', check=True
        )
        assert list(map(float, proc.stdout.split())) == values.tolist()

    # Empty lines and additional whitespace between the features:
    samples = '\n  ' + '  '.join(map(str, x[0])) + ' \t\n\n'
    proc = run(cmd, input=samples, capture_output=True, encoding='utf-8')
    assert proc.returncode == 0
    assert proc.stdout.splitlines() == [' '.join(proc.stdout.split())]
    assert list(map(float, proc.stdout.split())) == x_tr[0].tolist()

    # A sample with the wrong number of features:
    proc = run(cmd, input='1.0 2.0\n', capture_output=True, encoding='utf-8')
    assert proc.returncode != 0
