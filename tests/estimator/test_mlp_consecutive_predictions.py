import json
from os import environ
from pathlib import Path
from subprocess import check_output

import numpy as np
import pytest

# scikit-learn
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier, MLPRegressor

# sklearn-porter
from sklearn_porter import Estimator
from tests.utils import fs_mkdir

# Long running regression tests, see `make test-regression`:
pytestmark = pytest.mark.regression

# The drivers make all predictions with one instance of the ported estimator.
# They print copies of the results right after each prediction (`immediate`)
# and the returned arrays themselves after the last prediction (`stored`).

JS_DRIVER = """
var fs = require('fs');
var Estimator = require(process.argv[2]);
var data = JSON.parse(fs.readFileSync(process.argv[3]));
var x = JSON.parse(fs.readFileSync(process.argv[4]));
var est = {constructor};
var immediate = [], stored = [];
for (var i = 0; i < x.length; i++) {{
    stored.push(est.{method}(x[i]));
    immediate.push(stored[i].slice());
}}
console.log(JSON.stringify({{immediate: immediate, stored: stored}}));
"""

JAVA_DRIVER = """
import com.google.gson.Gson;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

public class Driver {{

    static class ModelData {{
        String hidden_activation;
        String output_activation;
        int[] layers;
        double[][][] weights;
        double[][] bias;
    }}

    public static void main(String[] args) throws Exception {{
        Gson gson = new Gson();
        ModelData data = gson.fromJson(new String(Files.readAllBytes(Paths.get(args[0]))), ModelData.class);
        double[][] x = gson.fromJson(new String(Files.readAllBytes(Paths.get(args[1]))), double[][].class);
        {class_name} est = {constructor};
        double[][] immediate = new double[x.length][];
        double[][] stored = new double[x.length][];
        for (int i = 0; i < x.length; i++) {{
            stored[i] = est.{method}(x[i]);
            immediate[i] = stored[i].clone();
        }}
        Map<String, double[][]> results = new HashMap<>();
        results.put("immediate", immediate);
        results.put("stored", stored);
        System.out.println(gson.toJson(results));
    }}
}}
"""

CONSTRUCTORS = {
    ('MLPClassifier', 'attached'): (
        '{new}(data.hidden_activation, data.output_activation, '
        'data.layers, data.weights, data.bias)'
    ),
    ('MLPRegressor', 'attached'): (
        '{new}(data.hidden_activation, data.layers, data.weights, data.bias)'
    ),
}


def run_driver(
    est: Estimator,
    language: str,
    template: str,
    x: np.ndarray,
    model_data: dict,
    tmp_dir: Path,
) -> dict:
    """Make all predictions with one instance of the ported estimator."""
    class_name = est.class_name
    method = 'predict' if class_name == 'MLPRegressor' else 'predictProba'

    out = est.save(language=language, template=template, directory=tmp_dir)
    if template == 'exported':
        src_path, data_path = map(Path, out)
    else:
        src_path, data_path = Path(out), tmp_dir / 'model_data.json'
        data_path.write_text(json.dumps(model_data))
    x_path = tmp_dir / 'x.json'
    x_path.write_text(json.dumps(x.tolist()))

    if language == 'js':
        new = 'new Estimator'
        constructor = (
            f'{new}(process.argv[3])'
            if template == 'exported'
            else CONSTRUCTORS[(class_name, template)].format(new=new)
        )
        module_path = tmp_dir / 'estimator.module.js'
        module_path.write_text(
            src_path.read_text() + f'\nmodule.exports = {class_name};\n'
        )
        driver_path = tmp_dir / 'driver.js'
        driver_path.write_text(
            JS_DRIVER.format(constructor=constructor, method=method)
        )
        cmd = ['node', driver_path, module_path, data_path, x_path]
    else:
        new = f'new {class_name}'
        constructor = (
            f'{new}(args[0])'
            if template == 'exported'
            else CONSTRUCTORS[(class_name, template)].format(new=new)
        )
        driver_path = tmp_dir / 'Driver.java'
        driver_path.write_text(
            JAVA_DRIVER.format(
                class_name=class_name, constructor=constructor, method=method
            )
        )
        class_path = f'{tmp_dir}:{environ["SKLEARN_PORTER_PYTEST_GSON_PATH"]}'
        check_output(
            ['javac', '-cp', class_path, '-d', tmp_dir, src_path, driver_path]
        )
        cmd = ['java', '-cp', class_path, 'Driver', data_path, x_path]

    return json.loads(check_output(cmd, text=True))


@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
@pytest.mark.parametrize('template', ['attached', 'exported'])
@pytest.mark.parametrize('language', ['java', 'js'])
@pytest.mark.parametrize('estimator', ['MLPClassifier', 'MLPRegressor'])
def test_estimator_mlp_with_consecutive_predictions(
    tmp_root_dir: Path, estimator: str, language: str, template: str
):
    """
    Test many consecutive predictions with one instance of a ported MLP.

    The Java and JS templates of the MLPClassifier and the JS templates of
    the MLPRegressor reuse the arrays of the network for each prediction
    (the other languages create new arrays). The results of earlier
    predictions must neither affect later ones nor be overwritten by them
    (#48).
    """
    x, y = load_iris(return_X_y=True)
    if estimator == 'MLPClassifier':
        # Three classes, so `predictProba` returns the output layer:
        orig_est = MLPClassifier(
            hidden_layer_sizes=(15, 10), max_iter=200, random_state=0
        ).fit(x, y)
        expected = orig_est.predict_proba(x)
    else:
        # Two outputs, so `predict` returns the output layer:
        orig_est = MLPRegressor(
            hidden_layer_sizes=(15, 10), max_iter=200, random_state=0
        ).fit(x[:, :2], x[:, 2:])
        x = x[:, :2]
        expected = orig_est.predict(x)

    model_data = dict(
        hidden_activation=orig_est.activation,
        output_activation=orig_est.out_activation_,
        layers=[len(b) for b in orig_est.intercepts_],
        weights=[w.tolist() for w in orig_est.coefs_],
        bias=[b.tolist() for b in orig_est.intercepts_],
    )

    tmp_dir = fs_mkdir(
        tmp_root_dir,
        [
            ('test', 'estimator_mlp_with_consecutive_predictions'),
            ('estimator', estimator),
            ('language', language),
            ('template', template),
        ],
    )

    results = run_driver(
        Estimator(orig_est), language, template, x, model_data, tmp_dir
    )

    np.testing.assert_allclose(results['immediate'], expected, atol=1e-12)
    np.testing.assert_allclose(results['stored'], expected, atol=1e-12)
