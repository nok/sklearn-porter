import random
import subprocess
from unittest import TestCase
from sklearn.datasets import load_iris
from sklearn import tree
from src.Export import Export


class Test(TestCase):
    # python -m unittest discover


    def setUp(self):
        # Basic DT model:
        self.clf = tree.DecisionTreeClassifier(random_state=0)
        self.iris = load_iris()
        self.n_features = len(self.iris.data[0])
        self.clf.fit(self.iris.data, self.iris.target)

        # Porting to Java:
        self.tmp_fn = 'Tmp'
        tree_src = Export.predict(self.clf)
        with open(self.tmp_fn + '.java', 'w') as file:
            java_src = ('class {0} {{ \n'
                        '    public static {1} \n'
                        '    public static void main(String[] args) {{ \n'
                        '        if (args.length == {2}) {{ \n'
                        '            float[] atts = new float[args.length]; \n'
                        '            for (int i = 0; i < args.length; i++) {{ \n'
                        '                atts[i] = Float.parseFloat(args[i]); \n'
                        '            }} \n'
                        '            System.out.println({0}.predict(atts)); \n'
                        '        }} \n'
                        '    }} \n'
                        '}}').format(self.tmp_fn, tree_src, self.n_features)
            # print java_src
            file.write(java_src)

        # Compiling Java test class:
        subprocess.call(['javac', self.tmp_fn + '.java'])

        # Generating .dot and .png file of the DT:
        # filename = 'tree'
        # tree.export_graphviz(self.clf, out_file=filename + '.dot')
        # subprocess.call('dot -Tpng %s -o %s' % (filename + '.dot', filename + '.png'), shell=True)


    def tearDown(self):
        subprocess.call(['rm', self.tmp_fn + '.class'])
        subprocess.call(['rm', self.tmp_fn + '.java'])
        del self.clf


    def test_data_type(self):
        self.assertRaises(ValueError, Export.predict, "")


    def test_random_features(self):
        preds_from_java, preds_from_py = [], []

        # Creating random features:
        for features in range(150):
            features = [random.uniform(0., 10.) for f in range(self.n_features)]
            preds_from_java.append(self._make_prediction_in_java(features))
            preds_from_py.append(self._make_prediction_in_py(features))

        self.assertEqual(preds_from_py, preds_from_java)


    def test_existing_features(self):
        preds_from_java, preds_from_py = [], []

        # Getting existing features:
        for features in self.iris.data:
            preds_from_java.append(self._make_prediction_in_java(features))
            preds_from_py.append(self._make_prediction_in_py(features))

        self.assertEqual(preds_from_py, preds_from_java)


    def _make_prediction_in_py(self, features):
        return int(self.clf.predict([features])[0])


    def _make_prediction_in_java(self, features):
        return int(subprocess.check_output(['java', self.tmp_fn] + [str(f).strip() for f in features]).strip())
