def export_pl(clf, method_name='predict'):
    """Convert a learned decision tree model to a Java based equivalent.

    Parameters
    ----------
    clf : sklearn.tree.tree.DecisionTreeClassifier
        The learned decision tree model.

    method_name : string, optional (default="predict")
        The name of the created method.

    See also
    --------
    sklearn.tree.tree.DecisionTreeClassifier

    Returns
    -------
    source : string
        The decision tree wrapped in a Java method.
    """

    import sklearn

    if type(clf) is not sklearn.tree.tree.DecisionTreeClassifier:
        raise ValueError('The data type is not an instance of sklearn.tree.tree.DecisionTreeClassifier')
    else:
        n_features = clf.n_features_
        n_classes = clf.n_classes_

        def _recurse(left, right, threshold, value, features, node, depth):
            out = ''
            indent = '\n' + '    ' * depth
            if threshold[node] != -2.:
                out += indent + 'if (atts[{0}] <= {1:.6f}f) {{'.format(features[node], threshold[node])
                if left[node] != -1.:
                    out += _recurse(left, right, threshold, value, features, left[node], depth + 1)
                out += indent + '} else {'
                if right[node] != -1.:
                    out += _recurse(left, right, threshold, value, features, right[node], depth + 1)
                out += indent + '}'
            else:
                out += ';'.join([indent + 'classes[{0}] = {1}'.format(i, int(v)) for i, v in enumerate(value[node][0])]) + ';'
            return out

        features = [[str(idx) for idx in range(n_features)][i] for i in clf.tree_.feature]
        conditions = _recurse(
            clf.tree_.children_left,
            clf.tree_.children_right,
            clf.tree_.threshold,
            clf.tree_.value, features, 0, 1)

        source = (
            'int {0} (float[] atts) {{ \n'
            '    int n_classes = {1}; \n'
            '    int[] classes = new int[n_classes]; \n'
            '    {2} \n\n'
            '    int idx = 0; \n'
            '    int val = classes[0]; \n'
            '    for (int i = 1; i < n_classes; i++) {{ \n'
            '        if (classes[i] > val) {{ \n'
            '            idx = i; \n'
            '        }} \n'
            '    }} \n'
            '    return idx; \n'
            '}}'
        ).format(str(method_name), n_classes, conditions)

        return source

def main():
    pass

if __name__ == '__main__':
    main()