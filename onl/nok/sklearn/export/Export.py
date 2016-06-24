import sklearn


class Export:

    @staticmethod
    def export(model, method='predict'):
        if not Export._is_supported_clf(model):
            return
        name = type(model).__name__
        path = 'onl.nok.sklearn.export.classifier' + '.' + name
        class_ = getattr(__import__(path, fromlist=[name]), name)
        return class_.export(model, method)

    @staticmethod
    def _get_all_supported_clf():
        return [
            sklearn.tree.tree.DecisionTreeClassifier
        ]

    @staticmethod
    def _is_supported_clf(clf):
        support = any(isinstance(clf, e) for e in Export._get_all_supported_clf())
        if not support:
            raise ValueError('The classifier is not an instance of the supported classifiers.')
        return support


def main():
    import sys
    if len(sys.argv) == 3:
        input_file = str(sys.argv[1])
        output_file = str(sys.argv[2])
        if input_file.endswith('.pkl') and output_file.endswith('.java'):
            from sklearn.externals import joblib
            with open(output_file, 'w') as file:
                clf = joblib.load(input_file)
                java_src = ('class {0} {{ \n'
                            '    {1} \n'
                            '    public static void main(String[] args) {{ \n'
                            '        if (args.length == {2}) {{ \n'
                            '            float[] atts = new float[args.length]; \n'
                            '            for (int i = 0; i < args.length; i++) {{ \n'
                            '                atts[i] = Float.parseFloat(args[i]); \n'
                            '            }} \n'
                            '            System.out.println({0}.predict(atts)); \n'
                            '        }} \n'
                            '    }} \n'
                            '}}').format(output_file.split('.')[0].title(), Export.export(clf), clf.n_features_)
                file.write(java_src)

if __name__ == '__main__':
    main()