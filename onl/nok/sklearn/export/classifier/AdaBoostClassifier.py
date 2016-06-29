from onl.nok.sklearn.export.classifier.Classifier import Classifier


class AdaBoostClassifier(Classifier):


    @staticmethod
    def get_supported_methods():
        return [
            'predict'
        ]


    @staticmethod
    def is_supported_method(method_name):
        support = method_name in AdaBoostClassifier.get_supported_methods()
        if not support:
            raise ValueError('The classifier does not support the given method.')
        return support


    @staticmethod
    def export(model, method_name='predict', class_name="Tmp"):
        # TODO: Implement export
        pass