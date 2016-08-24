

class Classifier(object):
    def __init__(self, language='java', method_name='predict', class_name='Tmp'):
        self.language = language
        self.method_name = method_name
        self.class_name = class_name

    def port(self, model):
        self.model = model
