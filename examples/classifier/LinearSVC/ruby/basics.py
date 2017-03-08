# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.datasets import load_iris

from sklearn_porter import Porter


iris_data = load_iris()
X, y = iris_data.data, iris_data.target
clf = svm.LinearSVC(C=1., random_state=0)
clf.fit(X, y)

# Cheese!

result = Porter(clf, language='ruby').export()
print(result)

"""
class Tmp

    def self.predict (atts)

        coefs = [[0.18424209458473811, 0.45123000025163923, -0.80794587716737576, -0.45071660033253858], [0.052877455748516447, -0.89214995228605254, 0.40398084459610972, -0.9376821661447452], [-0.85070784319293802, -0.98670214922204336, 1.381010448739191, 1.8654095662423917]]
        inters = [0.10956266406702335, 1.6636707776739579, -1.7096109416521363]

        class_val = -1.0/0.0
        class_idx = -1

        for i in 0 ... 3
            prob = 0
            for j in 0 ... 4
                prob += coefs[i][j] * atts[j].to_f
            end
            if prob + inters[i] > class_val
                class_val = prob + inters[i]
                class_idx = i
            end
        end

        return class_idx
    end

end

if ARGV.length == 4
    puts Tmp.predict(ARGV)
end
"""
