# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.datasets import load_iris
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

clf = svm.LinearSVC(C=1., random_state=0)
clf.fit(X, y)

porter = Porter(clf, language='ruby')
output = porter.export()
print(output)

"""
class LinearSVC

	def initialize (coefficients, intercepts)
		@coefficients = coefficients
		@intercepts = intercepts
	end

	def predict (features)
    	classVal = -1.0/0.0
    	classIdx = -1
    	for i in 0 ... @intercepts.length
    		prob = 0
    		for j in 0 ... @coefficients[i].length
    			prob += @coefficients[i][j] * features[j].to_f
    		end
    		if prob + @intercepts[i] > classVal
    			classVal = prob + @intercepts[i]
    			classIdx = i
    		end
    	end
    	return classIdx
    end

end

if ARGV.length == 4

	# Features:
	features = ARGV.collect { |i| i.to_f }

	# Parameters:
	coefficients = [[0.18424209458473811, 0.45123000025163923, -0.80794587716737576, -0.45071660033253858], [0.052877455748516447, -0.89214995228605254, 0.40398084459610972, -0.9376821661447452], [-0.85070784319293802, -0.98670214922204336, 1.381010448739191, 1.8654095662423917]]
	intercepts = [0.10956266406702335, 1.6636707776739579, -1.7096109416521363]

	# Prediction:
	clf = LinearSVC.new coefficients, intercepts
	estimation = clf.predict features
	puts estimation

end
"""
