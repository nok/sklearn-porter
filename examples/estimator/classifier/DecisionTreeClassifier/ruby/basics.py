# -*- coding: utf-8 -*-

from sklearn.tree import tree
from sklearn.datasets import load_iris
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

porter = Porter(clf, language='ruby')
output = porter.export()
print(output)

"""
class DecisionTreeClassifier

	def initialize (lChilds, rChilds, thresholds, indices, classes)
		@lChilds = lChilds
		@rChilds = rChilds
		@thresholds = thresholds
		@indices = indices
		@classes = classes
	end

	def findMax (nums)
		index = 0
		for i in 0 ... nums.length
			index = nums[i] > nums[index] ? i : index
		end
		return index
	end

	def predict (features, node=0)
		if @thresholds[node] != -2
			if features[@indices[node]] <= @thresholds[node]
				return predict features, @lChilds[node]
			else
				return predict features, @rChilds[node]
			end
		end
		return findMax @classes[node]
	end

end

if ARGV.length == 4

	# Features:
	features = ARGV.collect { |i| i.to_f }

	# Parameters:
	lChilds = [1, -1, 3, 4, 5, -1, -1, 8, -1, 10, -1, -1, 13, 14, -1, -1, -1]
	rChilds = [2, -1, 12, 7, 6, -1, -1, 9, -1, 11, -1, -1, 16, 15, -1, -1, -1]
	thresholds = [2.45000004768, -2.0, 1.75, 4.94999980927, 1.65000009537, -2.0, -2.0, 1.54999995232, -2.0, 5.44999980927, -2.0, -2.0, 4.85000038147, 3.09999990463, -2.0, -2.0, -2.0]
	indices = [2, 2, 3, 2, 3, 2, 2, 3, 2, 2, 2, 2, 2, 1, 2, 2, 2]
	classes = [[50, 50, 50], [50, 0, 0], [0, 50, 50], [0, 49, 5], [0, 47, 1], [0, 47, 0], [0, 0, 1], [0, 2, 4], [0, 0, 3], [0, 2, 1], [0, 2, 0], [0, 0, 1], [0, 1, 45], [0, 1, 2], [0, 0, 2], [0, 1, 0], [0, 0, 43]]

	# Prediction:
	clf = DecisionTreeClassifier.new lChilds, rChilds, thresholds, indices, classes
	estimation = clf.predict features
	puts estimation

end
"""
