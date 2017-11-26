# -*- coding: utf-8 -*-

from sklearn.tree import tree
from sklearn.datasets import load_iris
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

porter = Porter(clf)
output = porter.export(embed_data=True)
print(output)

"""
class DecisionTreeClassifier {

    public static int predict(double[] atts) {
        int[] classes = new int[3];
            
        if (atts[2] <= 2.45000004768) {
            classes[0] = 50; 
            classes[1] = 0; 
            classes[2] = 0; 
        } else {
            if (atts[3] <= 1.75) {
                if (atts[2] <= 4.94999980927) {
                    if (atts[3] <= 1.65000009537) {
                        classes[0] = 0; 
                        classes[1] = 47; 
                        classes[2] = 0; 
                    } else {
                        classes[0] = 0; 
                        classes[1] = 0; 
                        classes[2] = 1; 
                    }
                } else {
                    if (atts[3] <= 1.54999995232) {
                        classes[0] = 0; 
                        classes[1] = 0; 
                        classes[2] = 3; 
                    } else {
                        if (atts[2] <= 5.44999980927) {
                            classes[0] = 0; 
                            classes[1] = 2; 
                            classes[2] = 0; 
                        } else {
                            classes[0] = 0; 
                            classes[1] = 0; 
                            classes[2] = 1; 
                        }
                    }
                }
            } else {
                if (atts[2] <= 4.85000038147) {
                    if (atts[0] <= 5.94999980927) {
                        classes[0] = 0; 
                        classes[1] = 1; 
                        classes[2] = 0; 
                    } else {
                        classes[0] = 0; 
                        classes[1] = 0; 
                        classes[2] = 2; 
                    }
                } else {
                    classes[0] = 0; 
                    classes[1] = 0; 
                    classes[2] = 43; 
                }
            }
        }
    
        return findMax(classes);
    }

    private static int findMax(int[] nums) {
        int index = 0;
        for (int i = 0; i < nums.length; i++) {
            index = nums[i] > nums[index] ? i : index;
        }
        return index;
    }

    public static void main(String[] args) {
        if (args.length == 4) {

            // Features:
            double[] features = new double[args.length];
            for (int i = 0, l = args.length; i < l; i++) {
                features[i] = Double.parseDouble(args[i]);
            }

            // Prediction:
            int prediction = DecisionTreeClassifier.predict(features);
            System.out.println(prediction);

        }
    }
}
"""
