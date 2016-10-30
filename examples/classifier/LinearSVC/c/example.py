from sklearn import svm
from sklearn.datasets import load_iris

from onl.nok.sklearn.Porter import port

X, y = load_iris(return_X_y=True)
clf = svm.LinearSVC(C=1., random_state=0)
clf.fit(X, y)

# Cheese!

print(port(clf, language='c'))

"""
#include <stdio.h>
#include <math.h>

int predict (float atts[4]) {

    double coefs[3][4] = {{0.18424209458473811, 0.45123000025163923, -0.80794587716737576, -0.45071660033253858}, {0.052877455748516447, -0.89214995228605254, 0.40398084459610972, -0.9376821661447452}, {-0.85070784319293802, -0.98670214922204336, 1.381010448739191, 1.8654095662423917}};
    double inters[3] = {0.10956266406702335, 1.6636707776739579, -1.7096109416521363};

    double class_val = -INFINITY;
    int class_idx = -1;
    for (int i = 0; i < 3; i++) {
        double prob = 0.;
        for (int j = 0; j < 4; j++) {
            prob += coefs[i][j] * atts[j];
        }
        if (prob + inters[i] > class_val) {
            class_val = prob + inters[i];
            class_idx = i;
        }
    }
    return class_idx;
}

int main(int argc, const char * argv[]) {
    float test[4] = {1.f, 2.f, 3.f, 4.f};
    printf("Result: %d\\n", predict(test));
    return 0;
}
"""