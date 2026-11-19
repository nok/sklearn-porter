# Usage

The generated source code is a program and a library at the same time: you can run it with the features of a sample (see the [Quick Start](https://github.com/nok/sklearn-porter#quick-start)), and you can use its class in your own code. The header of each generated file shows both.

This page uses the classifier of the Quick Start:

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn_porter import save

X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)

save(clf, language='java', directory='.')  # or another language and template
```

- [Create an instance](#create-an-instance)
- [Methods](#methods)
- [Languages](#languages): [C](#c), [C#](#c-1), [Go](#go), [Java](#java), [JavaScript](#javascript), [PHP](#php), [Ruby](#ruby)
- [Pipelines](#pipelines)
- [Notes](#notes)


## Create an instance

The [template](https://github.com/nok/sklearn-porter#templates) defines how you get an instance of the class:

| Language   | `attached`                       | `exported`                                    | `combined`                       |
|------------|----------------------------------|-----------------------------------------------|----------------------------------|
| C          | – (functions)                    | `load("X.json")`                              | – (functions)                    |
| C#         | `X.Create()`                     | `X.Load("X.json")`                            | static methods¹                  |
| Go         | `NewX()`                         | `LoadX("X.json")`, returns `(X, error)`       | functions¹                       |
| Java       | `X.create()`                     | `X.load("X.json")`                            | static methods¹                  |
| JavaScript | `new X()`                        | `new X("X.json")` or `new X(data)`            | `new X()`                        |
| PHP        | `X::create()`                    | `X::load('X.json')`                           | static methods¹                  |
| Ruby       | `X.create`                       | `X.load('X.json')`                            | class methods¹                   |

`X` is the class name, by default the name of the class of scikit-learn (e.g. `DecisionTreeClassifier`), otherwise the argument `class_name` of `port` or `save`. With `attached` the model data is part of the source code, with `exported` it's in the JSON file of `save`.

¹ The transformers of the template `combined` are instances without arguments (e.g. `new StandardScaler()` in Java). Some estimators of the template `combined` also need an instance, e.g. a `GradientBoostingClassifier`. Pipelines and composite transformers (`ColumnTransformer`, `FeatureUnion`) always have the factory of the template `attached`. The header of each generated file shows the right call.


## Methods

The methods take the features of one sample as an array of floating point numbers, in the order of the columns of the training data:

- Classifiers: `predict` returns the label of the class (like scikit-learn), `predictIndex` the index of the class (in `classes_`) and `predictProba` the probabilities of the classes, if the estimator supports them.
- Regressors: `predict` returns the predicted value (or the values of all outputs).
- Outlier detectors: `predict` returns `1` (inlier) or `-1` (outlier).
- Transformers: `transform` returns the transformed features. Text transformers (e.g. a `CountVectorizer`) take a document (a string) instead.

The names follow the conventions of each language: `predictProba` in Java, JavaScript and PHP, `PredictProba` in C# and Go, `predict_proba` in C and Ruby.

The methods don't check the number of features, only the program does.


## Languages

### C

Define `SKLEARN_PORTER_NO_MAIN` before you include the generated file, so that it doesn't define `main`. The file defines the number of features (`N_FEATURES`) and of the classes of a classifier (`N_CLASSES`):

```c
#include <stdio.h>

#define SKLEARN_PORTER_NO_MAIN
#include "DecisionTreeClassifier.c"

int main(void) {
    double features[N_FEATURES] = {6.3, 3.3, 6.0, 2.5};
    printf("%d\n", predict(features));  // 2

    double probabilities[N_CLASSES];
    predict_proba(probabilities, features);
    printf("%f\n", probabilities[2]);  // 0.978261
    return 0;
}
```

```console
$ gcc main.c -std=c99 -O2 -lm -o main && ./main
```

With the template `exported` call `load("DecisionTreeClassifier.json");` once before the predictions. A transformer writes the transformed features into an array of the length `N_FEATURES_OUT`: `transform(values, features)`. Regressors with several outputs write their values into an array of the length `N_OUTPUTS`: `predict(values, features)`.

### C#

Add the generated file to your project and define `SKLEARN_PORTER_NO_MAIN`, so that it doesn't define `Main`:

```xml
<PropertyGroup>
  <DefineConstants>$(DefineConstants);SKLEARN_PORTER_NO_MAIN</DefineConstants>
</PropertyGroup>
```

```csharp
using System;

public static class App {
    public static void Main() {
        DecisionTreeClassifier estimator = DecisionTreeClassifier.Create();
        double[] features = { 6.3, 3.3, 6.0, 2.5 };
        Console.WriteLine(estimator.Predict(features));  // 2
        Console.WriteLine(estimator.PredictProba(features)[2]);  // 0.9782608695652174
    }
}
```

The templates `attached` and `combined` only need C# 7.3 (e.g. also .NET Framework or Unity), the template `exported` needs `System.Text.Json` of .NET 8 or newer.

### Go

Put the generated file into a package of your module: replace `package main` by the name of the package (e.g. `package model`) and remove the function `main`.

```go
package main

import (
	"fmt"

	"example.com/app/model"
)

func main() {
	estimator := model.NewDecisionTreeClassifier()
	features := []float64{6.3, 3.3, 6.0, 2.5}
	fmt.Println(estimator.Predict(features))         // 2
	fmt.Println(estimator.PredictProba(features)[2]) // 0.9782608695652174
}
```

```console
$ go run .
```

With the template `exported` `LoadDecisionTreeClassifier("DecisionTreeClassifier.json")` returns the estimator and an error. The estimators of the template `combined` are functions of the package, e.g. `model.Predict(features)`.

### Java

Compile the generated file with your code:

```java
public class App {
    public static void main(String[] args) {
        DecisionTreeClassifier estimator = DecisionTreeClassifier.create();
        double[] features = { 6.3, 3.3, 6.0, 2.5 };
        System.out.println(estimator.predict(features));  // 2
        System.out.println(estimator.predictProba(features)[2]);  // 0.9782608695652174
    }
}
```

```console
$ javac DecisionTreeClassifier.java App.java && java App
```

The classes of the generated file have no package and aren't `public`, so add `package` to the file, if your code is in a package. The template `exported` needs [Gson](https://github.com/google/gson) (`javac -cp gson.jar ...`, `java -cp .:gson.jar ...`) and `load` throws an `IOException`.

### JavaScript

The generated file runs in Node.js and in browsers. It exports the class with `module.exports` (Node.js) or as a global variable (e.g. in a browser):

```javascript
const DecisionTreeClassifier = require('./DecisionTreeClassifier.js');

const estimator = new DecisionTreeClassifier();
const features = [6.3, 3.3, 6.0, 2.5];
console.log(estimator.predict(features));  // 2
console.log(estimator.predictProba(features)[2]);  // 0.9782608695652174
```

An ES module imports it with `import DecisionTreeClassifier from './DecisionTreeClassifier.js';`. In a browser:

```html
<script src="DecisionTreeClassifier.js"></script>
<script>
    const estimator = new DecisionTreeClassifier();
    console.log(estimator.predict([6.3, 3.3, 6.0, 2.5]));  // 2
</script>
```

With the template `exported` the constructor takes the path of the JSON file (Node.js) or the parsed model data, e.g. in a browser:

```javascript
fetch('DecisionTreeClassifier.json')
    .then((response) => response.json())
    .then((data) => {
        const estimator = new DecisionTreeClassifier(data);
        console.log(estimator.predict([6.3, 3.3, 6.0, 2.5]));  // 2
    });
```

The classes don't use the APIs of Node.js (e.g. `require`, `process`), only the program and the reading of a JSON file do.

### PHP

Include the generated file, which runs its program only if PHP executes it directly:

```php
<?php
require_once 'DecisionTreeClassifier.php';

$estimator = DecisionTreeClassifier::create();
$features = [6.3, 3.3, 6.0, 2.5];
echo $estimator->predict($features), "\n";  // 2
echo $estimator->predictProba($features)[2], "\n";  // 0.97826086956522
```

```console
$ php app.php
```

### Ruby

Require the generated file, which runs its program only if Ruby executes it directly:

```ruby
require_relative 'DecisionTreeClassifier'

estimator = DecisionTreeClassifier.create
features = [6.3, 3.3, 6.0, 2.5]
puts estimator.predict(features)  # 2
puts estimator.predict_proba(features)[2]  # 0.9782608695652174
```

```console
$ ruby app.rb
```


## Pipelines

A ported [`Pipeline`](https://github.com/nok/sklearn-porter/blob/main/examples/pipelines.ipynb) is one source file with the class `Pipeline` (or the argument `class_name`). It has the factories of the table above (`Pipeline.create()`, `Pipeline.load("Pipeline.json")` etc., also with the template `combined`) and holds the transformers and the final estimator:

- `transform` applies the transformers to the features.
- `predict`, `predictIndex` and `predictProba` (as far as the final estimator has them) predict the transformed features.
- `estimator` (`Estimator` in C# and Go) is the final estimator, which expects transformed features.

```java
Pipeline pipeline = Pipeline.create();
pipeline.predict(new double[] { 14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065 });
```

In C the functions have the prefix `pipeline_`: `pipeline_transform(values, features)`, `pipeline_predict(features)`, `pipeline_predict_proba(probabilities, features)` and with the template `exported` `pipeline_load("Pipeline.json")`. In JavaScript `new Pipeline()` creates an object with the methods `predict`, `predictIndex` and `predictProba`.

A `ColumnTransformer` or a `FeatureUnion` has the factories and the method `transform` of the other transformers.


## Notes

- Each generated file is self-contained, so the names of its classes, functions and global variables can collide with those of another generated file in the same program, namespace or package. In C use one generated file per translation unit and in Go one per package. In Java, C#, PHP and Ruby choose different class names with `class_name`. The class names of the transformers in a pipeline can repeat in two pipelines (e.g. `StandardScaler`), so put them into different packages or namespaces.
- The classes don't change their model data, so you can create an instance once and use it for all predictions.
- The ported code predicts like scikit-learn, but the [estimator notes](https://github.com/nok/sklearn-porter/blob/main/docs/estimator-notes.md) and the [limitations](https://github.com/nok/sklearn-porter/blob/main/docs/limitations.md) list the known differences. Check the ported estimator with `test()` before you use it.
