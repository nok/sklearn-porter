from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from sklearn_porter import Porter

X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier(n_estimators=15, max_depth=None,
                             min_samples_split=2, random_state=0)
clf.fit(X, y)

# Cheese!

result = Porter(language='js').port(clf)
print(result)

"""
var predictor = function(atts) {

    var predict_00 = function(atts) {
        var i = 0, classes = new Array(3);

        if (atts[3] <= 0.75) {
            classes[0] = 47;
            classes[1] = 0;
            classes[2] = 0;
        } else {
            if (atts[2] <= 4.8500003814697266) {
                if (atts[3] <= 1.6500000953674316) {
                    classes[0] = 0;
                    classes[1] = 42;
                    classes[2] = 0;
                } else {
                    if (atts[1] <= 3.0) {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 3;
                    } else {
                        classes[0] = 0;
                        classes[1] = 1;
                        classes[2] = 0;
                    }
                }
            } else {
                if (atts[0] <= 6.5999999046325684) {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 27;
                } else {
                    if (atts[2] <= 5.1999998092651367) {
                        classes[0] = 0;
                        classes[1] = 1;
                        classes[2] = 0;
                    } else {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 29;
                    }
                }
            }
        }
        var class_idx = 0, class_val = classes[0];
        for (i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    };
    var predict_01 = function(atts) {
        var i = 0, classes = new Array(3);

        if (atts[3] <= 0.80000001192092896) {
            classes[0] = 46;
            classes[1] = 0;
            classes[2] = 0;
        } else {
            if (atts[3] <= 1.75) {
                if (atts[2] <= 4.9499998092651367) {
                    classes[0] = 0;
                    classes[1] = 58;
                    classes[2] = 0;
                } else {
                    if (atts[2] <= 5.4499998092651367) {
                        if (atts[1] <= 2.4500000476837158) {
                            classes[0] = 0;
                            classes[1] = 0;
                            classes[2] = 2;
                        } else {
                            classes[0] = 0;
                            classes[1] = 3;
                            classes[2] = 0;
                        }
                    } else {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 3;
                    }
                }
            } else {
                if (atts[2] <= 4.8500003814697266) {
                    if (atts[1] <= 3.0999999046325684) {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 2;
                    } else {
                        classes[0] = 0;
                        classes[1] = 1;
                        classes[2] = 0;
                    }
                } else {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 35;
                }
            }
        }
        var class_idx = 0, class_val = classes[0];
        for (i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    };
    var predict_02 = function(atts) {
        var i = 0, classes = new Array(3);

        if (atts[0] <= 5.5500001907348633) {
            if (atts[3] <= 0.80000001192092896) {
                classes[0] = 49;
                classes[1] = 0;
                classes[2] = 0;
            } else {
                if (atts[3] <= 1.6000000238418579) {
                    classes[0] = 0;
                    classes[1] = 12;
                    classes[2] = 0;
                } else {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 1;
                }
            }
        } else {
            if (atts[3] <= 1.5499999523162842) {
                if (atts[3] <= 0.75) {
                    classes[0] = 2;
                    classes[1] = 0;
                    classes[2] = 0;
                } else {
                    if (atts[2] <= 5.0) {
                        classes[0] = 0;
                        classes[1] = 32;
                        classes[2] = 0;
                    } else {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 1;
                    }
                }
            } else {
                if (atts[2] <= 4.6500000953674316) {
                    classes[0] = 0;
                    classes[1] = 1;
                    classes[2] = 0;
                } else {
                    if (atts[3] <= 1.7000000476837158) {
                        if (atts[2] <= 5.4499998092651367) {
                            classes[0] = 0;
                            classes[1] = 1;
                            classes[2] = 0;
                        } else {
                            classes[0] = 0;
                            classes[1] = 0;
                            classes[2] = 3;
                        }
                    } else {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 48;
                    }
                }
            }
        }
        var class_idx = 0, class_val = classes[0];
        for (i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    };
    var predict_03 = function(atts) {
        var i = 0, classes = new Array(3);

        if (atts[0] <= 5.4499998092651367) {
            if (atts[1] <= 2.8000001907348633) {
                if (atts[1] <= 2.4500000476837158) {
                    classes[0] = 0;
                    classes[1] = 5;
                    classes[2] = 0;
                } else {
                    if (atts[0] <= 5.0) {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 3;
                    } else {
                        classes[0] = 0;
                        classes[1] = 3;
                        classes[2] = 0;
                    }
                }
            } else {
                classes[0] = 41;
                classes[1] = 0;
                classes[2] = 0;
            }
        } else {
            if (atts[0] <= 6.25) {
                if (atts[3] <= 1.7000000476837158) {
                    if (atts[3] <= 0.60000002384185791) {
                        classes[0] = 3;
                        classes[1] = 0;
                        classes[2] = 0;
                    } else {
                        if (atts[1] <= 2.25) {
                            if (atts[3] <= 1.25) {
                                classes[0] = 0;
                                classes[1] = 1;
                                classes[2] = 0;
                            } else {
                                if (atts[2] <= 4.75) {
                                    classes[0] = 0;
                                    classes[1] = 3;
                                    classes[2] = 0;
                                } else {
                                    classes[0] = 0;
                                    classes[1] = 0;
                                    classes[2] = 1;
                                }
                            }
                        } else {
                            classes[0] = 0;
                            classes[1] = 37;
                            classes[2] = 0;
                        }
                    }
                } else {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 8;
                }
            } else {
                if (atts[2] <= 4.9499998092651367) {
                    classes[0] = 0;
                    classes[1] = 10;
                    classes[2] = 0;
                } else {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 35;
                }
            }
        }
        var class_idx = 0, class_val = classes[0];
        for (i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    };
    var predict_04 = function(atts) {
        var i = 0, classes = new Array(3);

        if (atts[3] <= 0.69999998807907104) {
            classes[0] = 50;
            classes[1] = 0;
            classes[2] = 0;
        } else {
            if (atts[3] <= 1.75) {
                if (atts[2] <= 5.0500001907348633) {
                    if (atts[2] <= 4.9499998092651367) {
                        classes[0] = 0;
                        classes[1] = 56;
                        classes[2] = 0;
                    } else {
                        if (atts[3] <= 1.6000000238418579) {
                            classes[0] = 0;
                            classes[1] = 0;
                            classes[2] = 1;
                        } else {
                            classes[0] = 0;
                            classes[1] = 3;
                            classes[2] = 0;
                        }
                    }
                } else {
                    if (atts[0] <= 6.0500001907348633) {
                        classes[0] = 0;
                        classes[1] = 2;
                        classes[2] = 0;
                    } else {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 5;
                    }
                }
            } else {
                classes[0] = 0;
                classes[1] = 0;
                classes[2] = 33;
            }
        }
        var class_idx = 0, class_val = classes[0];
        for (i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    };
    var predict_05 = function(atts) {
        var i = 0, classes = new Array(3);

        if (atts[3] <= 0.80000001192092896) {
            classes[0] = 49;
            classes[1] = 0;
            classes[2] = 0;
        } else {
            if (atts[2] <= 4.9499998092651367) {
                if (atts[0] <= 4.9499998092651367) {
                    if (atts[3] <= 1.3500000238418579) {
                        classes[0] = 0;
                        classes[1] = 1;
                        classes[2] = 0;
                    } else {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 1;
                    }
                } else {
                    if (atts[2] <= 4.75) {
                        classes[0] = 0;
                        classes[1] = 49;
                        classes[2] = 0;
                    } else {
                        if (atts[1] <= 2.5999999046325684) {
                            classes[0] = 0;
                            classes[1] = 1;
                            classes[2] = 0;
                        } else {
                            if (atts[0] <= 6.0500001907348633) {
                                classes[0] = 0;
                                classes[1] = 1;
                                classes[2] = 0;
                            } else {
                                if (atts[3] <= 1.5999999046325684) {
                                    classes[0] = 0;
                                    classes[1] = 1;
                                    classes[2] = 0;
                                } else {
                                    classes[0] = 0;
                                    classes[1] = 0;
                                    classes[2] = 3;
                                }
                            }
                        }
                    }
                }
            } else {
                classes[0] = 0;
                classes[1] = 0;
                classes[2] = 44;
            }
        }
        var class_idx = 0, class_val = classes[0];
        for (i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    };
    var predict_06 = function(atts) {
        var i = 0, classes = new Array(3);

        if (atts[3] <= 0.69999998807907104) {
            classes[0] = 46;
            classes[1] = 0;
            classes[2] = 0;
        } else {
            if (atts[2] <= 4.75) {
                if (atts[0] <= 4.9499998092651367) {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 2;
                } else {
                    classes[0] = 0;
                    classes[1] = 39;
                    classes[2] = 0;
                }
            } else {
                if (atts[2] <= 5.1499996185302734) {
                    if (atts[0] <= 6.5999999046325684) {
                        if (atts[3] <= 1.7000000476837158) {
                            if (atts[3] <= 1.5499999523162842) {
                                classes[0] = 0;
                                classes[1] = 0;
                                classes[2] = 2;
                            } else {
                                classes[0] = 0;
                                classes[1] = 1;
                                classes[2] = 0;
                            }
                        } else {
                            classes[0] = 0;
                            classes[1] = 0;
                            classes[2] = 19;
                        }
                    } else {
                        classes[0] = 0;
                        classes[1] = 3;
                        classes[2] = 0;
                    }
                } else {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 38;
                }
            }
        }
        var class_idx = 0, class_val = classes[0];
        for (i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    };
    var predict_07 = function(atts) {
        var i = 0, classes = new Array(3);

        if (atts[2] <= 2.5999999046325684) {
            classes[0] = 58;
            classes[1] = 0;
            classes[2] = 0;
        } else {
            if (atts[2] <= 4.75) {
                classes[0] = 0;
                classes[1] = 37;
                classes[2] = 0;
            } else {
                if (atts[2] <= 5.1499996185302734) {
                    if (atts[3] <= 1.75) {
                        if (atts[0] <= 6.5) {
                            if (atts[2] <= 4.9499998092651367) {
                                classes[0] = 0;
                                classes[1] = 1;
                                classes[2] = 0;
                            } else {
                                if (atts[0] <= 6.1500000953674316) {
                                    if (atts[3] <= 1.5499999523162842) {
                                        classes[0] = 0;
                                        classes[1] = 0;
                                        classes[2] = 2;
                                    } else {
                                        classes[0] = 0;
                                        classes[1] = 1;
                                        classes[2] = 0;
                                    }
                                } else {
                                    classes[0] = 0;
                                    classes[1] = 0;
                                    classes[2] = 2;
                                }
                            }
                        } else {
                            classes[0] = 0;
                            classes[1] = 2;
                            classes[2] = 0;
                        }
                    } else {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 13;
                    }
                } else {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 34;
                }
            }
        }
        var class_idx = 0, class_val = classes[0];
        for (i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    };
    var predict_08 = function(atts) {
        var i = 0, classes = new Array(3);

        if (atts[3] <= 0.69999998807907104) {
            classes[0] = 42;
            classes[1] = 0;
            classes[2] = 0;
        } else {
            if (atts[0] <= 6.25) {
                if (atts[2] <= 4.8000001907348633) {
                    if (atts[0] <= 4.9499998092651367) {
                        if (atts[1] <= 2.4500000476837158) {
                            classes[0] = 0;
                            classes[1] = 1;
                            classes[2] = 0;
                        } else {
                            classes[0] = 0;
                            classes[1] = 0;
                            classes[2] = 3;
                        }
                    } else {
                        classes[0] = 0;
                        classes[1] = 36;
                        classes[2] = 0;
                    }
                } else {
                    if (atts[3] <= 1.5499999523162842) {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 4;
                    } else {
                        if (atts[3] <= 1.7000000476837158) {
                            classes[0] = 0;
                            classes[1] = 2;
                            classes[2] = 0;
                        } else {
                            classes[0] = 0;
                            classes[1] = 0;
                            classes[2] = 4;
                        }
                    }
                }
            } else {
                if (atts[3] <= 1.75) {
                    if (atts[2] <= 5.0500001907348633) {
                        classes[0] = 0;
                        classes[1] = 15;
                        classes[2] = 0;
                    } else {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 4;
                    }
                } else {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 39;
                }
            }
        }
        var class_idx = 0, class_val = classes[0];
        for (i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    };
    var predict_09 = function(atts) {
        var i = 0, classes = new Array(3);

        if (atts[2] <= 2.5999999046325684) {
            classes[0] = 55;
            classes[1] = 0;
            classes[2] = 0;
        } else {
            if (atts[2] <= 4.9499998092651367) {
                if (atts[0] <= 5.9499998092651367) {
                    classes[0] = 0;
                    classes[1] = 23;
                    classes[2] = 0;
                } else {
                    if (atts[3] <= 1.6499999761581421) {
                        classes[0] = 0;
                        classes[1] = 16;
                        classes[2] = 0;
                    } else {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 4;
                    }
                }
            } else {
                if (atts[0] <= 6.5999999046325684) {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 33;
                } else {
                    if (atts[0] <= 6.75) {
                        if (atts[3] <= 2.0) {
                            classes[0] = 0;
                            classes[1] = 1;
                            classes[2] = 0;
                        } else {
                            classes[0] = 0;
                            classes[1] = 0;
                            classes[2] = 4;
                        }
                    } else {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 14;
                    }
                }
            }
        }
        var class_idx = 0, class_val = classes[0];
        for (i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    };
    var predict_10 = function(atts) {
        var i = 0, classes = new Array(3);

        if (atts[3] <= 0.80000001192092896) {
            classes[0] = 52;
            classes[1] = 0;
            classes[2] = 0;
        } else {
            if (atts[2] <= 4.75) {
                classes[0] = 0;
                classes[1] = 37;
                classes[2] = 0;
            } else {
                if (atts[3] <= 1.75) {
                    if (atts[2] <= 4.9499998092651367) {
                        classes[0] = 0;
                        classes[1] = 4;
                        classes[2] = 0;
                    } else {
                        if (atts[1] <= 2.6500000953674316) {
                            classes[0] = 0;
                            classes[1] = 0;
                            classes[2] = 2;
                        } else {
                            if (atts[3] <= 1.5499999523162842) {
                                classes[0] = 0;
                                classes[1] = 0;
                                classes[2] = 2;
                            } else {
                                if (atts[2] <= 5.4499998092651367) {
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
                    }
                } else {
                    if (atts[2] <= 4.8500003814697266) {
                        if (atts[1] <= 3.0999999046325684) {
                            classes[0] = 0;
                            classes[1] = 0;
                            classes[2] = 6;
                        } else {
                            classes[0] = 0;
                            classes[1] = 1;
                            classes[2] = 0;
                        }
                    } else {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 43;
                    }
                }
            }
        }
        var class_idx = 0, class_val = classes[0];
        for (i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    };
    var predict_11 = function(atts) {
        var i = 0, classes = new Array(3);

        if (atts[2] <= 2.5999999046325684) {
            classes[0] = 47;
            classes[1] = 0;
            classes[2] = 0;
        } else {
            if (atts[2] <= 4.75) {
                classes[0] = 0;
                classes[1] = 40;
                classes[2] = 0;
            } else {
                if (atts[2] <= 4.9499998092651367) {
                    if (atts[1] <= 3.0499999523162842) {
                        if (atts[3] <= 1.5999999046325684) {
                            classes[0] = 0;
                            classes[1] = 2;
                            classes[2] = 0;
                        } else {
                            classes[0] = 0;
                            classes[1] = 0;
                            classes[2] = 7;
                        }
                    } else {
                        classes[0] = 0;
                        classes[1] = 2;
                        classes[2] = 0;
                    }
                } else {
                    if (atts[0] <= 6.0500001907348633) {
                        if (atts[2] <= 5.0500001907348633) {
                            classes[0] = 0;
                            classes[1] = 0;
                            classes[2] = 4;
                        } else {
                            if (atts[0] <= 5.9499998092651367) {
                                classes[0] = 0;
                                classes[1] = 0;
                                classes[2] = 7;
                            } else {
                                classes[0] = 0;
                                classes[1] = 1;
                                classes[2] = 0;
                            }
                        }
                    } else {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 40;
                    }
                }
            }
        }
        var class_idx = 0, class_val = classes[0];
        for (i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    };
    var predict_12 = function(atts) {
        var i = 0, classes = new Array(3);

        if (atts[3] <= 0.80000001192092896) {
            classes[0] = 54;
            classes[1] = 0;
            classes[2] = 0;
        } else {
            if (atts[1] <= 2.4500000476837158) {
                if (atts[2] <= 4.75) {
                    classes[0] = 0;
                    classes[1] = 12;
                    classes[2] = 0;
                } else {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 1;
                }
            } else {
                if (atts[3] <= 1.6000000238418579) {
                    if (atts[2] <= 5.0) {
                        classes[0] = 0;
                        classes[1] = 23;
                        classes[2] = 0;
                    } else {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 2;
                    }
                } else {
                    if (atts[3] <= 1.75) {
                        if (atts[0] <= 5.8000001907348633) {
                            classes[0] = 0;
                            classes[1] = 0;
                            classes[2] = 3;
                        } else {
                            classes[0] = 0;
                            classes[1] = 2;
                            classes[2] = 0;
                        }
                    } else {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 53;
                    }
                }
            }
        }
        var class_idx = 0, class_val = classes[0];
        for (i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    };
    var predict_13 = function(atts) {
        var i = 0, classes = new Array(3);

        if (atts[0] <= 5.4499998092651367) {
            if (atts[3] <= 0.80000001192092896) {
                classes[0] = 36;
                classes[1] = 0;
                classes[2] = 0;
            } else {
                if (atts[2] <= 4.1999998092651367) {
                    classes[0] = 0;
                    classes[1] = 6;
                    classes[2] = 0;
                } else {
                    if (atts[1] <= 2.75) {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 1;
                    } else {
                        classes[0] = 0;
                        classes[1] = 1;
                        classes[2] = 0;
                    }
                }
            }
        } else {
            if (atts[2] <= 4.9000000953674316) {
                if (atts[1] <= 3.5999999046325684) {
                    classes[0] = 0;
                    classes[1] = 43;
                    classes[2] = 0;
                } else {
                    classes[0] = 7;
                    classes[1] = 0;
                    classes[2] = 0;
                }
            } else {
                if (atts[3] <= 1.7000000476837158) {
                    if (atts[3] <= 1.5499999523162842) {
                        classes[0] = 0;
                        classes[1] = 0;
                        classes[2] = 2;
                    } else {
                        classes[0] = 0;
                        classes[1] = 4;
                        classes[2] = 0;
                    }
                } else {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 50;
                }
            }
        }
        var class_idx = 0, class_val = classes[0];
        for (i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    };
    var predict_14 = function(atts) {
        var i = 0, classes = new Array(3);

        if (atts[2] <= 2.5999999046325684) {
            classes[0] = 52;
            classes[1] = 0;
            classes[2] = 0;
        } else {
            if (atts[3] <= 1.7000000476837158) {
                if (atts[0] <= 7.0) {
                    if (atts[2] <= 5.0) {
                        classes[0] = 0;
                        classes[1] = 48;
                        classes[2] = 0;
                    } else {
                        if (atts[0] <= 6.0500001907348633) {
                            classes[0] = 0;
                            classes[1] = 1;
                            classes[2] = 0;
                        } else {
                            classes[0] = 0;
                            classes[1] = 0;
                            classes[2] = 2;
                        }
                    }
                } else {
                    classes[0] = 0;
                    classes[1] = 0;
                    classes[2] = 1;
                }
            } else {
                classes[0] = 0;
                classes[1] = 0;
                classes[2] = 46;
            }
        }
        var class_idx = 0, class_val = classes[0];
        for (i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    };

    var predict = function(atts) {
        var i = 0, n_classes = 3;
        var classes = new Array(n_classes);
        for (i = 0; i < n_classes; i++) {
            classes[i] = 0;
        }

        classes[predict_00(atts)]++;
        classes[predict_01(atts)]++;
        classes[predict_02(atts)]++;
        classes[predict_03(atts)]++;
        classes[predict_04(atts)]++;
        classes[predict_05(atts)]++;
        classes[predict_06(atts)]++;
        classes[predict_07(atts)]++;
        classes[predict_08(atts)]++;
        classes[predict_09(atts)]++;
        classes[predict_10(atts)]++;
        classes[predict_11(atts)]++;
        classes[predict_12(atts)]++;
        classes[predict_13(atts)]++;
        classes[predict_14(atts)]++;

        var class_idx = 0, class_val = classes[0];
        for (i = 1; i < n_classes; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    }

    return predict(atts);
};

if (typeof process !== 'undefined' && typeof process.argv !== 'undefined') {
    if (process.argv.length - 2 == 4) {
        var argv = process.argv.slice(2);
        var prediction = predictor(argv);
        console.log(prediction);
    }
}
"""