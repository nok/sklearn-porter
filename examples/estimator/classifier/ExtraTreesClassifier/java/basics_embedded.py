# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier
from sklearn_porter import Porter


iris_data = load_iris()
X = iris_data.data
y = iris_data.target

clf = ExtraTreesClassifier(n_estimators=15, max_depth=None,
                           min_samples_split=2, random_state=0)
clf.fit(X, y)

porter = Porter(clf)
output = porter.export(embed_data=True)
print(output)

"""
class ExtraTreesClassifier {
    public static int predict_0(double[] features) {
        int[] classes = new int[3];
        
        if (features[0] <= 5.52112836752) {
            if (features[1] <= 2.54236914672) {
                if (features[2] <= 2.32859402389) {
                    classes[0] = 1; 
                    classes[1] = 0; 
                    classes[2] = 0; 
                } else {
                    if (features[3] <= 1.4511234333) {
                        classes[0] = 0; 
                        classes[1] = 8; 
                        classes[2] = 0; 
                    } else {
                        classes[0] = 0; 
                        classes[1] = 0; 
                        classes[2] = 1; 
                    }
                }
            } else {
                if (features[3] <= 0.83878367201) {
                    classes[0] = 46; 
                    classes[1] = 0; 
                    classes[2] = 0; 
                } else {
                    classes[0] = 0; 
                    classes[1] = 3; 
                    classes[2] = 0; 
                }
            }
        } else {
            if (features[3] <= 1.89369306498) {
                if (features[2] <= 5.18076136748) {
                    if (features[3] <= 0.25425931195) {
                        classes[0] = 1; 
                        classes[1] = 0; 
                        classes[2] = 0; 
                    } else {
                        if (features[3] <= 1.29791099198) {
                            if (features[1] <= 3.14667550028) {
                                classes[0] = 0; 
                                classes[1] = 8; 
                                classes[2] = 0; 
                            } else {
                                classes[0] = 2; 
                                classes[1] = 0; 
                                classes[2] = 0; 
                            }
                        } else {
                            if (features[3] <= 1.49177428996) {
                                classes[0] = 0; 
                                classes[1] = 17; 
                                classes[2] = 0; 
                            } else {
                                if (features[2] <= 4.64861094171) {
                                    classes[0] = 0; 
                                    classes[1] = 7; 
                                    classes[2] = 0; 
                                } else {
                                    if (features[3] <= 1.78027764001) {
                                        if (features[3] <= 1.50659268982) {
                                            if (features[2] <= 4.89970329741) {
                                                classes[0] = 0; 
                                                classes[1] = 1; 
                                                classes[2] = 0; 
                                            } else {
                                                if (features[2] <= 4.98785335085) {
                                                    classes[0] = 0; 
                                                    classes[1] = 2; 
                                                    classes[2] = 0; 
                                                } else {
                                                    classes[0] = 0; 
                                                    classes[1] = 0; 
                                                    classes[2] = 2; 
                                                }
                                            }
                                        } else {
                                            classes[0] = 0; 
                                            classes[1] = 3; 
                                            classes[2] = 0; 
                                        }
                                    } else {
                                        if (features[1] <= 3.12447121782) {
                                            classes[0] = 0; 
                                            classes[1] = 0; 
                                            classes[2] = 5; 
                                        } else {
                                            classes[0] = 0; 
                                            classes[1] = 1; 
                                            classes[2] = 0; 
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    classes[0] = 0; 
                    classes[1] = 0; 
                    classes[2] = 8; 
                }
            } else {
                classes[0] = 0; 
                classes[1] = 0; 
                classes[2] = 34; 
            }
        }
        int class_idx = 0;
        int class_val = classes[0];
        for (int i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    }
    
    public static int predict_1(double[] features) {
        int[] classes = new int[3];
        
        if (features[3] <= 1.55049555018) {
            if (features[2] <= 2.59232852929) {
                classes[0] = 50; 
                classes[1] = 0; 
                classes[2] = 0; 
            } else {
                if (features[0] <= 5.70518977343) {
                    classes[0] = 0; 
                    classes[1] = 21; 
                    classes[2] = 0; 
                } else {
                    if (features[1] <= 2.4211563618) {
                        if (features[1] <= 2.23763666184) {
                            if (features[3] <= 1.08522535119) {
                                classes[0] = 0; 
                                classes[1] = 1; 
                                classes[2] = 0; 
                            } else {
                                if (features[2] <= 4.55871034323) {
                                    classes[0] = 0; 
                                    classes[1] = 1; 
                                    classes[2] = 0; 
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 1; 
                                }
                            }
                        } else {
                            classes[0] = 0; 
                            classes[1] = 1; 
                            classes[2] = 0; 
                        }
                    } else {
                        if (features[1] <= 2.88778331709) {
                            if (features[3] <= 1.27295083379) {
                                classes[0] = 0; 
                                classes[1] = 4; 
                                classes[2] = 0; 
                            } else {
                                if (features[0] <= 6.57901134114) {
                                    if (features[2] <= 4.78938062738) {
                                        classes[0] = 0; 
                                        classes[1] = 2; 
                                        classes[2] = 0; 
                                    } else {
                                        if (features[0] <= 6.12176696672) {
                                            classes[0] = 0; 
                                            classes[1] = 0; 
                                            classes[2] = 1; 
                                        } else {
                                            if (features[2] <= 4.91622918275) {
                                                classes[0] = 0; 
                                                classes[1] = 1; 
                                                classes[2] = 0; 
                                            } else {
                                                classes[0] = 0; 
                                                classes[1] = 0; 
                                                classes[2] = 1; 
                                            }
                                        }
                                    }
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 1; 
                                    classes[2] = 0; 
                                }
                            }
                        } else {
                            classes[0] = 0; 
                            classes[1] = 13; 
                            classes[2] = 0; 
                        }
                    }
                }
            }
        } else {
            if (features[1] <= 3.1730407215) {
                if (features[3] <= 2.27062195979) {
                    if (features[0] <= 5.70347014875) {
                        classes[0] = 0; 
                        classes[1] = 0; 
                        classes[2] = 3; 
                    } else {
                        if (features[2] <= 5.63718610177) {
                            if (features[0] <= 6.61522923342) {
                                if (features[2] <= 5.56768407747) {
                                    if (features[1] <= 2.91513890853) {
                                        if (features[0] <= 6.36760077608) {
                                            if (features[0] <= 6.2352298442) {
                                                if (features[3] <= 1.65270259989) {
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
                                                classes[2] = 2; 
                                            }
                                        } else {
                                            classes[0] = 0; 
                                            classes[1] = 0; 
                                            classes[2] = 1; 
                                        }
                                    } else {
                                        classes[0] = 0; 
                                        classes[1] = 0; 
                                        classes[2] = 6; 
                                    }
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 3; 
                                }
                            } else {
                                if (features[0] <= 6.71800419294) {
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
                            classes[2] = 8; 
                        }
                    }
                } else {
                    classes[0] = 0; 
                    classes[1] = 0; 
                    classes[2] = 6; 
                }
            } else {
                if (features[3] <= 2.12457024836) {
                    if (features[3] <= 1.81961335953) {
                        if (features[0] <= 6.52049342951) {
                            classes[0] = 0; 
                            classes[1] = 3; 
                            classes[2] = 0; 
                        } else {
                            classes[0] = 0; 
                            classes[1] = 0; 
                            classes[2] = 1; 
                        }
                    } else {
                        classes[0] = 0; 
                        classes[1] = 0; 
                        classes[2] = 3; 
                    }
                } else {
                    classes[0] = 0; 
                    classes[1] = 0; 
                    classes[2] = 9; 
                }
            }
        }
        int class_idx = 0;
        int class_val = classes[0];
        for (int i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    }
    
    public static int predict_2(double[] features) {
        int[] classes = new int[3];
        
        if (features[3] <= 1.5465266341) {
            if (features[2] <= 4.60713170404) {
                if (features[2] <= 3.30032825798) {
                    if (features[3] <= 0.386587526669) {
                        classes[0] = 41; 
                        classes[1] = 0; 
                        classes[2] = 0; 
                    } else {
                        if (features[2] <= 2.15564750547) {
                            classes[0] = 9; 
                            classes[1] = 0; 
                            classes[2] = 0; 
                        } else {
                            classes[0] = 0; 
                            classes[1] = 3; 
                            classes[2] = 0; 
                        }
                    }
                } else {
                    classes[0] = 0; 
                    classes[1] = 35; 
                    classes[2] = 0; 
                }
            } else {
                if (features[1] <= 2.26967532377) {
                    classes[0] = 0; 
                    classes[1] = 0; 
                    classes[2] = 1; 
                } else {
                    if (features[2] <= 5.48270001541) {
                        if (features[2] <= 4.99722601253) {
                            classes[0] = 0; 
                            classes[1] = 7; 
                            classes[2] = 0; 
                        } else {
                            classes[0] = 0; 
                            classes[1] = 0; 
                            classes[2] = 1; 
                        }
                    } else {
                        classes[0] = 0; 
                        classes[1] = 0; 
                        classes[2] = 1; 
                    }
                }
            }
        } else {
            if (features[3] <= 1.603207012) {
                if (features[0] <= 6.51874902633) {
                    classes[0] = 0; 
                    classes[1] = 3; 
                    classes[2] = 0; 
                } else {
                    classes[0] = 0; 
                    classes[1] = 0; 
                    classes[2] = 1; 
                }
            } else {
                if (features[3] <= 1.80814361156) {
                    if (features[1] <= 2.59962235264) {
                        classes[0] = 0; 
                        classes[1] = 0; 
                        classes[2] = 2; 
                    } else {
                        if (features[1] <= 3.02662394793) {
                            if (features[3] <= 1.77883715966) {
                                classes[0] = 0; 
                                classes[1] = 1; 
                                classes[2] = 0; 
                            } else {
                                classes[0] = 0; 
                                classes[1] = 0; 
                                classes[2] = 8; 
                            }
                        } else {
                            if (features[2] <= 5.58271595967) {
                                if (features[1] <= 3.14527456161) {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 1; 
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 1; 
                                    classes[2] = 0; 
                                }
                            } else {
                                classes[0] = 0; 
                                classes[1] = 0; 
                                classes[2] = 1; 
                            }
                        }
                    }
                } else {
                    classes[0] = 0; 
                    classes[1] = 0; 
                    classes[2] = 34; 
                }
            }
        }
        int class_idx = 0;
        int class_val = classes[0];
        for (int i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    }
    
    public static int predict_3(double[] features) {
        int[] classes = new int[3];
        
        if (features[0] <= 5.10965102517) {
            if (features[3] <= 0.342374369907) {
                classes[0] = 31; 
                classes[1] = 0; 
                classes[2] = 0; 
            } else {
                if (features[3] <= 0.420141410785) {
                    classes[0] = 3; 
                    classes[1] = 0; 
                    classes[2] = 0; 
                } else {
                    if (features[1] <= 3.30137192332) {
                        if (features[2] <= 2.32187538756) {
                            classes[0] = 1; 
                            classes[1] = 0; 
                            classes[2] = 0; 
                        } else {
                            if (features[0] <= 4.95825619025) {
                                if (features[1] <= 2.47268127251) {
                                    classes[0] = 0; 
                                    classes[1] = 1; 
                                    classes[2] = 0; 
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 1; 
                                }
                            } else {
                                classes[0] = 0; 
                                classes[1] = 3; 
                                classes[2] = 0; 
                            }
                        }
                    } else {
                        classes[0] = 1; 
                        classes[1] = 0; 
                        classes[2] = 0; 
                    }
                }
            }
        } else {
            if (features[2] <= 5.00599388242) {
                if (features[3] <= 0.592155863234) {
                    classes[0] = 14; 
                    classes[1] = 0; 
                    classes[2] = 0; 
                } else {
                    if (features[2] <= 4.88691585855) {
                        if (features[2] <= 4.58459567778) {
                            classes[0] = 0; 
                            classes[1] = 32; 
                            classes[2] = 0; 
                        } else {
                            if (features[1] <= 3.00135217081) {
                                if (features[0] <= 6.28181215328) {
                                    if (features[3] <= 1.35660149972) {
                                        classes[0] = 0; 
                                        classes[1] = 1; 
                                        classes[2] = 0; 
                                    } else {
                                        if (features[3] <= 1.4692521397) {
                                            classes[0] = 0; 
                                            classes[1] = 2; 
                                            classes[2] = 0; 
                                        } else {
                                            classes[0] = 0; 
                                            classes[1] = 0; 
                                            classes[2] = 2; 
                                        }
                                    }
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 3; 
                                    classes[2] = 0; 
                                }
                            } else {
                                classes[0] = 0; 
                                classes[1] = 4; 
                                classes[2] = 0; 
                            }
                        }
                    } else {
                        if (features[0] <= 6.02478940741) {
                            classes[0] = 0; 
                            classes[1] = 0; 
                            classes[2] = 3; 
                        } else {
                            if (features[3] <= 1.53573128649) {
                                classes[0] = 0; 
                                classes[1] = 2; 
                                classes[2] = 0; 
                            } else {
                                if (features[1] <= 2.58270805636) {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 1; 
                                } else {
                                    if (features[0] <= 6.56690791608) {
                                        classes[0] = 0; 
                                        classes[1] = 0; 
                                        classes[2] = 2; 
                                    } else {
                                        classes[0] = 0; 
                                        classes[1] = 1; 
                                        classes[2] = 0; 
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                if (features[2] <= 5.18632453863) {
                    if (features[1] <= 2.70920826751) {
                        if (features[3] <= 1.87760057409) {
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
                        classes[2] = 5; 
                    }
                } else {
                    classes[0] = 0; 
                    classes[1] = 0; 
                    classes[2] = 34; 
                }
            }
        }
        int class_idx = 0;
        int class_val = classes[0];
        for (int i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    }
    
    public static int predict_4(double[] features) {
        int[] classes = new int[3];
        
        if (features[3] <= 1.34411613251) {
            if (features[1] <= 3.01014066227) {
                if (features[0] <= 5.55133473528) {
                    if (features[1] <= 2.45046894972) {
                        if (features[0] <= 5.34933996473) {
                            if (features[0] <= 4.87501529156) {
                                classes[0] = 1; 
                                classes[1] = 0; 
                                classes[2] = 0; 
                            } else {
                                classes[0] = 0; 
                                classes[1] = 3; 
                                classes[2] = 0; 
                            }
                        } else {
                            classes[0] = 0; 
                            classes[1] = 3; 
                            classes[2] = 0; 
                        }
                    } else {
                        if (features[1] <= 2.85457325045) {
                            classes[0] = 0; 
                            classes[1] = 3; 
                            classes[2] = 0; 
                        } else {
                            classes[0] = 7; 
                            classes[1] = 0; 
                            classes[2] = 0; 
                        }
                    }
                } else {
                    classes[0] = 0; 
                    classes[1] = 19; 
                    classes[2] = 0; 
                }
            } else {
                classes[0] = 42; 
                classes[1] = 0; 
                classes[2] = 0; 
            }
        } else {
            if (features[2] <= 4.54122132652) {
                if (features[3] <= 1.67969737853) {
                    classes[0] = 0; 
                    classes[1] = 10; 
                    classes[2] = 0; 
                } else {
                    classes[0] = 0; 
                    classes[1] = 0; 
                    classes[2] = 1; 
                }
            } else {
                if (features[3] <= 1.86812158914) {
                    if (features[3] <= 1.70827702212) {
                        if (features[2] <= 5.45877854198) {
                            if (features[1] <= 2.63222182317) {
                                if (features[1] <= 2.20756413547) {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 1; 
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 1; 
                                    classes[2] = 0; 
                                }
                            } else {
                                if (features[1] <= 2.97506566466) {
                                    if (features[0] <= 6.57000136283) {
                                        if (features[3] <= 1.43782315105) {
                                            classes[0] = 0; 
                                            classes[1] = 1; 
                                            classes[2] = 0; 
                                        } else {
                                            if (features[3] <= 1.53445889283) {
                                                if (features[0] <= 6.46420655971) {
                                                    classes[0] = 0; 
                                                    classes[1] = 0; 
                                                    classes[2] = 1; 
                                                } else {
                                                    classes[0] = 0; 
                                                    classes[1] = 1; 
                                                    classes[2] = 0; 
                                                }
                                            } else {
                                                classes[0] = 0; 
                                                classes[1] = 1; 
                                                classes[2] = 0; 
                                            }
                                        }
                                    } else {
                                        classes[0] = 0; 
                                        classes[1] = 1; 
                                        classes[2] = 0; 
                                    }
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 6; 
                                    classes[2] = 0; 
                                }
                            }
                        } else {
                            classes[0] = 0; 
                            classes[1] = 0; 
                            classes[2] = 2; 
                        }
                    } else {
                        if (features[1] <= 2.8407804212) {
                            classes[0] = 0; 
                            classes[1] = 0; 
                            classes[2] = 3; 
                        } else {
                            if (features[2] <= 6.15083697198) {
                                if (features[1] <= 3.13375324586) {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 6; 
                                } else {
                                    if (features[2] <= 5.82925935747) {
                                        classes[0] = 0; 
                                        classes[1] = 1; 
                                        classes[2] = 0; 
                                    } else {
                                        classes[0] = 0; 
                                        classes[1] = 0; 
                                        classes[2] = 1; 
                                    }
                                }
                            } else {
                                classes[0] = 0; 
                                classes[1] = 0; 
                                classes[2] = 1; 
                            }
                        }
                    }
                } else {
                    classes[0] = 0; 
                    classes[1] = 0; 
                    classes[2] = 34; 
                }
            }
        }
        int class_idx = 0;
        int class_val = classes[0];
        for (int i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    }
    
    public static int predict_5(double[] features) {
        int[] classes = new int[3];
        
        if (features[3] <= 1.76762133327) {
            if (features[1] <= 3.45456106177) {
                if (features[2] <= 3.65028546216) {
                    if (features[2] <= 3.18915864204) {
                        if (features[3] <= 0.951019813633) {
                            classes[0] = 29; 
                            classes[1] = 0; 
                            classes[2] = 0; 
                        } else {
                            classes[0] = 0; 
                            classes[1] = 1; 
                            classes[2] = 0; 
                        }
                    } else {
                        classes[0] = 0; 
                        classes[1] = 5; 
                        classes[2] = 0; 
                    }
                } else {
                    if (features[0] <= 5.1755356443) {
                        classes[0] = 0; 
                        classes[1] = 0; 
                        classes[2] = 1; 
                    } else {
                        if (features[3] <= 1.45338501524) {
                            if (features[1] <= 2.66075274742) {
                                if (features[3] <= 1.11977212075) {
                                    classes[0] = 0; 
                                    classes[1] = 4; 
                                    classes[2] = 0; 
                                } else {
                                    if (features[2] <= 5.03405350962) {
                                        classes[0] = 0; 
                                        classes[1] = 5; 
                                        classes[2] = 0; 
                                    } else {
                                        classes[0] = 0; 
                                        classes[1] = 0; 
                                        classes[2] = 1; 
                                    }
                                }
                            } else {
                                classes[0] = 0; 
                                classes[1] = 20; 
                                classes[2] = 0; 
                            }
                        } else {
                            if (features[1] <= 3.20006555968) {
                                if (features[1] <= 2.64893114355) {
                                    if (features[0] <= 6.0403355971) {
                                        classes[0] = 0; 
                                        classes[1] = 0; 
                                        classes[2] = 1; 
                                    } else {
                                        classes[0] = 0; 
                                        classes[1] = 2; 
                                        classes[2] = 0; 
                                    }
                                } else {
                                    if (features[0] <= 6.50336366188) {
                                        if (features[0] <= 5.60451214748) {
                                            classes[0] = 0; 
                                            classes[1] = 2; 
                                            classes[2] = 0; 
                                        } else {
                                            if (features[3] <= 1.5280856302) {
                                                if (features[0] <= 5.98533534592) {
                                                    classes[0] = 0; 
                                                    classes[1] = 1; 
                                                    classes[2] = 0; 
                                                } else {
                                                    if (features[1] <= 2.8357509786) {
                                                        if (features[2] <= 4.663137086) {
                                                            classes[0] = 0; 
                                                            classes[1] = 1; 
                                                            classes[2] = 0; 
                                                        } else {
                                                            classes[0] = 0; 
                                                            classes[1] = 0; 
                                                            classes[2] = 1; 
                                                        }
                                                    } else {
                                                        classes[0] = 0; 
                                                        classes[1] = 2; 
                                                        classes[2] = 0; 
                                                    }
                                                }
                                            } else {
                                                classes[0] = 0; 
                                                classes[1] = 1; 
                                                classes[2] = 0; 
                                            }
                                        }
                                    } else {
                                        if (features[1] <= 3.05637165108) {
                                            if (features[3] <= 1.65453526013) {
                                                classes[0] = 0; 
                                                classes[1] = 0; 
                                                classes[2] = 1; 
                                            } else {
                                                classes[0] = 0; 
                                                classes[1] = 1; 
                                                classes[2] = 0; 
                                            }
                                        } else {
                                            classes[0] = 0; 
                                            classes[1] = 2; 
                                            classes[2] = 0; 
                                        }
                                    }
                                }
                            } else {
                                classes[0] = 0; 
                                classes[1] = 2; 
                                classes[2] = 0; 
                            }
                        }
                    }
                }
            } else {
                classes[0] = 21; 
                classes[1] = 0; 
                classes[2] = 0; 
            }
        } else {
            if (features[3] <= 1.81219694316) {
                if (features[2] <= 5.56282895862) {
                    if (features[2] <= 4.81184906512) {
                        if (features[1] <= 3.16844224586) {
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
                        classes[2] = 5; 
                    }
                } else {
                    classes[0] = 0; 
                    classes[1] = 0; 
                    classes[2] = 4; 
                }
            } else {
                classes[0] = 0; 
                classes[1] = 0; 
                classes[2] = 34; 
            }
        }
        int class_idx = 0;
        int class_val = classes[0];
        for (int i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    }
    
    public static int predict_6(double[] features) {
        int[] classes = new int[3];
        
        if (features[3] <= 1.83807208772) {
            if (features[3] <= 0.949086299728) {
                classes[0] = 50; 
                classes[1] = 0; 
                classes[2] = 0; 
            } else {
                if (features[2] <= 3.92596141457) {
                    classes[0] = 0; 
                    classes[1] = 11; 
                    classes[2] = 0; 
                } else {
                    if (features[0] <= 5.75243973806) {
                        if (features[0] <= 5.56758460997) {
                            if (features[3] <= 1.30961587262) {
                                classes[0] = 0; 
                                classes[1] = 3; 
                                classes[2] = 0; 
                            } else {
                                if (features[0] <= 4.99652734613) {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 1; 
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 1; 
                                    classes[2] = 0; 
                                }
                            }
                        } else {
                            classes[0] = 0; 
                            classes[1] = 7; 
                            classes[2] = 0; 
                        }
                    } else {
                        if (features[3] <= 1.32262107945) {
                            classes[0] = 0; 
                            classes[1] = 9; 
                            classes[2] = 0; 
                        } else {
                            if (features[3] <= 1.53316595251) {
                                if (features[1] <= 2.93630410206) {
                                    if (features[1] <= 2.62713714657) {
                                        if (features[2] <= 5.21327146488) {
                                            if (features[0] <= 6.14508177984) {
                                                classes[0] = 0; 
                                                classes[1] = 0; 
                                                classes[2] = 1; 
                                            } else {
                                                classes[0] = 0; 
                                                classes[1] = 2; 
                                                classes[2] = 0; 
                                            }
                                        } else {
                                            classes[0] = 0; 
                                            classes[1] = 0; 
                                            classes[2] = 1; 
                                        }
                                    } else {
                                        if (features[0] <= 6.10490230891) {
                                            classes[0] = 0; 
                                            classes[1] = 2; 
                                            classes[2] = 0; 
                                        } else {
                                            if (features[3] <= 1.45769340279) {
                                                classes[0] = 0; 
                                                classes[1] = 1; 
                                                classes[2] = 0; 
                                            } else {
                                                if (features[2] <= 5.07028804346) {
                                                    classes[0] = 0; 
                                                    classes[1] = 1; 
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
                                    classes[0] = 0; 
                                    classes[1] = 8; 
                                    classes[2] = 0; 
                                }
                            } else {
                                if (features[2] <= 5.12604031073) {
                                    if (features[1] <= 3.1470031033) {
                                        if (features[2] <= 4.807945315) {
                                            classes[0] = 0; 
                                            classes[1] = 0; 
                                            classes[2] = 2; 
                                        } else {
                                            if (features[0] <= 6.15190329272) {
                                                if (features[0] <= 6.04304697271) {
                                                    if (features[1] <= 2.95806332672) {
                                                        classes[0] = 0; 
                                                        classes[1] = 1; 
                                                        classes[2] = 0; 
                                                    } else {
                                                        classes[0] = 0; 
                                                        classes[1] = 0; 
                                                        classes[2] = 1; 
                                                    }
                                                } else {
                                                    classes[0] = 0; 
                                                    classes[1] = 0; 
                                                    classes[2] = 1; 
                                                }
                                            } else {
                                                if (features[2] <= 4.92119560196) {
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
                                        classes[0] = 0; 
                                        classes[1] = 3; 
                                        classes[2] = 0; 
                                    }
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 7; 
                                }
                            }
                        }
                    }
                }
            }
        } else {
            classes[0] = 0; 
            classes[1] = 0; 
            classes[2] = 34; 
        }
        int class_idx = 0;
        int class_val = classes[0];
        for (int i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    }
    
    public static int predict_7(double[] features) {
        int[] classes = new int[3];
        
        if (features[0] <= 4.83507798095) {
            classes[0] = 16; 
            classes[1] = 0; 
            classes[2] = 0; 
        } else {
            if (features[2] <= 5.05212113171) {
                if (features[3] <= 1.1879841232) {
                    if (features[2] <= 2.41985028298) {
                        classes[0] = 34; 
                        classes[1] = 0; 
                        classes[2] = 0; 
                    } else {
                        classes[0] = 0; 
                        classes[1] = 10; 
                        classes[2] = 0; 
                    }
                } else {
                    if (features[3] <= 1.97009791611) {
                        if (features[3] <= 1.21035233337) {
                            classes[0] = 0; 
                            classes[1] = 5; 
                            classes[2] = 0; 
                        } else {
                            if (features[3] <= 1.79228240275) {
                                if (features[3] <= 1.35560125882) {
                                    classes[0] = 0; 
                                    classes[1] = 13; 
                                    classes[2] = 0; 
                                } else {
                                    if (features[1] <= 2.35063676993) {
                                        if (features[2] <= 4.7810477029) {
                                            classes[0] = 0; 
                                            classes[1] = 1; 
                                            classes[2] = 0; 
                                        } else {
                                            classes[0] = 0; 
                                            classes[1] = 0; 
                                            classes[2] = 1; 
                                        }
                                    } else {
                                        if (features[0] <= 5.23969249721) {
                                            if (features[3] <= 1.54009925042) {
                                                classes[0] = 0; 
                                                classes[1] = 1; 
                                                classes[2] = 0; 
                                            } else {
                                                classes[0] = 0; 
                                                classes[1] = 0; 
                                                classes[2] = 1; 
                                            }
                                        } else {
                                            classes[0] = 0; 
                                            classes[1] = 18; 
                                            classes[2] = 0; 
                                        }
                                    }
                                }
                            } else {
                                if (features[2] <= 4.81145991801) {
                                    if (features[0] <= 6.15311592932) {
                                        if (features[0] <= 5.91784028103) {
                                            classes[0] = 0; 
                                            classes[1] = 1; 
                                            classes[2] = 0; 
                                        } else {
                                            classes[0] = 0; 
                                            classes[1] = 0; 
                                            classes[2] = 1; 
                                        }
                                    } else {
                                        classes[0] = 0; 
                                        classes[1] = 0; 
                                        classes[2] = 1; 
                                    }
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 3; 
                                }
                            }
                        }
                    } else {
                        classes[0] = 0; 
                        classes[1] = 0; 
                        classes[2] = 2; 
                    }
                }
            } else {
                if (features[3] <= 1.96303146541) {
                    if (features[0] <= 6.32149848089) {
                        if (features[0] <= 5.97185726192) {
                            classes[0] = 0; 
                            classes[1] = 0; 
                            classes[2] = 3; 
                        } else {
                            if (features[1] <= 2.79578590244) {
                                if (features[3] <= 1.56474058102) {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 1; 
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
                        classes[1] = 0; 
                        classes[2] = 8; 
                    }
                } else {
                    classes[0] = 0; 
                    classes[1] = 0; 
                    classes[2] = 27; 
                }
            }
        }
        int class_idx = 0;
        int class_val = classes[0];
        for (int i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    }
    
    public static int predict_8(double[] features) {
        int[] classes = new int[3];
        
        if (features[2] <= 2.06910163978) {
            classes[0] = 50; 
            classes[1] = 0; 
            classes[2] = 0; 
        } else {
            if (features[3] <= 2.29983219774) {
                if (features[2] <= 5.75685015533) {
                    if (features[3] <= 1.33363532238) {
                        classes[0] = 0; 
                        classes[1] = 28; 
                        classes[2] = 0; 
                    } else {
                        if (features[1] <= 2.79582066137) {
                            if (features[3] <= 1.6143543486) {
                                if (features[2] <= 4.2721874374) {
                                    classes[0] = 0; 
                                    classes[1] = 1; 
                                    classes[2] = 0; 
                                } else {
                                    if (features[3] <= 1.40767164553) {
                                        classes[0] = 0; 
                                        classes[1] = 0; 
                                        classes[2] = 1; 
                                    } else {
                                        if (features[1] <= 2.4055632554) {
                                            if (features[0] <= 6.16869284105) {
                                                classes[0] = 0; 
                                                classes[1] = 0; 
                                                classes[2] = 1; 
                                            } else {
                                                classes[0] = 0; 
                                                classes[1] = 1; 
                                                classes[2] = 0; 
                                            }
                                        } else {
                                            classes[0] = 0; 
                                            classes[1] = 2; 
                                            classes[2] = 0; 
                                        }
                                    }
                                }
                            } else {
                                classes[0] = 0; 
                                classes[1] = 0; 
                                classes[2] = 7; 
                            }
                        } else {
                            if (features[3] <= 2.02675739385) {
                                if (features[3] <= 1.60936797125) {
                                    if (features[2] <= 4.63464527803) {
                                        classes[0] = 0; 
                                        classes[1] = 10; 
                                        classes[2] = 0; 
                                    } else {
                                        if (features[2] <= 4.95207328243) {
                                            classes[0] = 0; 
                                            classes[1] = 6; 
                                            classes[2] = 0; 
                                        } else {
                                            classes[0] = 0; 
                                            classes[1] = 0; 
                                            classes[2] = 1; 
                                        }
                                    }
                                } else {
                                    if (features[1] <= 2.87208741558) {
                                        classes[0] = 0; 
                                        classes[1] = 0; 
                                        classes[2] = 2; 
                                    } else {
                                        if (features[0] <= 6.51739754754) {
                                            if (features[1] <= 3.14517282716) {
                                                classes[0] = 0; 
                                                classes[1] = 0; 
                                                classes[2] = 7; 
                                            } else {
                                                if (features[0] <= 6.10520981558) {
                                                    classes[0] = 0; 
                                                    classes[1] = 1; 
                                                    classes[2] = 0; 
                                                } else {
                                                    classes[0] = 0; 
                                                    classes[1] = 0; 
                                                    classes[2] = 1; 
                                                }
                                            }
                                        } else {
                                            classes[0] = 0; 
                                            classes[1] = 1; 
                                            classes[2] = 0; 
                                        }
                                    }
                                }
                            } else {
                                classes[0] = 0; 
                                classes[1] = 0; 
                                classes[2] = 5; 
                            }
                        }
                    }
                } else {
                    classes[0] = 0; 
                    classes[1] = 0; 
                    classes[2] = 11; 
                }
            } else {
                classes[0] = 0; 
                classes[1] = 0; 
                classes[2] = 14; 
            }
        }
        int class_idx = 0;
        int class_val = classes[0];
        for (int i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    }
    
    public static int predict_9(double[] features) {
        int[] classes = new int[3];
        
        if (features[2] <= 1.79847512721) {
            classes[0] = 48; 
            classes[1] = 0; 
            classes[2] = 0; 
        } else {
            if (features[2] <= 2.74361452509) {
                classes[0] = 2; 
                classes[1] = 0; 
                classes[2] = 0; 
            } else {
                if (features[3] <= 1.57262799357) {
                    if (features[1] <= 2.85801286757) {
                        if (features[3] <= 1.00717250817) {
                            classes[0] = 0; 
                            classes[1] = 7; 
                            classes[2] = 0; 
                        } else {
                            if (features[3] <= 1.45085963469) {
                                if (features[0] <= 5.47884707116) {
                                    classes[0] = 0; 
                                    classes[1] = 2; 
                                    classes[2] = 0; 
                                } else {
                                    if (features[2] <= 4.31053268922) {
                                        classes[0] = 0; 
                                        classes[1] = 9; 
                                        classes[2] = 0; 
                                    } else {
                                        if (features[3] <= 1.25101182855) {
                                            classes[0] = 0; 
                                            classes[1] = 2; 
                                            classes[2] = 0; 
                                        } else {
                                            if (features[3] <= 1.30080715959) {
                                                classes[0] = 0; 
                                                classes[1] = 2; 
                                                classes[2] = 0; 
                                            } else {
                                                if (features[1] <= 2.61829279316) {
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
                                    }
                                }
                            } else {
                                if (features[2] <= 5.03297563755) {
                                    if (features[2] <= 4.94728199763) {
                                        classes[0] = 0; 
                                        classes[1] = 3; 
                                        classes[2] = 0; 
                                    } else {
                                        classes[0] = 0; 
                                        classes[1] = 0; 
                                        classes[2] = 1; 
                                    }
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 1; 
                                }
                            }
                        }
                    } else {
                        classes[0] = 0; 
                        classes[1] = 19; 
                        classes[2] = 0; 
                    }
                } else {
                    if (features[2] <= 6.1966904607) {
                        if (features[2] <= 5.71454934224) {
                            if (features[3] <= 1.74839534387) {
                                if (features[0] <= 5.12442503694) {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 1; 
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 4; 
                                    classes[2] = 0; 
                                }
                            } else {
                                if (features[0] <= 5.96352456831) {
                                    if (features[1] <= 2.94031347369) {
                                        classes[0] = 0; 
                                        classes[1] = 0; 
                                        classes[2] = 5; 
                                    } else {
                                        if (features[1] <= 3.15975956747) {
                                            classes[0] = 0; 
                                            classes[1] = 0; 
                                            classes[2] = 1; 
                                        } else {
                                            classes[0] = 0; 
                                            classes[1] = 1; 
                                            classes[2] = 0; 
                                        }
                                    }
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 24; 
                                }
                            }
                        } else {
                            classes[0] = 0; 
                            classes[1] = 0; 
                            classes[2] = 10; 
                        }
                    } else {
                        classes[0] = 0; 
                        classes[1] = 0; 
                        classes[2] = 6; 
                    }
                }
            }
        }
        int class_idx = 0;
        int class_val = classes[0];
        for (int i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    }
    
    public static int predict_10(double[] features) {
        int[] classes = new int[3];
        
        if (features[0] <= 4.97928031983) {
            if (features[1] <= 2.4799516419) {
                if (features[2] <= 2.10811895449) {
                    classes[0] = 1; 
                    classes[1] = 0; 
                    classes[2] = 0; 
                } else {
                    classes[0] = 0; 
                    classes[1] = 1; 
                    classes[2] = 0; 
                }
            } else {
                if (features[2] <= 2.30318907732) {
                    classes[0] = 19; 
                    classes[1] = 0; 
                    classes[2] = 0; 
                } else {
                    classes[0] = 0; 
                    classes[1] = 0; 
                    classes[2] = 1; 
                }
            }
        } else {
            if (features[2] <= 4.05669012958) {
                if (features[2] <= 3.86459559464) {
                    if (features[1] <= 2.44935583863) {
                        classes[0] = 0; 
                        classes[1] = 4; 
                        classes[2] = 0; 
                    } else {
                        if (features[1] <= 3.10137020947) {
                            if (features[0] <= 5.41826741368) {
                                if (features[1] <= 2.7351736651) {
                                    classes[0] = 0; 
                                    classes[1] = 1; 
                                    classes[2] = 0; 
                                } else {
                                    classes[0] = 1; 
                                    classes[1] = 0; 
                                    classes[2] = 0; 
                                }
                            } else {
                                classes[0] = 0; 
                                classes[1] = 2; 
                                classes[2] = 0; 
                            }
                        } else {
                            classes[0] = 29; 
                            classes[1] = 0; 
                            classes[2] = 0; 
                        }
                    }
                } else {
                    classes[0] = 0; 
                    classes[1] = 8; 
                    classes[2] = 0; 
                }
            } else {
                if (features[3] <= 2.45003233615) {
                    if (features[0] <= 7.23812963503) {
                        if (features[2] <= 4.98558902729) {
                            if (features[3] <= 1.41751850788) {
                                classes[0] = 0; 
                                classes[1] = 19; 
                                classes[2] = 0; 
                            } else {
                                if (features[3] <= 1.90854828312) {
                                    if (features[3] <= 1.50772268259) {
                                        classes[0] = 0; 
                                        classes[1] = 10; 
                                        classes[2] = 0; 
                                    } else {
                                        if (features[1] <= 2.88437084442) {
                                            classes[0] = 0; 
                                            classes[1] = 0; 
                                            classes[2] = 2; 
                                        } else {
                                            if (features[3] <= 1.75599265637) {
                                                classes[0] = 0; 
                                                classes[1] = 2; 
                                                classes[2] = 0; 
                                            } else {
                                                if (features[1] <= 3.08777646964) {
                                                    classes[0] = 0; 
                                                    classes[1] = 0; 
                                                    classes[2] = 2; 
                                                } else {
                                                    classes[0] = 0; 
                                                    classes[1] = 1; 
                                                    classes[2] = 0; 
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 1; 
                                }
                            }
                        } else {
                            if (features[3] <= 1.99781450659) {
                                if (features[3] <= 1.78657603468) {
                                    if (features[2] <= 5.03111292495) {
                                        if (features[0] <= 6.33527863764) {
                                            classes[0] = 0; 
                                            classes[1] = 0; 
                                            classes[2] = 1; 
                                        } else {
                                            classes[0] = 0; 
                                            classes[1] = 1; 
                                            classes[2] = 0; 
                                        }
                                    } else {
                                        if (features[3] <= 1.50683362438) {
                                            classes[0] = 0; 
                                            classes[1] = 0; 
                                            classes[2] = 2; 
                                        } else {
                                            if (features[2] <= 5.78742812572) {
                                                classes[0] = 0; 
                                                classes[1] = 1; 
                                                classes[2] = 0; 
                                            } else {
                                                classes[0] = 0; 
                                                classes[1] = 0; 
                                                classes[2] = 1; 
                                            }
                                        }
                                    }
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 10; 
                                }
                            } else {
                                classes[0] = 0; 
                                classes[1] = 0; 
                                classes[2] = 19; 
                            }
                        }
                    } else {
                        classes[0] = 0; 
                        classes[1] = 0; 
                        classes[2] = 8; 
                    }
                } else {
                    classes[0] = 0; 
                    classes[1] = 0; 
                    classes[2] = 3; 
                }
            }
        }
        int class_idx = 0;
        int class_val = classes[0];
        for (int i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    }
    
    public static int predict_11(double[] features) {
        int[] classes = new int[3];
        
        if (features[1] <= 3.17695713938) {
            if (features[3] <= 0.653710134741) {
                classes[0] = 13; 
                classes[1] = 0; 
                classes[2] = 0; 
            } else {
                if (features[3] <= 1.84692143776) {
                    if (features[2] <= 4.72083714884) {
                        if (features[0] <= 5.76277296868) {
                            if (features[3] <= 1.63603445654) {
                                classes[0] = 0; 
                                classes[1] = 21; 
                                classes[2] = 0; 
                            } else {
                                classes[0] = 0; 
                                classes[1] = 0; 
                                classes[2] = 1; 
                            }
                        } else {
                            classes[0] = 0; 
                            classes[1] = 19; 
                            classes[2] = 0; 
                        }
                    } else {
                        if (features[3] <= 1.72083445585) {
                            if (features[2] <= 4.99717962611) {
                                classes[0] = 0; 
                                classes[1] = 3; 
                                classes[2] = 0; 
                            } else {
                                if (features[0] <= 7.1981562798) {
                                    if (features[2] <= 5.2574357164) {
                                        if (features[0] <= 6.43674473859) {
                                            if (features[0] <= 6.29707190604) {
                                                if (features[3] <= 1.56817332761) {
                                                    classes[0] = 0; 
                                                    classes[1] = 0; 
                                                    classes[2] = 1; 
                                                } else {
                                                    classes[0] = 0; 
                                                    classes[1] = 1; 
                                                    classes[2] = 0; 
                                                }
                                            } else {
                                                classes[0] = 0; 
                                                classes[1] = 0; 
                                                classes[2] = 1; 
                                            }
                                        } else {
                                            classes[0] = 0; 
                                            classes[1] = 1; 
                                            classes[2] = 0; 
                                        }
                                    } else {
                                        classes[0] = 0; 
                                        classes[1] = 0; 
                                        classes[2] = 1; 
                                    }
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 1; 
                                }
                            }
                        } else {
                            classes[0] = 0; 
                            classes[1] = 0; 
                            classes[2] = 10; 
                        }
                    }
                } else {
                    classes[0] = 0; 
                    classes[1] = 0; 
                    classes[2] = 22; 
                }
            }
        } else {
            if (features[2] <= 3.80477503848) {
                classes[0] = 37; 
                classes[1] = 0; 
                classes[2] = 0; 
            } else {
                if (features[3] <= 2.17932182111) {
                    if (features[2] <= 5.27717916842) {
                        if (features[3] <= 1.85421629204) {
                            classes[0] = 0; 
                            classes[1] = 5; 
                            classes[2] = 0; 
                        } else {
                            classes[0] = 0; 
                            classes[1] = 0; 
                            classes[2] = 1; 
                        }
                    } else {
                        classes[0] = 0; 
                        classes[1] = 0; 
                        classes[2] = 3; 
                    }
                } else {
                    classes[0] = 0; 
                    classes[1] = 0; 
                    classes[2] = 9; 
                }
            }
        }
        int class_idx = 0;
        int class_val = classes[0];
        for (int i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    }
    
    public static int predict_12(double[] features) {
        int[] classes = new int[3];
        
        if (features[3] <= 1.70211286632) {
            if (features[1] <= 2.98496185999) {
                if (features[3] <= 1.49709481667) {
                    if (features[0] <= 5.62314517759) {
                        if (features[0] <= 4.77902952181) {
                            classes[0] = 2; 
                            classes[1] = 0; 
                            classes[2] = 0; 
                        } else {
                            classes[0] = 0; 
                            classes[1] = 13; 
                            classes[2] = 0; 
                        }
                    } else {
                        if (features[3] <= 1.32841182532) {
                            classes[0] = 0; 
                            classes[1] = 14; 
                            classes[2] = 0; 
                        } else {
                            if (features[1] <= 2.76291002328) {
                                classes[0] = 0; 
                                classes[1] = 0; 
                                classes[2] = 1; 
                            } else {
                                classes[0] = 0; 
                                classes[1] = 2; 
                                classes[2] = 0; 
                            }
                        }
                    }
                } else {
                    if (features[3] <= 1.63764923579) {
                        if (features[2] <= 4.60214248283) {
                            classes[0] = 0; 
                            classes[1] = 3; 
                            classes[2] = 0; 
                        } else {
                            if (features[1] <= 2.72195609415) {
                                if (features[0] <= 6.08809298094) {
                                    if (features[3] <= 1.54371566988) {
                                        classes[0] = 0; 
                                        classes[1] = 0; 
                                        classes[2] = 1; 
                                    } else {
                                        classes[0] = 0; 
                                        classes[1] = 1; 
                                        classes[2] = 0; 
                                    }
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 1; 
                                    classes[2] = 0; 
                                }
                            } else {
                                classes[0] = 0; 
                                classes[1] = 0; 
                                classes[2] = 1; 
                            }
                        }
                    } else {
                        classes[0] = 0; 
                        classes[1] = 0; 
                        classes[2] = 1; 
                    }
                }
            } else {
                if (features[3] <= 0.656647500335) {
                    classes[0] = 48; 
                    classes[1] = 0; 
                    classes[2] = 0; 
                } else {
                    if (features[2] <= 5.37976942994) {
                        classes[0] = 0; 
                        classes[1] = 15; 
                        classes[2] = 0; 
                    } else {
                        classes[0] = 0; 
                        classes[1] = 0; 
                        classes[2] = 1; 
                    }
                }
            }
        } else {
            if (features[2] <= 5.29611100387) {
                if (features[2] <= 4.81921486643) {
                    if (features[0] <= 6.07960868768) {
                        if (features[1] <= 3.17995715408) {
                            classes[0] = 0; 
                            classes[1] = 0; 
                            classes[2] = 1; 
                        } else {
                            classes[0] = 0; 
                            classes[1] = 1; 
                            classes[2] = 0; 
                        }
                    } else {
                        classes[0] = 0; 
                        classes[1] = 0; 
                        classes[2] = 1; 
                    }
                } else {
                    classes[0] = 0; 
                    classes[1] = 0; 
                    classes[2] = 13; 
                }
            } else {
                classes[0] = 0; 
                classes[1] = 0; 
                classes[2] = 30; 
            }
        }
        int class_idx = 0;
        int class_val = classes[0];
        for (int i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    }
    
    public static int predict_13(double[] features) {
        int[] classes = new int[3];
        
        if (features[0] <= 7.51411419807) {
            if (features[1] <= 2.71537228011) {
                if (features[0] <= 5.04674662545) {
                    if (features[0] <= 4.72004178107) {
                        classes[0] = 1; 
                        classes[1] = 0; 
                        classes[2] = 0; 
                    } else {
                        if (features[2] <= 3.80689917177) {
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
                    if (features[2] <= 4.90430628847) {
                        if (features[2] <= 4.62357554358) {
                            classes[0] = 0; 
                            classes[1] = 16; 
                            classes[2] = 0; 
                        } else {
                            if (features[3] <= 1.78692055175) {
                                classes[0] = 0; 
                                classes[1] = 1; 
                                classes[2] = 0; 
                            } else {
                                classes[0] = 0; 
                                classes[1] = 0; 
                                classes[2] = 1; 
                            }
                        }
                    } else {
                        if (features[0] <= 5.90348468451) {
                            classes[0] = 0; 
                            classes[1] = 0; 
                            classes[2] = 3; 
                        } else {
                            if (features[0] <= 6.00699647884) {
                                if (features[2] <= 5.09235664734) {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 1; 
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 1; 
                                    classes[2] = 0; 
                                }
                            } else {
                                classes[0] = 0; 
                                classes[1] = 0; 
                                classes[2] = 4; 
                            }
                        }
                    }
                }
            } else {
                if (features[2] <= 3.90287513279) {
                    if (features[3] <= 0.929422172271) {
                        classes[0] = 49; 
                        classes[1] = 0; 
                        classes[2] = 0; 
                    } else {
                        classes[0] = 0; 
                        classes[1] = 1; 
                        classes[2] = 0; 
                    }
                } else {
                    if (features[3] <= 1.95141861134) {
                        if (features[2] <= 5.92028195024) {
                            if (features[1] <= 3.03375967057) {
                                if (features[1] <= 2.951577879) {
                                    if (features[3] <= 1.56513112669) {
                                        if (features[3] <= 1.41060222756) {
                                            classes[0] = 0; 
                                            classes[1] = 10; 
                                            classes[2] = 0; 
                                        } else {
                                            if (features[1] <= 2.82396918971) {
                                                if (features[0] <= 6.37404161079) {
                                                    classes[0] = 0; 
                                                    classes[1] = 0; 
                                                    classes[2] = 1; 
                                                } else {
                                                    classes[0] = 0; 
                                                    classes[1] = 1; 
                                                    classes[2] = 0; 
                                                }
                                            } else {
                                                classes[0] = 0; 
                                                classes[1] = 1; 
                                                classes[2] = 0; 
                                            }
                                        }
                                    } else {
                                        classes[0] = 0; 
                                        classes[1] = 0; 
                                        classes[2] = 2; 
                                    }
                                } else {
                                    if (features[3] <= 1.73493299091) {
                                        if (features[2] <= 5.50738899496) {
                                            classes[0] = 0; 
                                            classes[1] = 8; 
                                            classes[2] = 0; 
                                        } else {
                                            classes[0] = 0; 
                                            classes[1] = 0; 
                                            classes[2] = 1; 
                                        }
                                    } else {
                                        classes[0] = 0; 
                                        classes[1] = 0; 
                                        classes[2] = 4; 
                                    }
                                }
                            } else {
                                if (features[1] <= 3.26679073963) {
                                    if (features[0] <= 6.85103319003) {
                                        if (features[2] <= 5.30808415322) {
                                            classes[0] = 0; 
                                            classes[1] = 4; 
                                            classes[2] = 0; 
                                        } else {
                                            classes[0] = 0; 
                                            classes[1] = 0; 
                                            classes[2] = 1; 
                                        }
                                    } else {
                                        classes[0] = 0; 
                                        classes[1] = 2; 
                                        classes[2] = 0; 
                                    }
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 2; 
                                    classes[2] = 0; 
                                }
                            }
                        } else {
                            classes[0] = 0; 
                            classes[1] = 0; 
                            classes[2] = 3; 
                        }
                    } else {
                        classes[0] = 0; 
                        classes[1] = 0; 
                        classes[2] = 22; 
                    }
                }
            }
        } else {
            classes[0] = 0; 
            classes[1] = 0; 
            classes[2] = 6; 
        }
        int class_idx = 0;
        int class_val = classes[0];
        for (int i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    }
    
    public static int predict_14(double[] features) {
        int[] classes = new int[3];
        
        if (features[2] <= 1.52136814805) {
            classes[0] = 37; 
            classes[1] = 0; 
            classes[2] = 0; 
        } else {
            if (features[2] <= 3.37330671524) {
                if (features[2] <= 1.92848321006) {
                    classes[0] = 13; 
                    classes[1] = 0; 
                    classes[2] = 0; 
                } else {
                    classes[0] = 0; 
                    classes[1] = 3; 
                    classes[2] = 0; 
                }
            } else {
                if (features[3] <= 1.42038445124) {
                    if (features[0] <= 5.27180029744) {
                        classes[0] = 0; 
                        classes[1] = 2; 
                        classes[2] = 0; 
                    } else {
                        if (features[3] <= 1.39844428031) {
                            classes[0] = 0; 
                            classes[1] = 24; 
                            classes[2] = 0; 
                        } else {
                            if (features[1] <= 2.81581969015) {
                                if (features[0] <= 6.57526062858) {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 1; 
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 1; 
                                    classes[2] = 0; 
                                }
                            } else {
                                classes[0] = 0; 
                                classes[1] = 5; 
                                classes[2] = 0; 
                            }
                        }
                    }
                } else {
                    if (features[2] <= 5.073323223) {
                        if (features[2] <= 4.72772830442) {
                            if (features[3] <= 1.68309830787) {
                                classes[0] = 0; 
                                classes[1] = 10; 
                                classes[2] = 0; 
                            } else {
                                classes[0] = 0; 
                                classes[1] = 0; 
                                classes[2] = 1; 
                            }
                        } else {
                            if (features[3] <= 1.79554005679) {
                                if (features[2] <= 4.92263211048) {
                                    classes[0] = 0; 
                                    classes[1] = 2; 
                                    classes[2] = 0; 
                                } else {
                                    if (features[1] <= 2.54751616598) {
                                        classes[0] = 0; 
                                        classes[1] = 0; 
                                        classes[2] = 1; 
                                    } else {
                                        classes[0] = 0; 
                                        classes[1] = 1; 
                                        classes[2] = 0; 
                                    }
                                }
                            } else {
                                if (features[1] <= 3.03028433439) {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 7; 
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 1; 
                                    classes[2] = 0; 
                                }
                            }
                        }
                    } else {
                        if (features[1] <= 3.1109665641) {
                            if (features[3] <= 1.87377453373) {
                                if (features[0] <= 6.17584005474) {
                                    if (features[0] <= 5.93320851071) {
                                        classes[0] = 0; 
                                        classes[1] = 0; 
                                        classes[2] = 1; 
                                    } else {
                                        classes[0] = 0; 
                                        classes[1] = 1; 
                                        classes[2] = 0; 
                                    }
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 7; 
                                }
                            } else {
                                classes[0] = 0; 
                                classes[1] = 0; 
                                classes[2] = 19; 
                            }
                        } else {
                            classes[0] = 0; 
                            classes[1] = 0; 
                            classes[2] = 13; 
                        }
                    }
                }
            }
        }
        int class_idx = 0;
        int class_val = classes[0];
        for (int i = 1; i < 3; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    }
    
    public static int predict(double[] features) {
        int n_classes = 3;
        int[] classes = new int[n_classes];
        classes[ExtraTreesClassifier.predict_0(features)]++;
        classes[ExtraTreesClassifier.predict_1(features)]++;
        classes[ExtraTreesClassifier.predict_2(features)]++;
        classes[ExtraTreesClassifier.predict_3(features)]++;
        classes[ExtraTreesClassifier.predict_4(features)]++;
        classes[ExtraTreesClassifier.predict_5(features)]++;
        classes[ExtraTreesClassifier.predict_6(features)]++;
        classes[ExtraTreesClassifier.predict_7(features)]++;
        classes[ExtraTreesClassifier.predict_8(features)]++;
        classes[ExtraTreesClassifier.predict_9(features)]++;
        classes[ExtraTreesClassifier.predict_10(features)]++;
        classes[ExtraTreesClassifier.predict_11(features)]++;
        classes[ExtraTreesClassifier.predict_12(features)]++;
        classes[ExtraTreesClassifier.predict_13(features)]++;
        classes[ExtraTreesClassifier.predict_14(features)]++;
    
        int class_idx = 0;
        int class_val = classes[0];
        for (int i = 1; i < n_classes; i++) {
            if (classes[i] > class_val) {
                class_idx = i;
                class_val = classes[i];
            }
        }
        return class_idx;
    }

    public static void main(String[] args) {
        if (args.length == 4) {

            // Features:
            double[] features = new double[args.length];
            for (int i = 0, l = args.length; i < l; i++) {
                features[i] = Double.parseDouble(args[i]);
            }

            // Prediction:
            int prediction = ExtraTreesClassifier.predict(features);
            System.out.println(prediction);

        }
    }
}
"""
