from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier

from onl.nok.sklearn.Porter import port

iris = load_iris()
clf = ExtraTreesClassifier(n_estimators=15, random_state=0)
clf.fit(iris.data, iris.target)

# Cheese!

print(port(clf))

# class Tmp {
#     public static int predict_00(float[] atts) {
#         int[] classes = new int[3];
#         if (atts[0] <= 5.5211283675173766) {
#             if (atts[1] <= 2.5423691467210121) {
#                 if (atts[2] <= 2.3285940238943343) {
#                     classes[0] = 1;
#                     classes[1] = 0;
#                     classes[2] = 0;
#                 } else {
#                     if (atts[3] <= 1.4511234333012406) {
#                         classes[0] = 0;
#                         classes[1] = 8;
#                         classes[2] = 0;
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 1;
#                     }
#                 }
#             } else {
#                 if (atts[3] <= 0.83878367200965764) {
#                     classes[0] = 46;
#                     classes[1] = 0;
#                     classes[2] = 0;
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 3;
#                     classes[2] = 0;
#                 }
#             }
#         } else {
#             if (atts[3] <= 1.8936930649824477) {
#                 if (atts[2] <= 5.1807613674845001) {
#                     if (atts[3] <= 0.2542593119501258) {
#                         classes[0] = 1;
#                         classes[1] = 0;
#                         classes[2] = 0;
#                     } else {
#                         if (atts[3] <= 1.2979109919829468) {
#                             if (atts[1] <= 3.1466755002784144) {
#                                 classes[0] = 0;
#                                 classes[1] = 8;
#                                 classes[2] = 0;
#                             } else {
#                                 classes[0] = 2;
#                                 classes[1] = 0;
#                                 classes[2] = 0;
#                             }
#                         } else {
#                             if (atts[3] <= 1.4917742899580739) {
#                                 classes[0] = 0;
#                                 classes[1] = 17;
#                                 classes[2] = 0;
#                             } else {
#                                 if (atts[2] <= 4.6486109417083465) {
#                                     classes[0] = 0;
#                                     classes[1] = 7;
#                                     classes[2] = 0;
#                                 } else {
#                                     if (atts[3] <= 1.7802776400056488) {
#                                         if (atts[3] <= 1.5065926898187254) {
#                                             if (atts[2] <= 4.899703297413823) {
#                                                 classes[0] = 0;
#                                                 classes[1] = 1;
#                                                 classes[2] = 0;
#                                             } else {
#                                                 if (atts[2] <= 4.9878533508461969) {
#                                                     classes[0] = 0;
#                                                     classes[1] = 2;
#                                                     classes[2] = 0;
#                                                 } else {
#                                                     classes[0] = 0;
#                                                     classes[1] = 0;
#                                                     classes[2] = 2;
#                                                 }
#                                             }
#                                         } else {
#                                             classes[0] = 0;
#                                             classes[1] = 3;
#                                             classes[2] = 0;
#                                         }
#                                     } else {
#                                         if (atts[1] <= 3.1244712178243654) {
#                                             classes[0] = 0;
#                                             classes[1] = 0;
#                                             classes[2] = 5;
#                                         } else {
#                                             classes[0] = 0;
#                                             classes[1] = 1;
#                                             classes[2] = 0;
#                                         }
#                                     }
#                                 }
#                             }
#                         }
#                     }
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 0;
#                     classes[2] = 8;
#                 }
#             } else {
#                 classes[0] = 0;
#                 classes[1] = 0;
#                 classes[2] = 34;
#             }
#         }
#         int class_idx = 0;
#         int class_val = classes[0];
#         for (int i = 1; i < 3; i++) {
#             if (classes[i] > class_val) {
#                 class_idx = i;
#                 class_val = classes[i];
#             }
#         }
#         return class_idx;
#     }
#
#     public static int predict_01(float[] atts) {
#         int[] classes = new int[3];
#         if (atts[3] <= 1.5504955501838129) {
#             if (atts[2] <= 2.5923285292922946) {
#                 classes[0] = 50;
#                 classes[1] = 0;
#                 classes[2] = 0;
#             } else {
#                 if (atts[0] <= 5.7051897734310577) {
#                     classes[0] = 0;
#                     classes[1] = 21;
#                     classes[2] = 0;
#                 } else {
#                     if (atts[1] <= 2.4211563618020882) {
#                         if (atts[1] <= 2.2376366618358423) {
#                             if (atts[3] <= 1.0852253511944903) {
#                                 classes[0] = 0;
#                                 classes[1] = 1;
#                                 classes[2] = 0;
#                             } else {
#                                 if (atts[2] <= 4.5587103432317777) {
#                                     classes[0] = 0;
#                                     classes[1] = 1;
#                                     classes[2] = 0;
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 1;
#                                 }
#                             }
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 1;
#                             classes[2] = 0;
#                         }
#                     } else {
#                         if (atts[1] <= 2.8877833170885867) {
#                             if (atts[3] <= 1.2729508337904472) {
#                                 classes[0] = 0;
#                                 classes[1] = 4;
#                                 classes[2] = 0;
#                             } else {
#                                 if (atts[0] <= 6.5790113411443398) {
#                                     if (atts[2] <= 4.7893806273809885) {
#                                         classes[0] = 0;
#                                         classes[1] = 2;
#                                         classes[2] = 0;
#                                     } else {
#                                         if (atts[0] <= 6.1217669667164261) {
#                                             classes[0] = 0;
#                                             classes[1] = 0;
#                                             classes[2] = 1;
#                                         } else {
#                                             if (atts[2] <= 4.9162291827513958) {
#                                                 classes[0] = 0;
#                                                 classes[1] = 1;
#                                                 classes[2] = 0;
#                                             } else {
#                                                 classes[0] = 0;
#                                                 classes[1] = 0;
#                                                 classes[2] = 1;
#                                             }
#                                         }
#                                     }
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 1;
#                                     classes[2] = 0;
#                                 }
#                             }
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 13;
#                             classes[2] = 0;
#                         }
#                     }
#                 }
#             }
#         } else {
#             if (atts[1] <= 3.1730407214994258) {
#                 if (atts[3] <= 2.2706219597859412) {
#                     if (atts[0] <= 5.7034701487511237) {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 3;
#                     } else {
#                         if (atts[2] <= 5.6371861017671403) {
#                             if (atts[0] <= 6.61522923341691) {
#                                 if (atts[2] <= 5.5676840774650911) {
#                                     if (atts[1] <= 2.9151389085288946) {
#                                         if (atts[0] <= 6.3676007760812388) {
#                                             if (atts[0] <= 6.2352298441972724) {
#                                                 if (atts[3] <= 1.652702599886597) {
#                                                     classes[0] = 0;
#                                                     classes[1] = 1;
#                                                     classes[2] = 0;
#                                                 } else {
#                                                     classes[0] = 0;
#                                                     classes[1] = 0;
#                                                     classes[2] = 3;
#                                                 }
#                                             } else {
#                                                 classes[0] = 0;
#                                                 classes[1] = 0;
#                                                 classes[2] = 2;
#                                             }
#                                         } else {
#                                             classes[0] = 0;
#                                             classes[1] = 0;
#                                             classes[2] = 1;
#                                         }
#                                     } else {
#                                         classes[0] = 0;
#                                         classes[1] = 0;
#                                         classes[2] = 6;
#                                     }
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 3;
#                                 }
#                             } else {
#                                 if (atts[0] <= 6.7180041929368821) {
#                                     classes[0] = 0;
#                                     classes[1] = 1;
#                                     classes[2] = 0;
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 2;
#                                 }
#                             }
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 0;
#                             classes[2] = 8;
#                         }
#                     }
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 0;
#                     classes[2] = 6;
#                 }
#             } else {
#                 if (atts[3] <= 2.1245702483609095) {
#                     if (atts[3] <= 1.8196133595321136) {
#                         if (atts[0] <= 6.5204934295107302) {
#                             classes[0] = 0;
#                             classes[1] = 3;
#                             classes[2] = 0;
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 0;
#                             classes[2] = 1;
#                         }
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 3;
#                     }
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 0;
#                     classes[2] = 9;
#                 }
#             }
#         }
#         int class_idx = 0;
#         int class_val = classes[0];
#         for (int i = 1; i < 3; i++) {
#             if (classes[i] > class_val) {
#                 class_idx = i;
#                 class_val = classes[i];
#             }
#         }
#         return class_idx;
#     }
#
#     public static int predict_02(float[] atts) {
#         int[] classes = new int[3];
#         if (atts[3] <= 1.5465266340960862) {
#             if (atts[2] <= 4.6071317040414321) {
#                 if (atts[2] <= 3.3003282579766613) {
#                     if (atts[3] <= 0.38658752666916157) {
#                         classes[0] = 41;
#                         classes[1] = 0;
#                         classes[2] = 0;
#                     } else {
#                         if (atts[2] <= 2.1556475054731816) {
#                             classes[0] = 9;
#                             classes[1] = 0;
#                             classes[2] = 0;
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 3;
#                             classes[2] = 0;
#                         }
#                     }
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 35;
#                     classes[2] = 0;
#                 }
#             } else {
#                 if (atts[1] <= 2.269675323772093) {
#                     classes[0] = 0;
#                     classes[1] = 0;
#                     classes[2] = 1;
#                 } else {
#                     if (atts[2] <= 5.4827000154137897) {
#                         if (atts[2] <= 4.9972260125339991) {
#                             classes[0] = 0;
#                             classes[1] = 7;
#                             classes[2] = 0;
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 0;
#                             classes[2] = 1;
#                         }
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 1;
#                     }
#                 }
#             }
#         } else {
#             if (atts[3] <= 1.6032070119961934) {
#                 if (atts[0] <= 6.5187490263266428) {
#                     classes[0] = 0;
#                     classes[1] = 3;
#                     classes[2] = 0;
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 0;
#                     classes[2] = 1;
#                 }
#             } else {
#                 if (atts[3] <= 1.8081436115623295) {
#                     if (atts[1] <= 2.5996223526415174) {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 2;
#                     } else {
#                         if (atts[1] <= 3.0266239479308128) {
#                             if (atts[3] <= 1.7788371596579802) {
#                                 classes[0] = 0;
#                                 classes[1] = 1;
#                                 classes[2] = 0;
#                             } else {
#                                 classes[0] = 0;
#                                 classes[1] = 0;
#                                 classes[2] = 8;
#                             }
#                         } else {
#                             if (atts[2] <= 5.5827159596680156) {
#                                 if (atts[1] <= 3.1452745616104556) {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 1;
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 1;
#                                     classes[2] = 0;
#                                 }
#                             } else {
#                                 classes[0] = 0;
#                                 classes[1] = 0;
#                                 classes[2] = 1;
#                             }
#                         }
#                     }
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 0;
#                     classes[2] = 34;
#                 }
#             }
#         }
#         int class_idx = 0;
#         int class_val = classes[0];
#         for (int i = 1; i < 3; i++) {
#             if (classes[i] > class_val) {
#                 class_idx = i;
#                 class_val = classes[i];
#             }
#         }
#         return class_idx;
#     }
#
#     public static int predict_03(float[] atts) {
#         int[] classes = new int[3];
#         if (atts[0] <= 5.1096510251748901) {
#             if (atts[3] <= 0.34237436990699954) {
#                 classes[0] = 31;
#                 classes[1] = 0;
#                 classes[2] = 0;
#             } else {
#                 if (atts[3] <= 0.42014141078495837) {
#                     classes[0] = 3;
#                     classes[1] = 0;
#                     classes[2] = 0;
#                 } else {
#                     if (atts[1] <= 3.3013719233224972) {
#                         if (atts[2] <= 2.3218753875601132) {
#                             classes[0] = 1;
#                             classes[1] = 0;
#                             classes[2] = 0;
#                         } else {
#                             if (atts[0] <= 4.9582561902467441) {
#                                 if (atts[1] <= 2.4726812725055445) {
#                                     classes[0] = 0;
#                                     classes[1] = 1;
#                                     classes[2] = 0;
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 1;
#                                 }
#                             } else {
#                                 classes[0] = 0;
#                                 classes[1] = 3;
#                                 classes[2] = 0;
#                             }
#                         }
#                     } else {
#                         classes[0] = 1;
#                         classes[1] = 0;
#                         classes[2] = 0;
#                     }
#                 }
#             }
#         } else {
#             if (atts[2] <= 5.0059938824178367) {
#                 if (atts[3] <= 0.59215586323442948) {
#                     classes[0] = 14;
#                     classes[1] = 0;
#                     classes[2] = 0;
#                 } else {
#                     if (atts[2] <= 4.8869158585494921) {
#                         if (atts[2] <= 4.5845956777760826) {
#                             classes[0] = 0;
#                             classes[1] = 32;
#                             classes[2] = 0;
#                         } else {
#                             if (atts[1] <= 3.0013521708088704) {
#                                 if (atts[0] <= 6.2818121532768014) {
#                                     if (atts[3] <= 1.3566014997210807) {
#                                         classes[0] = 0;
#                                         classes[1] = 1;
#                                         classes[2] = 0;
#                                     } else {
#                                         if (atts[3] <= 1.4692521397047558) {
#                                             classes[0] = 0;
#                                             classes[1] = 2;
#                                             classes[2] = 0;
#                                         } else {
#                                             classes[0] = 0;
#                                             classes[1] = 0;
#                                             classes[2] = 2;
#                                         }
#                                     }
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 3;
#                                     classes[2] = 0;
#                                 }
#                             } else {
#                                 classes[0] = 0;
#                                 classes[1] = 4;
#                                 classes[2] = 0;
#                             }
#                         }
#                     } else {
#                         if (atts[0] <= 6.024789407414449) {
#                             classes[0] = 0;
#                             classes[1] = 0;
#                             classes[2] = 3;
#                         } else {
#                             if (atts[3] <= 1.5357312864913262) {
#                                 classes[0] = 0;
#                                 classes[1] = 2;
#                                 classes[2] = 0;
#                             } else {
#                                 if (atts[1] <= 2.5827080563561564) {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 1;
#                                 } else {
#                                     if (atts[0] <= 6.5669079160763015) {
#                                         classes[0] = 0;
#                                         classes[1] = 0;
#                                         classes[2] = 2;
#                                     } else {
#                                         classes[0] = 0;
#                                         classes[1] = 1;
#                                         classes[2] = 0;
#                                     }
#                                 }
#                             }
#                         }
#                     }
#                 }
#             } else {
#                 if (atts[2] <= 5.1863245386304202) {
#                     if (atts[1] <= 2.7092082675123628) {
#                         if (atts[3] <= 1.8776005740851462) {
#                             classes[0] = 0;
#                             classes[1] = 1;
#                             classes[2] = 0;
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 0;
#                             classes[2] = 2;
#                         }
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 5;
#                     }
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 0;
#                     classes[2] = 34;
#                 }
#             }
#         }
#         int class_idx = 0;
#         int class_val = classes[0];
#         for (int i = 1; i < 3; i++) {
#             if (classes[i] > class_val) {
#                 class_idx = i;
#                 class_val = classes[i];
#             }
#         }
#         return class_idx;
#     }
#
#     public static int predict_04(float[] atts) {
#         int[] classes = new int[3];
#         if (atts[3] <= 1.3441161325132915) {
#             if (atts[1] <= 3.0101406622719056) {
#                 if (atts[0] <= 5.5513347352769937) {
#                     if (atts[1] <= 2.4504689497176879) {
#                         if (atts[0] <= 5.3493399647294266) {
#                             if (atts[0] <= 4.8750152915599827) {
#                                 classes[0] = 1;
#                                 classes[1] = 0;
#                                 classes[2] = 0;
#                             } else {
#                                 classes[0] = 0;
#                                 classes[1] = 3;
#                                 classes[2] = 0;
#                             }
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 3;
#                             classes[2] = 0;
#                         }
#                     } else {
#                         if (atts[1] <= 2.8545732504476669) {
#                             classes[0] = 0;
#                             classes[1] = 3;
#                             classes[2] = 0;
#                         } else {
#                             classes[0] = 7;
#                             classes[1] = 0;
#                             classes[2] = 0;
#                         }
#                     }
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 19;
#                     classes[2] = 0;
#                 }
#             } else {
#                 classes[0] = 42;
#                 classes[1] = 0;
#                 classes[2] = 0;
#             }
#         } else {
#             if (atts[2] <= 4.5412213265156476) {
#                 if (atts[3] <= 1.6796973785314397) {
#                     classes[0] = 0;
#                     classes[1] = 10;
#                     classes[2] = 0;
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 0;
#                     classes[2] = 1;
#                 }
#             } else {
#                 if (atts[3] <= 1.8681215891414575) {
#                     if (atts[3] <= 1.7082770221162646) {
#                         if (atts[2] <= 5.4587785419790711) {
#                             if (atts[1] <= 2.6322218231672321) {
#                                 if (atts[1] <= 2.2075641354665572) {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 1;
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 1;
#                                     classes[2] = 0;
#                                 }
#                             } else {
#                                 if (atts[1] <= 2.9750656646611109) {
#                                     if (atts[0] <= 6.5700013628278997) {
#                                         if (atts[3] <= 1.4378231510535273) {
#                                             classes[0] = 0;
#                                             classes[1] = 1;
#                                             classes[2] = 0;
#                                         } else {
#                                             if (atts[3] <= 1.5344588928285092) {
#                                                 if (atts[0] <= 6.46420655970892) {
#                                                     classes[0] = 0;
#                                                     classes[1] = 0;
#                                                     classes[2] = 1;
#                                                 } else {
#                                                     classes[0] = 0;
#                                                     classes[1] = 1;
#                                                     classes[2] = 0;
#                                                 }
#                                             } else {
#                                                 classes[0] = 0;
#                                                 classes[1] = 1;
#                                                 classes[2] = 0;
#                                             }
#                                         }
#                                     } else {
#                                         classes[0] = 0;
#                                         classes[1] = 1;
#                                         classes[2] = 0;
#                                     }
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 6;
#                                     classes[2] = 0;
#                                 }
#                             }
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 0;
#                             classes[2] = 2;
#                         }
#                     } else {
#                         if (atts[1] <= 2.8407804212030414) {
#                             classes[0] = 0;
#                             classes[1] = 0;
#                             classes[2] = 3;
#                         } else {
#                             if (atts[2] <= 6.150836971984635) {
#                                 if (atts[1] <= 3.1337532458575437) {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 6;
#                                 } else {
#                                     if (atts[2] <= 5.8292593574657765) {
#                                         classes[0] = 0;
#                                         classes[1] = 1;
#                                         classes[2] = 0;
#                                     } else {
#                                         classes[0] = 0;
#                                         classes[1] = 0;
#                                         classes[2] = 1;
#                                     }
#                                 }
#                             } else {
#                                 classes[0] = 0;
#                                 classes[1] = 0;
#                                 classes[2] = 1;
#                             }
#                         }
#                     }
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 0;
#                     classes[2] = 34;
#                 }
#             }
#         }
#         int class_idx = 0;
#         int class_val = classes[0];
#         for (int i = 1; i < 3; i++) {
#             if (classes[i] > class_val) {
#                 class_idx = i;
#                 class_val = classes[i];
#             }
#         }
#         return class_idx;
#     }
#
#     public static int predict_05(float[] atts) {
#         int[] classes = new int[3];
#         if (atts[3] <= 1.7676213332657358) {
#             if (atts[1] <= 3.4545610617739615) {
#                 if (atts[2] <= 3.6502854621611109) {
#                     if (atts[2] <= 3.1891586420402427) {
#                         if (atts[3] <= 0.95101981363257859) {
#                             classes[0] = 29;
#                             classes[1] = 0;
#                             classes[2] = 0;
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 1;
#                             classes[2] = 0;
#                         }
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 5;
#                         classes[2] = 0;
#                     }
#                 } else {
#                     if (atts[0] <= 5.1755356442982308) {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 1;
#                     } else {
#                         if (atts[3] <= 1.4533850152404659) {
#                             if (atts[1] <= 2.6607527474224346) {
#                                 if (atts[3] <= 1.1197721207462903) {
#                                     classes[0] = 0;
#                                     classes[1] = 4;
#                                     classes[2] = 0;
#                                 } else {
#                                     if (atts[2] <= 5.0340535096243002) {
#                                         classes[0] = 0;
#                                         classes[1] = 5;
#                                         classes[2] = 0;
#                                     } else {
#                                         classes[0] = 0;
#                                         classes[1] = 0;
#                                         classes[2] = 1;
#                                     }
#                                 }
#                             } else {
#                                 classes[0] = 0;
#                                 classes[1] = 20;
#                                 classes[2] = 0;
#                             }
#                         } else {
#                             if (atts[1] <= 3.2000655596791701) {
#                                 if (atts[1] <= 2.648931143548773) {
#                                     if (atts[0] <= 6.0403355971033852) {
#                                         classes[0] = 0;
#                                         classes[1] = 0;
#                                         classes[2] = 1;
#                                     } else {
#                                         classes[0] = 0;
#                                         classes[1] = 2;
#                                         classes[2] = 0;
#                                     }
#                                 } else {
#                                     if (atts[0] <= 6.5033636618817878) {
#                                         if (atts[0] <= 5.6045121474787978) {
#                                             classes[0] = 0;
#                                             classes[1] = 2;
#                                             classes[2] = 0;
#                                         } else {
#                                             if (atts[3] <= 1.5280856302044936) {
#                                                 if (atts[0] <= 5.9853353459190419) {
#                                                     classes[0] = 0;
#                                                     classes[1] = 1;
#                                                     classes[2] = 0;
#                                                 } else {
#                                                     if (atts[1] <= 2.8357509785984978) {
#                                                         if (atts[2] <= 4.663137085998029) {
#                                                             classes[0] = 0;
#                                                             classes[1] = 1;
#                                                             classes[2] = 0;
#                                                         } else {
#                                                             classes[0] = 0;
#                                                             classes[1] = 0;
#                                                             classes[2] = 1;
#                                                         }
#                                                     } else {
#                                                         classes[0] = 0;
#                                                         classes[1] = 2;
#                                                         classes[2] = 0;
#                                                     }
#                                                 }
#                                             } else {
#                                                 classes[0] = 0;
#                                                 classes[1] = 1;
#                                                 classes[2] = 0;
#                                             }
#                                         }
#                                     } else {
#                                         if (atts[1] <= 3.0563716510809584) {
#                                             if (atts[3] <= 1.6545352601337104) {
#                                                 classes[0] = 0;
#                                                 classes[1] = 0;
#                                                 classes[2] = 1;
#                                             } else {
#                                                 classes[0] = 0;
#                                                 classes[1] = 1;
#                                                 classes[2] = 0;
#                                             }
#                                         } else {
#                                             classes[0] = 0;
#                                             classes[1] = 2;
#                                             classes[2] = 0;
#                                         }
#                                     }
#                                 }
#                             } else {
#                                 classes[0] = 0;
#                                 classes[1] = 2;
#                                 classes[2] = 0;
#                             }
#                         }
#                     }
#                 }
#             } else {
#                 classes[0] = 21;
#                 classes[1] = 0;
#                 classes[2] = 0;
#             }
#         } else {
#             if (atts[3] <= 1.8121969431622151) {
#                 if (atts[2] <= 5.5628289586225659) {
#                     if (atts[2] <= 4.8118490651150063) {
#                         if (atts[1] <= 3.1684422458572552) {
#                             classes[0] = 0;
#                             classes[1] = 0;
#                             classes[2] = 2;
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 1;
#                             classes[2] = 0;
#                         }
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 5;
#                     }
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 0;
#                     classes[2] = 4;
#                 }
#             } else {
#                 classes[0] = 0;
#                 classes[1] = 0;
#                 classes[2] = 34;
#             }
#         }
#         int class_idx = 0;
#         int class_val = classes[0];
#         for (int i = 1; i < 3; i++) {
#             if (classes[i] > class_val) {
#                 class_idx = i;
#                 class_val = classes[i];
#             }
#         }
#         return class_idx;
#     }
#
#     public static int predict_06(float[] atts) {
#         int[] classes = new int[3];
#         if (atts[3] <= 1.8380720877184731) {
#             if (atts[3] <= 0.94908629972759717) {
#                 classes[0] = 50;
#                 classes[1] = 0;
#                 classes[2] = 0;
#             } else {
#                 if (atts[2] <= 3.9259614145650565) {
#                     classes[0] = 0;
#                     classes[1] = 11;
#                     classes[2] = 0;
#                 } else {
#                     if (atts[0] <= 5.7524397380621881) {
#                         if (atts[0] <= 5.5675846099678861) {
#                             if (atts[3] <= 1.3096158726185634) {
#                                 classes[0] = 0;
#                                 classes[1] = 3;
#                                 classes[2] = 0;
#                             } else {
#                                 if (atts[0] <= 4.9965273461288433) {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 1;
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 1;
#                                     classes[2] = 0;
#                                 }
#                             }
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 7;
#                             classes[2] = 0;
#                         }
#                     } else {
#                         if (atts[3] <= 1.3226210794537889) {
#                             classes[0] = 0;
#                             classes[1] = 9;
#                             classes[2] = 0;
#                         } else {
#                             if (atts[3] <= 1.5331659525110946) {
#                                 if (atts[1] <= 2.9363041020633207) {
#                                     if (atts[1] <= 2.6271371465694107) {
#                                         if (atts[2] <= 5.2132714648801279) {
#                                             if (atts[0] <= 6.1450817798403774) {
#                                                 classes[0] = 0;
#                                                 classes[1] = 0;
#                                                 classes[2] = 1;
#                                             } else {
#                                                 classes[0] = 0;
#                                                 classes[1] = 2;
#                                                 classes[2] = 0;
#                                             }
#                                         } else {
#                                             classes[0] = 0;
#                                             classes[1] = 0;
#                                             classes[2] = 1;
#                                         }
#                                     } else {
#                                         if (atts[0] <= 6.1049023089068344) {
#                                             classes[0] = 0;
#                                             classes[1] = 2;
#                                             classes[2] = 0;
#                                         } else {
#                                             if (atts[3] <= 1.4576934027935942) {
#                                                 classes[0] = 0;
#                                                 classes[1] = 1;
#                                                 classes[2] = 0;
#                                             } else {
#                                                 if (atts[2] <= 5.0702880434553546) {
#                                                     classes[0] = 0;
#                                                     classes[1] = 1;
#                                                     classes[2] = 0;
#                                                 } else {
#                                                     classes[0] = 0;
#                                                     classes[1] = 0;
#                                                     classes[2] = 1;
#                                                 }
#                                             }
#                                         }
#                                     }
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 8;
#                                     classes[2] = 0;
#                                 }
#                             } else {
#                                 if (atts[2] <= 5.1260403107316117) {
#                                     if (atts[1] <= 3.1470031032977439) {
#                                         if (atts[2] <= 4.8079453149979541) {
#                                             classes[0] = 0;
#                                             classes[1] = 0;
#                                             classes[2] = 2;
#                                         } else {
#                                             if (atts[0] <= 6.1519032927184263) {
#                                                 if (atts[0] <= 6.0430469727063381) {
#                                                     if (atts[1] <= 2.9580633267166325) {
#                                                         classes[0] = 0;
#                                                         classes[1] = 1;
#                                                         classes[2] = 0;
#                                                     } else {
#                                                         classes[0] = 0;
#                                                         classes[1] = 0;
#                                                         classes[2] = 1;
#                                                     }
#                                                 } else {
#                                                     classes[0] = 0;
#                                                     classes[1] = 0;
#                                                     classes[2] = 1;
#                                                 }
#                                             } else {
#                                                 if (atts[2] <= 4.9211956019572716) {
#                                                     classes[0] = 0;
#                                                     classes[1] = 0;
#                                                     classes[2] = 1;
#                                                 } else {
#                                                     classes[0] = 0;
#                                                     classes[1] = 1;
#                                                     classes[2] = 0;
#                                                 }
#                                             }
#                                         }
#                                     } else {
#                                         classes[0] = 0;
#                                         classes[1] = 3;
#                                         classes[2] = 0;
#                                     }
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 7;
#                                 }
#                             }
#                         }
#                     }
#                 }
#             }
#         } else {
#             classes[0] = 0;
#             classes[1] = 0;
#             classes[2] = 34;
#         }
#         int class_idx = 0;
#         int class_val = classes[0];
#         for (int i = 1; i < 3; i++) {
#             if (classes[i] > class_val) {
#                 class_idx = i;
#                 class_val = classes[i];
#             }
#         }
#         return class_idx;
#     }
#
#     public static int predict_07(float[] atts) {
#         int[] classes = new int[3];
#         if (atts[0] <= 4.8350779809500537) {
#             classes[0] = 16;
#             classes[1] = 0;
#             classes[2] = 0;
#         } else {
#             if (atts[2] <= 5.0521211317065813) {
#                 if (atts[3] <= 1.187984123199987) {
#                     if (atts[2] <= 2.4198502829763324) {
#                         classes[0] = 34;
#                         classes[1] = 0;
#                         classes[2] = 0;
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 10;
#                         classes[2] = 0;
#                     }
#                 } else {
#                     if (atts[3] <= 1.9700979161064909) {
#                         if (atts[3] <= 1.2103523333737463) {
#                             classes[0] = 0;
#                             classes[1] = 5;
#                             classes[2] = 0;
#                         } else {
#                             if (atts[3] <= 1.7922824027484205) {
#                                 if (atts[3] <= 1.3556012588196729) {
#                                     classes[0] = 0;
#                                     classes[1] = 13;
#                                     classes[2] = 0;
#                                 } else {
#                                     if (atts[1] <= 2.3506367699266271) {
#                                         if (atts[2] <= 4.781047702897828) {
#                                             classes[0] = 0;
#                                             classes[1] = 1;
#                                             classes[2] = 0;
#                                         } else {
#                                             classes[0] = 0;
#                                             classes[1] = 0;
#                                             classes[2] = 1;
#                                         }
#                                     } else {
#                                         if (atts[0] <= 5.2396924972122507) {
#                                             if (atts[3] <= 1.540099250418558) {
#                                                 classes[0] = 0;
#                                                 classes[1] = 1;
#                                                 classes[2] = 0;
#                                             } else {
#                                                 classes[0] = 0;
#                                                 classes[1] = 0;
#                                                 classes[2] = 1;
#                                             }
#                                         } else {
#                                             classes[0] = 0;
#                                             classes[1] = 18;
#                                             classes[2] = 0;
#                                         }
#                                     }
#                                 }
#                             } else {
#                                 if (atts[2] <= 4.8114599180137345) {
#                                     if (atts[0] <= 6.1531159293238975) {
#                                         if (atts[0] <= 5.9178402810269528) {
#                                             classes[0] = 0;
#                                             classes[1] = 1;
#                                             classes[2] = 0;
#                                         } else {
#                                             classes[0] = 0;
#                                             classes[1] = 0;
#                                             classes[2] = 1;
#                                         }
#                                     } else {
#                                         classes[0] = 0;
#                                         classes[1] = 0;
#                                         classes[2] = 1;
#                                     }
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 3;
#                                 }
#                             }
#                         }
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 2;
#                     }
#                 }
#             } else {
#                 if (atts[3] <= 1.9630314654063374) {
#                     if (atts[0] <= 6.3214984808907442) {
#                         if (atts[0] <= 5.9718572619240069) {
#                             classes[0] = 0;
#                             classes[1] = 0;
#                             classes[2] = 3;
#                         } else {
#                             if (atts[1] <= 2.7957859024446297) {
#                                 if (atts[3] <= 1.5647405810243964) {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 1;
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 1;
#                                     classes[2] = 0;
#                                 }
#                             } else {
#                                 classes[0] = 0;
#                                 classes[1] = 0;
#                                 classes[2] = 2;
#                             }
#                         }
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 8;
#                     }
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 0;
#                     classes[2] = 27;
#                 }
#             }
#         }
#         int class_idx = 0;
#         int class_val = classes[0];
#         for (int i = 1; i < 3; i++) {
#             if (classes[i] > class_val) {
#                 class_idx = i;
#                 class_val = classes[i];
#             }
#         }
#         return class_idx;
#     }
#
#     public static int predict_08(float[] atts) {
#         int[] classes = new int[3];
#         if (atts[2] <= 2.0691016397809667) {
#             classes[0] = 50;
#             classes[1] = 0;
#             classes[2] = 0;
#         } else {
#             if (atts[3] <= 2.2998321977443212) {
#                 if (atts[2] <= 5.7568501553340212) {
#                     if (atts[3] <= 1.3336353223788777) {
#                         classes[0] = 0;
#                         classes[1] = 28;
#                         classes[2] = 0;
#                     } else {
#                         if (atts[1] <= 2.7958206613730536) {
#                             if (atts[3] <= 1.6143543486045333) {
#                                 if (atts[2] <= 4.272187437395063) {
#                                     classes[0] = 0;
#                                     classes[1] = 1;
#                                     classes[2] = 0;
#                                 } else {
#                                     if (atts[3] <= 1.4076716455330911) {
#                                         classes[0] = 0;
#                                         classes[1] = 0;
#                                         classes[2] = 1;
#                                     } else {
#                                         if (atts[1] <= 2.4055632554020563) {
#                                             if (atts[0] <= 6.1686928410480846) {
#                                                 classes[0] = 0;
#                                                 classes[1] = 0;
#                                                 classes[2] = 1;
#                                             } else {
#                                                 classes[0] = 0;
#                                                 classes[1] = 1;
#                                                 classes[2] = 0;
#                                             }
#                                         } else {
#                                             classes[0] = 0;
#                                             classes[1] = 2;
#                                             classes[2] = 0;
#                                         }
#                                     }
#                                 }
#                             } else {
#                                 classes[0] = 0;
#                                 classes[1] = 0;
#                                 classes[2] = 7;
#                             }
#                         } else {
#                             if (atts[3] <= 2.0267573938538184) {
#                                 if (atts[3] <= 1.6093679712505249) {
#                                     if (atts[2] <= 4.6346452780257099) {
#                                         classes[0] = 0;
#                                         classes[1] = 10;
#                                         classes[2] = 0;
#                                     } else {
#                                         if (atts[2] <= 4.9520732824288594) {
#                                             classes[0] = 0;
#                                             classes[1] = 6;
#                                             classes[2] = 0;
#                                         } else {
#                                             classes[0] = 0;
#                                             classes[1] = 0;
#                                             classes[2] = 1;
#                                         }
#                                     }
#                                 } else {
#                                     if (atts[1] <= 2.8720874155782443) {
#                                         classes[0] = 0;
#                                         classes[1] = 0;
#                                         classes[2] = 2;
#                                     } else {
#                                         if (atts[0] <= 6.5173975475393169) {
#                                             if (atts[1] <= 3.1451728271597688) {
#                                                 classes[0] = 0;
#                                                 classes[1] = 0;
#                                                 classes[2] = 7;
#                                             } else {
#                                                 if (atts[0] <= 6.1052098155768029) {
#                                                     classes[0] = 0;
#                                                     classes[1] = 1;
#                                                     classes[2] = 0;
#                                                 } else {
#                                                     classes[0] = 0;
#                                                     classes[1] = 0;
#                                                     classes[2] = 1;
#                                                 }
#                                             }
#                                         } else {
#                                             classes[0] = 0;
#                                             classes[1] = 1;
#                                             classes[2] = 0;
#                                         }
#                                     }
#                                 }
#                             } else {
#                                 classes[0] = 0;
#                                 classes[1] = 0;
#                                 classes[2] = 5;
#                             }
#                         }
#                     }
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 0;
#                     classes[2] = 11;
#                 }
#             } else {
#                 classes[0] = 0;
#                 classes[1] = 0;
#                 classes[2] = 14;
#             }
#         }
#         int class_idx = 0;
#         int class_val = classes[0];
#         for (int i = 1; i < 3; i++) {
#             if (classes[i] > class_val) {
#                 class_idx = i;
#                 class_val = classes[i];
#             }
#         }
#         return class_idx;
#     }
#
#     public static int predict_09(float[] atts) {
#         int[] classes = new int[3];
#         if (atts[2] <= 1.7984751272085284) {
#             classes[0] = 48;
#             classes[1] = 0;
#             classes[2] = 0;
#         } else {
#             if (atts[2] <= 2.743614525085631) {
#                 classes[0] = 2;
#                 classes[1] = 0;
#                 classes[2] = 0;
#             } else {
#                 if (atts[3] <= 1.5726279935672078) {
#                     if (atts[1] <= 2.8580128675676453) {
#                         if (atts[3] <= 1.0071725081685803) {
#                             classes[0] = 0;
#                             classes[1] = 7;
#                             classes[2] = 0;
#                         } else {
#                             if (atts[3] <= 1.4508596346903717) {
#                                 if (atts[0] <= 5.4788470711552755) {
#                                     classes[0] = 0;
#                                     classes[1] = 2;
#                                     classes[2] = 0;
#                                 } else {
#                                     if (atts[2] <= 4.3105326892188245) {
#                                         classes[0] = 0;
#                                         classes[1] = 9;
#                                         classes[2] = 0;
#                                     } else {
#                                         if (atts[3] <= 1.2510118285538396) {
#                                             classes[0] = 0;
#                                             classes[1] = 2;
#                                             classes[2] = 0;
#                                         } else {
#                                             if (atts[3] <= 1.3008071595868549) {
#                                                 classes[0] = 0;
#                                                 classes[1] = 2;
#                                                 classes[2] = 0;
#                                             } else {
#                                                 if (atts[1] <= 2.6182927931585578) {
#                                                     classes[0] = 0;
#                                                     classes[1] = 0;
#                                                     classes[2] = 1;
#                                                 } else {
#                                                     classes[0] = 0;
#                                                     classes[1] = 1;
#                                                     classes[2] = 0;
#                                                 }
#                                             }
#                                         }
#                                     }
#                                 }
#                             } else {
#                                 if (atts[2] <= 5.0329756375451549) {
#                                     if (atts[2] <= 4.9472819976263134) {
#                                         classes[0] = 0;
#                                         classes[1] = 3;
#                                         classes[2] = 0;
#                                     } else {
#                                         classes[0] = 0;
#                                         classes[1] = 0;
#                                         classes[2] = 1;
#                                     }
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 1;
#                                 }
#                             }
#                         }
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 19;
#                         classes[2] = 0;
#                     }
#                 } else {
#                     if (atts[2] <= 6.1966904607047031) {
#                         if (atts[2] <= 5.7145493422412432) {
#                             if (atts[3] <= 1.7483953438728816) {
#                                 if (atts[0] <= 5.1244250369354614) {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 1;
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 4;
#                                     classes[2] = 0;
#                                 }
#                             } else {
#                                 if (atts[0] <= 5.9635245683145435) {
#                                     if (atts[1] <= 2.9403134736939704) {
#                                         classes[0] = 0;
#                                         classes[1] = 0;
#                                         classes[2] = 5;
#                                     } else {
#                                         if (atts[1] <= 3.1597595674715184) {
#                                             classes[0] = 0;
#                                             classes[1] = 0;
#                                             classes[2] = 1;
#                                         } else {
#                                             classes[0] = 0;
#                                             classes[1] = 1;
#                                             classes[2] = 0;
#                                         }
#                                     }
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 24;
#                                 }
#                             }
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 0;
#                             classes[2] = 10;
#                         }
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 6;
#                     }
#                 }
#             }
#         }
#         int class_idx = 0;
#         int class_val = classes[0];
#         for (int i = 1; i < 3; i++) {
#             if (classes[i] > class_val) {
#                 class_idx = i;
#                 class_val = classes[i];
#             }
#         }
#         return class_idx;
#     }
#
#     public static int predict_10(float[] atts) {
#         int[] classes = new int[3];
#         if (atts[0] <= 4.9792803198265787) {
#             if (atts[1] <= 2.4799516418973426) {
#                 if (atts[2] <= 2.1081189544909256) {
#                     classes[0] = 1;
#                     classes[1] = 0;
#                     classes[2] = 0;
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 1;
#                     classes[2] = 0;
#                 }
#             } else {
#                 if (atts[2] <= 2.3031890773229251) {
#                     classes[0] = 19;
#                     classes[1] = 0;
#                     classes[2] = 0;
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 0;
#                     classes[2] = 1;
#                 }
#             }
#         } else {
#             if (atts[2] <= 4.0566901295803843) {
#                 if (atts[2] <= 3.864595594637342) {
#                     if (atts[1] <= 2.4493558386314596) {
#                         classes[0] = 0;
#                         classes[1] = 4;
#                         classes[2] = 0;
#                     } else {
#                         if (atts[1] <= 3.1013702094660816) {
#                             if (atts[0] <= 5.4182674136813782) {
#                                 if (atts[1] <= 2.7351736650965983) {
#                                     classes[0] = 0;
#                                     classes[1] = 1;
#                                     classes[2] = 0;
#                                 } else {
#                                     classes[0] = 1;
#                                     classes[1] = 0;
#                                     classes[2] = 0;
#                                 }
#                             } else {
#                                 classes[0] = 0;
#                                 classes[1] = 2;
#                                 classes[2] = 0;
#                             }
#                         } else {
#                             classes[0] = 29;
#                             classes[1] = 0;
#                             classes[2] = 0;
#                         }
#                     }
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 8;
#                     classes[2] = 0;
#                 }
#             } else {
#                 if (atts[3] <= 2.4500323361484484) {
#                     if (atts[0] <= 7.2381296350332578) {
#                         if (atts[2] <= 4.9855890272850028) {
#                             if (atts[3] <= 1.4175185078836599) {
#                                 classes[0] = 0;
#                                 classes[1] = 19;
#                                 classes[2] = 0;
#                             } else {
#                                 if (atts[3] <= 1.9085482831152847) {
#                                     if (atts[3] <= 1.5077226825858063) {
#                                         classes[0] = 0;
#                                         classes[1] = 10;
#                                         classes[2] = 0;
#                                     } else {
#                                         if (atts[1] <= 2.8843708444178024) {
#                                             classes[0] = 0;
#                                             classes[1] = 0;
#                                             classes[2] = 2;
#                                         } else {
#                                             if (atts[3] <= 1.7559926563657857) {
#                                                 classes[0] = 0;
#                                                 classes[1] = 2;
#                                                 classes[2] = 0;
#                                             } else {
#                                                 if (atts[1] <= 3.0877764696391852) {
#                                                     classes[0] = 0;
#                                                     classes[1] = 0;
#                                                     classes[2] = 2;
#                                                 } else {
#                                                     classes[0] = 0;
#                                                     classes[1] = 1;
#                                                     classes[2] = 0;
#                                                 }
#                                             }
#                                         }
#                                     }
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 1;
#                                 }
#                             }
#                         } else {
#                             if (atts[3] <= 1.9978145065895823) {
#                                 if (atts[3] <= 1.7865760346812085) {
#                                     if (atts[2] <= 5.0311129249450408) {
#                                         if (atts[0] <= 6.33527863763707) {
#                                             classes[0] = 0;
#                                             classes[1] = 0;
#                                             classes[2] = 1;
#                                         } else {
#                                             classes[0] = 0;
#                                             classes[1] = 1;
#                                             classes[2] = 0;
#                                         }
#                                     } else {
#                                         if (atts[3] <= 1.5068336243767517) {
#                                             classes[0] = 0;
#                                             classes[1] = 0;
#                                             classes[2] = 2;
#                                         } else {
#                                             if (atts[2] <= 5.7874281257170956) {
#                                                 classes[0] = 0;
#                                                 classes[1] = 1;
#                                                 classes[2] = 0;
#                                             } else {
#                                                 classes[0] = 0;
#                                                 classes[1] = 0;
#                                                 classes[2] = 1;
#                                             }
#                                         }
#                                     }
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 10;
#                                 }
#                             } else {
#                                 classes[0] = 0;
#                                 classes[1] = 0;
#                                 classes[2] = 19;
#                             }
#                         }
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 8;
#                     }
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 0;
#                     classes[2] = 3;
#                 }
#             }
#         }
#         int class_idx = 0;
#         int class_val = classes[0];
#         for (int i = 1; i < 3; i++) {
#             if (classes[i] > class_val) {
#                 class_idx = i;
#                 class_val = classes[i];
#             }
#         }
#         return class_idx;
#     }
#
#     public static int predict_11(float[] atts) {
#         int[] classes = new int[3];
#         if (atts[1] <= 3.1769571393777749) {
#             if (atts[3] <= 0.65371013474073503) {
#                 classes[0] = 13;
#                 classes[1] = 0;
#                 classes[2] = 0;
#             } else {
#                 if (atts[3] <= 1.8469214377642704) {
#                     if (atts[2] <= 4.7208371488439003) {
#                         if (atts[0] <= 5.7627729686766793) {
#                             if (atts[3] <= 1.6360344565374696) {
#                                 classes[0] = 0;
#                                 classes[1] = 21;
#                                 classes[2] = 0;
#                             } else {
#                                 classes[0] = 0;
#                                 classes[1] = 0;
#                                 classes[2] = 1;
#                             }
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 19;
#                             classes[2] = 0;
#                         }
#                     } else {
#                         if (atts[3] <= 1.7208344558505428) {
#                             if (atts[2] <= 4.9971796261133532) {
#                                 classes[0] = 0;
#                                 classes[1] = 3;
#                                 classes[2] = 0;
#                             } else {
#                                 if (atts[0] <= 7.1981562798039125) {
#                                     if (atts[2] <= 5.257435716402731) {
#                                         if (atts[0] <= 6.4367447385930507) {
#                                             if (atts[0] <= 6.2970719060390206) {
#                                                 if (atts[3] <= 1.5681733276103198) {
#                                                     classes[0] = 0;
#                                                     classes[1] = 0;
#                                                     classes[2] = 1;
#                                                 } else {
#                                                     classes[0] = 0;
#                                                     classes[1] = 1;
#                                                     classes[2] = 0;
#                                                 }
#                                             } else {
#                                                 classes[0] = 0;
#                                                 classes[1] = 0;
#                                                 classes[2] = 1;
#                                             }
#                                         } else {
#                                             classes[0] = 0;
#                                             classes[1] = 1;
#                                             classes[2] = 0;
#                                         }
#                                     } else {
#                                         classes[0] = 0;
#                                         classes[1] = 0;
#                                         classes[2] = 1;
#                                     }
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 1;
#                                 }
#                             }
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 0;
#                             classes[2] = 10;
#                         }
#                     }
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 0;
#                     classes[2] = 22;
#                 }
#             }
#         } else {
#             if (atts[2] <= 3.8047750384801153) {
#                 classes[0] = 37;
#                 classes[1] = 0;
#                 classes[2] = 0;
#             } else {
#                 if (atts[3] <= 2.1793218211052987) {
#                     if (atts[2] <= 5.2771791684203357) {
#                         if (atts[3] <= 1.8542162920413572) {
#                             classes[0] = 0;
#                             classes[1] = 5;
#                             classes[2] = 0;
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 0;
#                             classes[2] = 1;
#                         }
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 3;
#                     }
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 0;
#                     classes[2] = 9;
#                 }
#             }
#         }
#         int class_idx = 0;
#         int class_val = classes[0];
#         for (int i = 1; i < 3; i++) {
#             if (classes[i] > class_val) {
#                 class_idx = i;
#                 class_val = classes[i];
#             }
#         }
#         return class_idx;
#     }
#
#     public static int predict_12(float[] atts) {
#         int[] classes = new int[3];
#         if (atts[3] <= 1.7021128663169045) {
#             if (atts[1] <= 2.9849618599913255) {
#                 if (atts[3] <= 1.4970948166731561) {
#                     if (atts[0] <= 5.6231451775870234) {
#                         if (atts[0] <= 4.7790295218134444) {
#                             classes[0] = 2;
#                             classes[1] = 0;
#                             classes[2] = 0;
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 13;
#                             classes[2] = 0;
#                         }
#                     } else {
#                         if (atts[3] <= 1.3284118253233357) {
#                             classes[0] = 0;
#                             classes[1] = 14;
#                             classes[2] = 0;
#                         } else {
#                             if (atts[1] <= 2.7629100232801327) {
#                                 classes[0] = 0;
#                                 classes[1] = 0;
#                                 classes[2] = 1;
#                             } else {
#                                 classes[0] = 0;
#                                 classes[1] = 2;
#                                 classes[2] = 0;
#                             }
#                         }
#                     }
#                 } else {
#                     if (atts[3] <= 1.6376492357878205) {
#                         if (atts[2] <= 4.602142482827186) {
#                             classes[0] = 0;
#                             classes[1] = 3;
#                             classes[2] = 0;
#                         } else {
#                             if (atts[1] <= 2.7219560941499337) {
#                                 if (atts[0] <= 6.0880929809363096) {
#                                     if (atts[3] <= 1.5437156698788206) {
#                                         classes[0] = 0;
#                                         classes[1] = 0;
#                                         classes[2] = 1;
#                                     } else {
#                                         classes[0] = 0;
#                                         classes[1] = 1;
#                                         classes[2] = 0;
#                                     }
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 1;
#                                     classes[2] = 0;
#                                 }
#                             } else {
#                                 classes[0] = 0;
#                                 classes[1] = 0;
#                                 classes[2] = 1;
#                             }
#                         }
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 1;
#                     }
#                 }
#             } else {
#                 if (atts[3] <= 0.65664750033467612) {
#                     classes[0] = 48;
#                     classes[1] = 0;
#                     classes[2] = 0;
#                 } else {
#                     if (atts[2] <= 5.3797694299386132) {
#                         classes[0] = 0;
#                         classes[1] = 15;
#                         classes[2] = 0;
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 1;
#                     }
#                 }
#             }
#         } else {
#             if (atts[2] <= 5.2961110038745645) {
#                 if (atts[2] <= 4.8192148664349324) {
#                     if (atts[0] <= 6.0796086876843409) {
#                         if (atts[1] <= 3.1799571540756144) {
#                             classes[0] = 0;
#                             classes[1] = 0;
#                             classes[2] = 1;
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 1;
#                             classes[2] = 0;
#                         }
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 1;
#                     }
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 0;
#                     classes[2] = 13;
#                 }
#             } else {
#                 classes[0] = 0;
#                 classes[1] = 0;
#                 classes[2] = 30;
#             }
#         }
#         int class_idx = 0;
#         int class_val = classes[0];
#         for (int i = 1; i < 3; i++) {
#             if (classes[i] > class_val) {
#                 class_idx = i;
#                 class_val = classes[i];
#             }
#         }
#         return class_idx;
#     }
#
#     public static int predict_13(float[] atts) {
#         int[] classes = new int[3];
#         if (atts[0] <= 7.5141141980731554) {
#             if (atts[1] <= 2.7153722801061679) {
#                 if (atts[0] <= 5.0467466254497211) {
#                     if (atts[0] <= 4.7200417810678674) {
#                         classes[0] = 1;
#                         classes[1] = 0;
#                         classes[2] = 0;
#                     } else {
#                         if (atts[2] <= 3.8068991717706888) {
#                             classes[0] = 0;
#                             classes[1] = 3;
#                             classes[2] = 0;
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 0;
#                             classes[2] = 1;
#                         }
#                     }
#                 } else {
#                     if (atts[2] <= 4.9043062884719539) {
#                         if (atts[2] <= 4.6235755435786769) {
#                             classes[0] = 0;
#                             classes[1] = 16;
#                             classes[2] = 0;
#                         } else {
#                             if (atts[3] <= 1.7869205517467841) {
#                                 classes[0] = 0;
#                                 classes[1] = 1;
#                                 classes[2] = 0;
#                             } else {
#                                 classes[0] = 0;
#                                 classes[1] = 0;
#                                 classes[2] = 1;
#                             }
#                         }
#                     } else {
#                         if (atts[0] <= 5.9034846845099169) {
#                             classes[0] = 0;
#                             classes[1] = 0;
#                             classes[2] = 3;
#                         } else {
#                             if (atts[0] <= 6.0069964788449255) {
#                                 if (atts[2] <= 5.092356647339539) {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 1;
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 1;
#                                     classes[2] = 0;
#                                 }
#                             } else {
#                                 classes[0] = 0;
#                                 classes[1] = 0;
#                                 classes[2] = 4;
#                             }
#                         }
#                     }
#                 }
#             } else {
#                 if (atts[2] <= 3.9028751327869577) {
#                     if (atts[3] <= 0.92942217227084012) {
#                         classes[0] = 49;
#                         classes[1] = 0;
#                         classes[2] = 0;
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 1;
#                         classes[2] = 0;
#                     }
#                 } else {
#                     if (atts[3] <= 1.95141861134337) {
#                         if (atts[2] <= 5.9202819502434796) {
#                             if (atts[1] <= 3.0337596705736161) {
#                                 if (atts[1] <= 2.9515778790039278) {
#                                     if (atts[3] <= 1.565131126685861) {
#                                         if (atts[3] <= 1.4106022275635259) {
#                                             classes[0] = 0;
#                                             classes[1] = 10;
#                                             classes[2] = 0;
#                                         } else {
#                                             if (atts[1] <= 2.8239691897096861) {
#                                                 if (atts[0] <= 6.3740416107871889) {
#                                                     classes[0] = 0;
#                                                     classes[1] = 0;
#                                                     classes[2] = 1;
#                                                 } else {
#                                                     classes[0] = 0;
#                                                     classes[1] = 1;
#                                                     classes[2] = 0;
#                                                 }
#                                             } else {
#                                                 classes[0] = 0;
#                                                 classes[1] = 1;
#                                                 classes[2] = 0;
#                                             }
#                                         }
#                                     } else {
#                                         classes[0] = 0;
#                                         classes[1] = 0;
#                                         classes[2] = 2;
#                                     }
#                                 } else {
#                                     if (atts[3] <= 1.7349329909051114) {
#                                         if (atts[2] <= 5.5073889949611754) {
#                                             classes[0] = 0;
#                                             classes[1] = 8;
#                                             classes[2] = 0;
#                                         } else {
#                                             classes[0] = 0;
#                                             classes[1] = 0;
#                                             classes[2] = 1;
#                                         }
#                                     } else {
#                                         classes[0] = 0;
#                                         classes[1] = 0;
#                                         classes[2] = 4;
#                                     }
#                                 }
#                             } else {
#                                 if (atts[1] <= 3.2667907396292932) {
#                                     if (atts[0] <= 6.8510331900259231) {
#                                         if (atts[2] <= 5.3080841532160195) {
#                                             classes[0] = 0;
#                                             classes[1] = 4;
#                                             classes[2] = 0;
#                                         } else {
#                                             classes[0] = 0;
#                                             classes[1] = 0;
#                                             classes[2] = 1;
#                                         }
#                                     } else {
#                                         classes[0] = 0;
#                                         classes[1] = 2;
#                                         classes[2] = 0;
#                                     }
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 2;
#                                     classes[2] = 0;
#                                 }
#                             }
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 0;
#                             classes[2] = 3;
#                         }
#                     } else {
#                         classes[0] = 0;
#                         classes[1] = 0;
#                         classes[2] = 22;
#                     }
#                 }
#             }
#         } else {
#             classes[0] = 0;
#             classes[1] = 0;
#             classes[2] = 6;
#         }
#         int class_idx = 0;
#         int class_val = classes[0];
#         for (int i = 1; i < 3; i++) {
#             if (classes[i] > class_val) {
#                 class_idx = i;
#                 class_val = classes[i];
#             }
#         }
#         return class_idx;
#     }
#
#     public static int predict_14(float[] atts) {
#         int[] classes = new int[3];
#         if (atts[2] <= 1.5213681480470247) {
#             classes[0] = 37;
#             classes[1] = 0;
#             classes[2] = 0;
#         } else {
#             if (atts[2] <= 3.3733067152396501) {
#                 if (atts[2] <= 1.928483210061239) {
#                     classes[0] = 13;
#                     classes[1] = 0;
#                     classes[2] = 0;
#                 } else {
#                     classes[0] = 0;
#                     classes[1] = 3;
#                     classes[2] = 0;
#                 }
#             } else {
#                 if (atts[3] <= 1.4203844512442054) {
#                     if (atts[0] <= 5.2718002974390057) {
#                         classes[0] = 0;
#                         classes[1] = 2;
#                         classes[2] = 0;
#                     } else {
#                         if (atts[3] <= 1.3984442803066099) {
#                             classes[0] = 0;
#                             classes[1] = 24;
#                             classes[2] = 0;
#                         } else {
#                             if (atts[1] <= 2.815819690150918) {
#                                 if (atts[0] <= 6.5752606285820789) {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 1;
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 1;
#                                     classes[2] = 0;
#                                 }
#                             } else {
#                                 classes[0] = 0;
#                                 classes[1] = 5;
#                                 classes[2] = 0;
#                             }
#                         }
#                     }
#                 } else {
#                     if (atts[2] <= 5.0733232229961533) {
#                         if (atts[2] <= 4.7277283044181084) {
#                             if (atts[3] <= 1.6830983078709849) {
#                                 classes[0] = 0;
#                                 classes[1] = 10;
#                                 classes[2] = 0;
#                             } else {
#                                 classes[0] = 0;
#                                 classes[1] = 0;
#                                 classes[2] = 1;
#                             }
#                         } else {
#                             if (atts[3] <= 1.7955400567946675) {
#                                 if (atts[2] <= 4.9226321104784594) {
#                                     classes[0] = 0;
#                                     classes[1] = 2;
#                                     classes[2] = 0;
#                                 } else {
#                                     if (atts[1] <= 2.5475161659836072) {
#                                         classes[0] = 0;
#                                         classes[1] = 0;
#                                         classes[2] = 1;
#                                     } else {
#                                         classes[0] = 0;
#                                         classes[1] = 1;
#                                         classes[2] = 0;
#                                     }
#                                 }
#                             } else {
#                                 if (atts[1] <= 3.0302843343947554) {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 7;
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 1;
#                                     classes[2] = 0;
#                                 }
#                             }
#                         }
#                     } else {
#                         if (atts[1] <= 3.110966564102843) {
#                             if (atts[3] <= 1.8737745337319733) {
#                                 if (atts[0] <= 6.1758400547445209) {
#                                     if (atts[0] <= 5.9332085107091395) {
#                                         classes[0] = 0;
#                                         classes[1] = 0;
#                                         classes[2] = 1;
#                                     } else {
#                                         classes[0] = 0;
#                                         classes[1] = 1;
#                                         classes[2] = 0;
#                                     }
#                                 } else {
#                                     classes[0] = 0;
#                                     classes[1] = 0;
#                                     classes[2] = 7;
#                                 }
#                             } else {
#                                 classes[0] = 0;
#                                 classes[1] = 0;
#                                 classes[2] = 19;
#                             }
#                         } else {
#                             classes[0] = 0;
#                             classes[1] = 0;
#                             classes[2] = 13;
#                         }
#                     }
#                 }
#             }
#         }
#         int class_idx = 0;
#         int class_val = classes[0];
#         for (int i = 1; i < 3; i++) {
#             if (classes[i] > class_val) {
#                 class_idx = i;
#                 class_val = classes[i];
#             }
#         }
#         return class_idx;
#     }
#
#     public static int predict(float[] atts) {
#         int n_classes = 3;
#
#         int[] classes = new int[n_classes];
#         classes[Tmp.predict_00(atts)]++;
#         classes[Tmp.predict_01(atts)]++;
#         classes[Tmp.predict_02(atts)]++;
#         classes[Tmp.predict_03(atts)]++;
#         classes[Tmp.predict_04(atts)]++;
#         classes[Tmp.predict_05(atts)]++;
#         classes[Tmp.predict_06(atts)]++;
#         classes[Tmp.predict_07(atts)]++;
#         classes[Tmp.predict_08(atts)]++;
#         classes[Tmp.predict_09(atts)]++;
#         classes[Tmp.predict_10(atts)]++;
#         classes[Tmp.predict_11(atts)]++;
#         classes[Tmp.predict_12(atts)]++;
#         classes[Tmp.predict_13(atts)]++;
#         classes[Tmp.predict_14(atts)]++;
#
#         int class_idx = 0;
#         int class_val = classes[0];
#         for (int i = 1; i < n_classes; i++) {
#             if (classes[i] > class_val) {
#                 class_idx = i;
#                 class_val = classes[i];
#             }
#         }
#         return class_idx;
#     }
#
#     public static void main(String[] args) {
#         if (args.length == 4) {
#             float[] atts = new float[args.length];
#             for (int i = 0, l = args.length; i < l; i++) {
#                 atts[i] = Float.parseFloat(args[i]);
#             }
#             System.out.println(Tmp.predict(atts));
#         }
#     }
# }
