{% extends 'base.combined.class' %}

{% block content %}
{% if is_test %}
import java.util.Arrays;

{% endif %}
class {{ class_name }} {

    public static int[] compute(double[] features) {
        int[] classes = new int[{{ n_classes }}];
        {{ tree | indent(4, True) }}
        return classes;
    }

    private static int findMax(int[] nums) {
        int idx = 0;
        for (int i = 0; i < nums.length; i++) {
            idx = nums[i] > nums[idx] ? i : idx;
        }
        return idx;
    }

    private static double[] normVals(int[] nums) {
        int i = 0, l = nums.length;
        double[] result = new double[l];
        double sum = 0.;
        for (i = 0; i < l; i++) {
            sum += nums[i];
        }
        if(sum == 0) {
            for (i = 0; i < l; i++) {
                result[i] = 1.0 / nums.length;
            }
        } else {
            for (i = 0; i < l; i++) {
                result[i] = nums[i] / sum;
            }
        }
        return result;
    }

    public static int predict(double[] features) {
        return findMax(compute(features));
    }

    public static double[] predictProba (double[] features) {
        return normVals(compute(features));
    }

    public static void main(String[] args) {
        int nFeatures = {{ n_features }};
        if (args.length != nFeatures) {
            throw new IllegalArgumentException("You have to pass " +  String.valueOf(nFeatures) + " features.");
        }

        // Features:
        double[] features = new double[args.length];
        for (int i = 0, l = args.length; i < l; i++) {
            features[i] = Double.parseDouble(args[i]);
        }

        // Get class prediction:
        int prediction = {{ class_name }}.predict(features);
        {% if not is_test %}
        System.out.println("Predicted class: #" + String.valueOf(prediction));
        {% endif %}

        // Get class probabilities:
        double[] probabilities = {{ class_name }}.predictProba(features);
        {% if not is_test %}
        for (int i = 0; i < probabilities.length; i++) {
            System.out.print(String.valueOf(probabilities[i]));
            if (i != probabilities.length - 1) {
                System.out.print(",");
            }
        }
        {% endif %}

        {% if is_test %}
        System.out.println("{\"predict\": " + String.valueOf(prediction) + ", \"predict_proba\": " + String.join(",", Arrays.toString(probabilities)) + "}");
        {% endif %}
    }
}
{% endblock %}