"""
Implement a program which,
– given part 1 images,
∗ creates an m-NN classifer (for a user specified m),
∗ creates a decision-tree classifier,
For this task, you can use feature space of your choice.
– for the part 2 images, predicts the most likely labels using the user selected classifier.
The system should also output per-label precision, recall, and F1-score values as well as
output an overall accuracy value.
"""
from sklearn.tree import DecisionTreeClassifier

from Task7 import task7
from Task6 import task6
from Task5 import task5


def train_dt(data, labels):
    model = DecisionTreeClassifier(random_state=0)
    model.fit(data, labels)
    return model

def classify_dt(dt, data):
    predictions = dt.predict(data)
    return predictions

if __name__ == '__main__':

    path_part1 = "../Part1"
    path_part2 = "../Part2"
    print("Reduce and classification using all the illness brain vectors together")
    features_vect = ["hog_features", "cm10x10_features", "resnet_avgpool_1024_features", "resnet_fc_1000_features", "resnet_layer3_1024_features"]
    num_n = 5
    for feature in features_vect:
        print(f"Using feature space: {feature}")
        print(f"Loading training data for {feature}...")
        if feature == "rgb":
            data_train, labels_train = task6.load_images(path_part1, target_size=(224, 224))
            data_test, labels_test = task6.load_images(path_part2, target_size=(224, 224))
        else:
            data_train, labels_train, images_names_train = task5.load_data_and_label_feature(feature, root_dir="../Task2/new_results")
            data_test, labels_test, images_names_test = task5.load_data_and_label_feature(feature,root_dir="../Task2/part2_results")
            print(f"Using Knn for classification in space features {feature}...")
            knn = task7.train_knn(data_train, labels_train,num_n)
            predictions = task7.classify_knn(knn, data_test)
            task7.multiclass_metrics_macro_verbose(labels_test, predictions)
            print("-" * 60)
            print(f"Using Decision tree for classification in space features {feature}...")
            dt = train_dt(data_train, labels_train)
            predictions = classify_dt(dt, data_test)
            task7.multiclass_metrics_macro_verbose(labels_test, predictions)
            print("-" * 300)



