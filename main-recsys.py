import os
import mip
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor


def recommendation_system(train_index, test_index, boruta_feats, all_labels, features_data):
    predictions = []

    for h in range(8):
        boruta_feat = boruta_feats[h]

        set_features = [mip.features()[x] for x in range(len(mip.features())) if boruta_feat.__contains__(x)]

        feat_test = features_data.loc[[i for i in test_index], set_features]
        feat_train = features_data.loc[[i for i in train_index], set_features]

        labels = all_labels['heur{}'.format(h)]
        # labels_test = labels.loc[[i for i in test_index]]
        labels_train = labels.loc[[i for i in train_index]]
        rf = RandomForestRegressor(n_jobs=-1, random_state=210)
        rf.fit(feat_train, labels_train)
        pred = rf.predict(feat_test)

        x = [k for k in pred]
        predictions.append(x)
    return predictions


if __name__ == '__main__':

    # ---------------------------------------------------------
    # times used by each instance in each heuristic and configuration
    exec_time = [[[] for _ in range(8)], [[] for _ in range(8)],
                 [[] for _ in range(8)], [[] for _ in range(8)]]

    pos = 0
    for config in ["config1", "config2", "config3", "config4"]:
        file = open("input-recsys//results_{}.txt".format(config), 'r')
        h = 0
        for i in file.readlines():
            textf = float(i.replace("\n", '').split(';')[3])
            if textf == -1:
                textf = 10800.0
            exec_time[pos][h].append(textf)
            h += 1
            if h == 8:
                h = 0
        pos += 1
    # ---------------------------------------------------------

    # best features for each configuration according to Boruta
    boruta_feats = [[], [], [], []]

    pos = 0
    for c in ["config1", "config2", "config3", "config4"]:
        arq = open("input-recsys//relevant_features_{}.txt".format(c), 'r')
        for line in arq.readlines():
            v = line.replace('\n', '').split(',')
            val = [int(i) for i in v]
            boruta_feats[pos].append(val)
        pos += 1
    # ---------------------------------------------------------

    print("=== Cross Over ===")

    cv = KFold(n_splits=10, random_state=41, shuffle=True)

    solutions_found = [0, 0]

    fold = 0

    data_aux = pd.read_csv('input-recsys//dataset_config1.xls')
    features_aux = data_aux.drop(['heur0', 'heur1', 'heur2', 'heur3', 'heur4', 'heur5', 'heur6', 'heur7'], axis=1)

    for train_index, test_index in cv.split(features_aux):

        all_answers_per_conf = [[], [], [], []]

        # ---------------------------------------------------------
        tabname = 'config1'
        data = pd.read_csv('input-recsys//dataset_{}.xls'.format(tabname))
        all_labels = data[['heur0', 'heur1', 'heur2', 'heur3', 'heur4', 'heur5', 'heur6', 'heur7']]
        features = data.drop(['heur0', 'heur1', 'heur2', 'heur3', 'heur4', 'heur5', 'heur6', 'heur7'], axis=1)

        # ---------------------------------------------------------
        pred_highest = [[], [], [], []] # highest prediction per configuration
        pred_highest_h = [[], [], [], []] # highest prediction heuristic by configuration

        print("Solving Fold {}...".format(fold))
        fold += 1

        # ---------------------------------------------------------
        predictions = recommendation_system(train_index, test_index, boruta_feats[0], all_labels, features)
        for i in range(len(test_index)):
            pred = [predictions[heur][i] for heur in range(8)]
            pred_highest_h[0].append(pred.index(max(pred)))
            pred_highest[0].append(max(pred))
            answer = [v for v in all_labels.loc[test_index[i]]]
            all_answers_per_conf[0].append(answer)
        # ---------------------------------------------------------

        tabname = 'config2'
        data = pd.read_csv('input-recsys//dataset_{}.xls'.format(tabname))
        all_labels = data[['heur0', 'heur1', 'heur2', 'heur3', 'heur4', 'heur5', 'heur6', 'heur7']]
        features = data.drop(['heur0', 'heur1', 'heur2', 'heur3', 'heur4', 'heur5', 'heur6', 'heur7'], axis=1)

        # ---------------------------------------------------------
        predictions = recommendation_system(train_index, test_index, boruta_feats[1], all_labels, features)
        for i in range(len(test_index)):
            pred = [predictions[heur][i] for heur in range(8)]
            pred_highest_h[1].append(pred.index(max(pred)))
            pred_highest[1].append(max(pred))
            answer = [v for v in all_labels.loc[test_index[i]]]
            all_answers_per_conf[1].append(answer)
        # ---------------------------------------------------------

        tabname = 'config3'
        data = pd.read_csv('input-recsys//dataset_{}.xls'.format(tabname))
        all_labels = data[['heur0', 'heur1', 'heur2', 'heur3', 'heur4', 'heur5', 'heur6', 'heur7']]
        features = data.drop(['heur0', 'heur1', 'heur2', 'heur3', 'heur4', 'heur5', 'heur6', 'heur7'], axis=1)

        # ---------------------------------------------------------
        predictions = recommendation_system(train_index, test_index, boruta_feats[2], all_labels, features)
        for i in range(len(test_index)):
            pred = [predictions[heur][i] for heur in range(8)]
            pred_highest_h[2].append(pred.index(max(pred)))
            pred_highest[2].append(max(pred))
            answer = [v for v in all_labels.loc[test_index[i]]]
            all_answers_per_conf[2].append(answer)
        # ---------------------------------------------------------

        tabname = 'config4'
        data = pd.read_csv('input-recsys//dataset_{}.xls'.format(tabname))
        all_labels = data[['heur0', 'heur1', 'heur2', 'heur3', 'heur4', 'heur5', 'heur6', 'heur7']]
        features = data.drop(['heur0', 'heur1', 'heur2', 'heur3', 'heur4', 'heur5', 'heur6', 'heur7'], axis=1)

        # ---------------------------------------------------------
        predictions = recommendation_system(train_index, test_index, boruta_feats[3], all_labels, features)
        for i in range(len(test_index)):
            pred = [predictions[heur][i] for heur in range(8)]
            pred_highest_h[3].append(pred.index(max(pred)))
            pred_highest[3].append(max(pred))
            answer = [v for v in all_labels.loc[test_index[i]]]
            all_answers_per_conf[3].append(answer)
        # ---------------------------------------------------------

        # Analysis of results
        for i in range(len(test_index)):

            # ---------------------------------------------------------
            # optimal heuristic
            x1 = all_answers_per_conf[0][i]
            x2 = all_answers_per_conf[1][i]
            x3 = all_answers_per_conf[2][i]
            x4 = all_answers_per_conf[3][i]

            v = max([max(x1), max(x2), max(x3), max(x4)])

            # the optimal heuristic found a feasible solution!
            if v >= 101:
                solutions_found[0] += 1
            # ---------------------------------------------------------

            # ---------------------------------------------------------
            # heuristic recommendation system

            # Higher prediction per configuration for the instance
            x = [pred_highest[0][i], pred_highest[1][i], pred_highest[2][i], pred_highest[3][i]]

            # best configuration for the instance according to the predictions
            conf = x.index(max(x))

            # best heuristic for the instance in configuration conf (prediction)
            h = pred_highest_h[conf][i]

            # recommendation system found a feasible solution!
            if all_answers_per_conf[conf][i][h] >= 101:
                solutions_found[1] += 1
            # ---------------------------------------------------------

        print("(feasible solutions found) "
              "Optimal heuristic = {}"
              " Recommendation system = {}".format(solutions_found[0], solutions_found[1]))

    print("\n=== Final result ===")
    print("- Optimal heuristic found {} feasible solutions.".format(solutions_found[0]))
    print("- Recommendation system found {} feasible solutions. ".format(solutions_found[1]))
