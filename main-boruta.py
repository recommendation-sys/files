import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
import numpy as np
# import socket
import sys


if __name__ == '__main__':

    data = ''
    h = ''
    config = ''


    config = sys.argv[1]
    data = pd.read_csv('dataset_{}.xls'.format(config))
    h = int(sys.argv[2])
    
    targets = ['heur0', 'heur1', 'heur2', 'heur3', 'heur4', 'heur5', 'heur6', 'heur7']

    tgs_remove = [targets[i] for i in range(0, 8)]
    
    X = data.drop(tgs_remove, axis=1)
    y = data[targets[h]]

    rfc = RandomForestRegressor(random_state=41, n_jobs=-1)
    boruta_selector = BorutaPy(rfc, verbose=1, random_state=41, max_iter=10)
    boruta_selector.fit(np.array(X), np.array(y))

    # Let's create a table and see exactly what features were confirmed/rejected.
    selected_rf_features = pd.DataFrame({'Feature': list(X.columns), 'Ranking': boruta_selector.ranking_})

    print("\n === Selected features (index, name_feature) === ({} h={}) ".format(config, h))

    indices = []
    for index, row in selected_rf_features.sort_values(by='Ranking').iterrows():
        if row['Ranking'] == 1:
            print("{} {}".format(index, row['Feature']))
            indices.append(int(index))

    print("\n=== Indices of confirmed features===\n{} \n".format(indices))