import os
import mip


def calc_metric(v):
    perc = 100 - v * 100 / 10800
    return int(perc)


def criate_dataset():

    for tabname in ['config1', 'config2', 'config3', 'config4']:

        file_feat = open("input-createds//instance_features_values.txt", 'r')

        features = [f for f in file_feat.readlines()]

        t = ''
        for i in mip.features():
            t += '"{}",'.format(i)

        t += '"heur0","heur1","heur2","heur3","heur4","heur5","heur6","heur7"\n'

        lines = [t]

        files = os.listdir("instances//")

        results = open("input-createds//results_{}.txt".format(tabname), 'r')

        pos = 0
        count = 0

        met = [[0 for _ in range(8)] for _ in range(files.__len__())]
        for j in results.readlines():
            text = j.replace('\n', '').split(';')
            if text[4] == 'Infeasible':
                met[pos][int(text[5])] = calc_metric(float(text[3]))
            elif text[4] == 'TimeLimit':
                met[pos][int(text[5])] = 0
            else:
                met[pos][int(text[5])] = 101 + calc_metric(float(text[3]))
            count += 1
            if count > 0 and count % 8 == 0:
                pos += 1

        results.close()

        for i in range(files.__len__()):

            t = ''
            linha = features[i].replace('\n', '').split(' ')

            for f in range(linha.__len__()):
                t += str(round(float(linha[f]), 1)) + ','

            t += '{},{},{},{},{},{},{},{}\n'.format(met[i][0], met[i][1], met[i][2], met[i][3], met[i][4], met[i][5],
                                                    met[i][6], met[i][7])
            lines.append(t)

        # .xls
        file = open("input-createds//dataset_{}.xls".format(tabname), 'w')
        file.writelines(lines)
        file.close()

        print("dataset_{} created successfully.".format(tabname))



if __name__ == '__main__':

    criate_dataset()

