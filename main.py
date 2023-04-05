import pandas as pd
import numpy as np


def classify(data,test):
    total_size = data.shape[0]
    print("\n")
    print("train veri boyutu=", total_size)
    print("test veri boyutu=", test.shape[0])

    # değişkenleri tanımlıyoruz
    correct = 0
    tp = 0
    tn = 0

    fp = 0
    fn = 0

    countGood = 0
    countBad = 0

    print("\n")
    print("Hedef    Adet    Olasılık")
    # train veri kümeinde kaç tane 'good' ve 'bad' olduğunu hesaplıyoruz
    for x in range(data.shape[0]):
        if data[x,data.shape[1]-1] == 'good':
            countGood +=1
        if data[x,data.shape[1]-1] == 'bad':
            countBad +=1

    # 'good' ve 'bad' in verideki olasılıklarını yazdırıyoruz
    probGood=countGood/total_size
    probBad= countBad / total_size

    print('Good', "\t", countGood, "\t", probGood)
    print('Bad', "\t", countBad, "\t", probBad)

    prob0 = np.zeros((test.shape[1] - 1))
    prob1 = np.zeros((test.shape[1] - 1))

    # test row eleman sayısı kadar bir döngü kuruyoruz
    for m in range(test.shape[0]):
        # test column sayısı kadar bir döngü kuruyoruz
        for k in range(test.shape[1] - 1):
            count1 = count0 = 0
            # train row sayısı kadar döngü kuruyoruz
            for j in range(data.shape[0]):
                # 'bad' kaç kere görüldüğünü kontrol ediyor
                if test[m, k] == data[j, k] and data[j, data.shape[1] - 1] == 'bad':
                    count0 += 1
                # 'good' kaç kere görüldüğünü kontrol ediyor
                if test[m, k] == data[j, k] and data[j, data.shape[1] - 1] == 'good':
                    count1 += 1
            prob0[k] = count0 / countBad
            prob1[k] = count1 / countGood

        prob_bad = probBad
        prob_good = probGood

        for i in range(test.shape[1] - 1):

            prob_bad = prob_bad * prob0[i]
            prob_good = prob_good * prob1[i]
        if prob_bad > prob_good:
            predict = 'bad'
        else:
            predict = 'good'


        # burada ise correct, true positive, true negative, false positive, false negative adetlerini
        # koşula göre belirliyoruz

        if predict == test[m, test.shape[1] - 1]:
            correct += 1
        if predict == 'good' and test[m, test.shape[1] - 1] == 'good':
            tp += 1
        if predict == 'bad' and test[m, test.shape[1] - 1] == 'bad':
            tn += 1
        if predict == 'good' and test[m, test.shape[1] - 1] == 'bad':
            fn += 1
        if predict == 'bad' and test[m, test.shape[1] - 1] == 'good':
            fp += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    tp_rate = tp / (tp + fn)
    tn_rate = tn / (fp + tn)

    print("\n\n")

    print('Test Sonucu: ')
    print('Accuracy(Doğruluk): {}'.format(accuracy))
    print('TP Rate(Gerçek Pozitif oranı): {}'.format(tp_rate))
    print('TN Rate(Gerçek Negatif oranı): {}'.format(tn_rate))
    print('TP adeti(Gerçek Pozitif): {}'.format(tp))
    print('TN adedti(Gerçek Negatif): {}'.format(tn))

    return


data_train = pd.read_csv('trainSet.csv')
data_test = pd.read_csv('testSet.csv')

testing = np.array(data_test)
training = np.array(data_train)

classify(training, testing)

