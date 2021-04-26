import csv
import pandas as pd

naive_bayes = pd.read_csv("naive_bayes.csv").to_dict()["target"]
bi_dir_ltsm = pd.read_csv("bi_dir_lstm.csv").to_dict()["target"]
logistic_regression = pd.read_csv("logistic_regression.csv").to_dict()["target"]
ltsm = pd.read_csv("lstm.csv").to_dict()["target"]
svm = pd.read_csv("svm.csv").to_dict()["target"]
ids = pd.read_csv("svm.csv").to_dict()["id"]

naive_bayes_acc = 0.7566
bi_dir_ltsm_acc = 0.7833
logistic_regression_acc = 0.7588
ltsm_acc = 0.7803
svm_acc = 0.8032

def get_p_yes(result, acc):
    if result == 1:
        return acc
    else:
        return 1-acc

with open('ensemble.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["id", "target"])
    for key in svm.keys():
        p_yes_ary = [
                    get_p_yes(naive_bayes[key], naive_bayes_acc),
                    get_p_yes(bi_dir_ltsm[key], bi_dir_ltsm_acc),
                    get_p_yes(logistic_regression[key], logistic_regression_acc),
                    get_p_yes(ltsm[key], ltsm_acc),
                    get_p_yes(svm[key], svm_acc)]
        p_yes = 1.0
        p_no = 1.0
        for i in p_yes_ary:
            p_yes = p_yes * i
            p_no = p_no * (1.0-i)
        writer.writerow([ids[key], int(p_yes > p_no)])


# with open('ensemble.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     writer.writerow(["id", "target"])
#     for key in svm.keys():
#         num_yes = naive_bayes[key] + bi_dir_ltsm[key] + logistic_regression[key] + ltsm[key] + svm[key]
#         writer.writerow([ids[key], int(num_yes >= 3)])
