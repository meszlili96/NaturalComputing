import os
import matplotlib.pyplot as plt
from sklearn import metrics


def create_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)


def compute_scores(n, r, training_file_path, test_file_path, is_self):
    command = 'java -jar negsel2.jar -self {} -n {} -r {} -c -l < {}'.format(training_file_path, n, r, test_file_path)
    # process non-self (positive)
    stream = os.popen(command)
    output = stream.read()
    scores = [(float(line), 0 if is_self else 1) for line in output.splitlines()]
    return scores


# returns AUC score
def detect_non_self(r,
                    self_file_path,
                    non_self_file_path,
                    output_folder,
                    n=10,
                    training_file_path='english.train'):
    # process non-self (positive)
    non_self = compute_scores(n, r, training_file_path, non_self_file_path, False)
    positive_num = len(non_self)

    # process self (negative)
    self = compute_scores(n, r, training_file_path, self_file_path, True)
    negative_num = len(self)

    result = non_self + self
    sorted_by_score = sorted(result, key=lambda tup: tup[0])

    sensitivities = []
    r_specificities = []

    for index, item in enumerate(sorted_by_score):
        # below the threshold
        negative = sorted_by_score[:index]
        # above the threshold
        positive = sorted_by_score[index:]
        # which non-selves were detected as self
        tn_num = len([item for item in negative if item[1] == 0])
        # which selves were detected as non-self
        tp_num = len([item for item in positive if item[1] == 1])

        sensitivities.append(tp_num / positive_num)
        r_specificities.append(1 - tn_num / negative_num)

    auc = metrics.auc(r_specificities, sensitivities)

    plt.figure()
    plt.plot(r_specificities, sensitivities, color='darkorange', lw=2, label='AUC = %0.2f' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - specificity')
    plt.ylabel('sensitivity')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.savefig('{}/ROC curve, r={}.png'.format(output_folder, r))

    return auc


def analyze_auc_by_substring(self_file, non_self_file, output_folder, substring_range=range(1, 10)):
    # Task 1.2
    create_dir(output_folder)
    r_performance = []
    results = ''
    for r in substring_range:
        auc = detect_non_self(r, self_file, non_self_file, output_folder)
        print('For r = {} AUC = {} \n'.format(r, auc))
        results += 'For r = {} AUC = {} \n'.format(r, auc)
        r_performance.append((r, auc))

    sorted_by_auc = sorted(r_performance, key=lambda tup: tup[1], reverse=True)
    results += 'r = {} discriminates self and non-self best, AUC = {}'.format(sorted_by_auc[0][0], sorted_by_auc[0][1])

    results_file = open("{}/results.txt".format(output_folder), "w")
    results_file.write(results)
    results_file.close()


def main():
    english_test_file = 'english.test'

    # Task 1.2
    not_english_test_file = 'tagalog.test'
    output_folder = 'task 1.2'
    analyze_auc_by_substring(english_test_file, not_english_test_file, output_folder)

    # Task 1.3
    create_dir('task 1.3')
    output_folders = ['task 1.3/hiligaynon', 'task 1.3/middle-english', 'task 1.3/plautdietsch', 'task 1.3/xhosa']
    languages_files = ['lang/hiligaynon.txt', 'lang/middle-english.txt', 'lang/plautdietsch.txt', 'lang/xhosa.txt']
    for output_folder, languages_file in zip(output_folders, languages_files):
        analyze_auc_by_substring(english_test_file, languages_file, output_folder)

    print('done')




if __name__ == "__main__":
    main()