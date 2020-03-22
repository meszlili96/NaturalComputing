import os
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import subprocess


def create_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)


def compute_scores_from_file(n, r, training_file_path, test_file_path, is_self):
    command = 'java -jar negsel2.jar -self {} -n {} -r {} -c -l < {}'.format(training_file_path, n, r, test_file_path)
    # process non-self (positive)
    stream = os.popen(command)
    output = stream.read()
    scores = [(float(line), 0 if is_self else 1) for line in output.splitlines()]
    return scores


def detect_non_self_task1(r,
                          self_file_path,
                          non_self_file_path,
                          output_folder,
                          n=10,
                          training_file_path='english.train'):
    # process non-self (positive)
    non_self = compute_scores_from_file(n, r, training_file_path, non_self_file_path, False)
    positive_num = len(non_self)

    # process self (negative)
    self = compute_scores_from_file(n, r, training_file_path, self_file_path, True)
    negative_num = len(self)

    result = non_self + self
    auc = perform_roc_analysis(result, positive_num, negative_num, output_folder, r)
    return auc


# test_data - a tuple (sequence, class)
def compute_scores(n, r, training_file_path, test_file_path, labels, alphabet='task 2/snd-cert/snd-cert.alpha'):
    command = 'java -jar negsel2.jar -alphabet file://\'{}\' -self \'{}\' -n {} -r {} -c -l < {}'.format(alphabet, training_file_path, n, r, test_file_path)

    stream = os.popen(command)
    output = stream.read()
    lines = output.splitlines()

    computed_scores = []
    for index, line in enumerate(lines):
        scores = np.array([float(score) for score in line.rstrip().split()])
        computed_scores.append((scores.mean(), labels[index]))

    return computed_scores


def detect_non_sel_task2(r,
                         labels,
                         output_folder,
                         n=7,
                         test_file_path='syscalls/snd-cert/snd-cert.1.test',
                         training_file_path='task 2/snd-cert/snd-cert.train'):
    positive_num = len([x for x in labels if x == 1])
    negative_num = len([x for x in labels if x == 0])

    scores = compute_scores(n, r, training_file_path, test_file_path, labels)
    auc = perform_roc_analysis(scores, positive_num, negative_num, output_folder, r)
    return auc


# returns AUC score
def perform_roc_analysis(data, positive_num, negative_num, output_folder, r, plot=False):
    sorted_by_score = sorted(data, key=lambda tup: tup[0])

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
    if plot:
        plt.show()
    else:
        plt.savefig('{}/ROC curve, r={}.png'.format(output_folder, r))

    return auc


def analyze_auc_by_substring(self_file, non_self_file, output_folder, substring_range=range(1, 10)):
    # Task 1.2
    create_dir(output_folder)
    r_performance = []
    results = ''
    for r in substring_range:
        auc = detect_non_self_task1(r, self_file, non_self_file, output_folder)
        print('For r = {} AUC = {} \n'.format(r, auc))
        results += 'For r = {} AUC = {} \n'.format(r, auc)
        r_performance.append((r, auc))

    sorted_by_auc = sorted(r_performance, key=lambda tup: tup[1], reverse=True)
    results += 'r = {} discriminates self and non-self best, AUC = {}'.format(sorted_by_auc[0][0], sorted_by_auc[0][1])

    results_file = open("{}/results.txt".format(output_folder), "w")
    results_file.write(results)
    results_file.close()


def task1():
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



def fixed_length_substrings(string, chunk_lenght):
    substrings = []
    for i in range(len(string)-chunk_lenght+1):
        substrings.append(string[i:i + chunk_lenght])

    return substrings


def process_training_set(file_path):
    samples = []
    with open(file_path) as file:
        for line in file:
            samples.append(line.rstrip())

    # use mim length as chunk length for now
    min_length = np.array([len(sample) for sample in samples]).min()
    print("Shortest string in training set {}".format(min_length))

    # we want only to include unique substrings
    training_sequences = set()
    for sample in samples:
        training_sequences.update(fixed_length_substrings(sample, min_length))

    return training_sequences


def write_strings_to_file(strings, file_path):
    with open(file_path, 'w') as file:
        for string in strings:
            file.write(string + "\n")


# folder - 'snd-cert' or 'snd-unm'
def get_labeled_test_data(folder, file_index):
    sequences = []
    sequences_file = os.path.join('syscalls', folder, folder + '.' + str(file_index)+'.test')
    with open(sequences_file) as file:
        for sequence in file:
            sequences.append(sequence.rstrip())

    min_length = np.array([len(sample) for sample in sequences]).min()
    print("Shortest string in test set {}".format(min_length))

    labels = []
    labels_file = os.path.join('syscalls', folder, folder + '.' + str(file_index) + '.labels')
    with open(labels_file) as file:
        for label in file:
            labels.append(int(label.rstrip()))

    return tuple(zip(sequences, labels))


def get_labes(folder, file_index):
    labels = []
    labels_file = os.path.join('syscalls', folder, folder + '.' + str(file_index) + '.labels')
    with open(labels_file) as file:
        for label in file:
            labels.append(int(label.rstrip()))

    return labels


#def analyze_sequence():



def main():
    #create_dir('task 2')
    #create_dir('task 2/snd-cert')

    #training_sequences = process_training_set('syscalls/snd-cert/snd-cert.train')
    #write_strings_to_file(training_sequences, 'task 2/snd-cert/snd-cert.train')

    #test_data = get_labeled_test_data('snd-cert', 1)
    labels = get_labes('snd-cert', 1)
    auc = detect_non_sel_task2(3, labels, 'task 2/snd-cert/1')
    print(auc)

    print('done')


if __name__ == "__main__":
    main()