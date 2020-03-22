import os
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from collections import Counter
import subprocess


def create_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)

# process the test file at once and handles the case when strings are longer than 'n'
# returns an array of scores for each string from a test file
def compute_scores(n, r, training_file_path, test_file_path, alphabet):
    command = 'java -jar negsel2.jar -alphabet file://\'{}\' -self \'{}\' -n {} -r {} -c -l < {}'.format(alphabet, training_file_path, n, r, test_file_path)

    stream = os.popen(command)
    output = stream.read()
    lines = output.splitlines()

    scores = []
    for line in lines:
        # if a string is longer than 'n' the program returns scores for all substrings of length 'n'
        # separated by whitespaces
        line_scores = np.array([float(score) for score in line.rstrip().split()])
        # we average these scores to obtain the string score
        scores.append(line_scores.mean())

    return scores


# for large test files the implementation above is too inefficient
# an optimized method takes the unique substrings of each string in test data set and their occurrences
# instead of test file. It should be passed as an array of Counter objects for each string from test set to
# 'counters' parameter
def compute_scores_optimized(n, r, training_file_path, counters, alphabet):
    command = 'java -jar negsel2.jar -alphabet file://\'{}\' -self \'{}\' -n {} -r {} -c -l'.format(alphabet, training_file_path, n, r)

    # now we are processing the test set by substrings
    # run the command to launch the subprocess
    process = subprocess.Popen([command], shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
    # contains scores of string from test sets
    computed_scores = []
    for counter in counters:
        scores = []
        # for each unique substring
        for unique_sequence in counter.keys():
            # pass the substring to this process
            process.stdin.write(unique_sequence.encode('utf-8'))
            process.stdin.write(b'\n')
            process.stdin.flush()
            # read the output score
            output = process.stdout.readline()
            # multiply score to the number of substring occurrences
            score = float(output.rstrip()) * counter[unique_sequence]
            scores.append(score)

        # score of the string is an average of all substrings scores
        computed_scores.append(sum(scores)/sum(counter.values()))

    process.terminate()

    return computed_scores


# data - array of tuples (score, label)
# positive_num - number of positives in the data set
# negative_num - number of negatives in the data set
# output_folder - folder where to save the ROC curve plot
# plot - if True, shows the ROC curve plot, otherwise saves to a file
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
    plt.plot(r_specificities, sensitivities, color='darkorange', lw=2, label='AUC = %0.3f' % auc)
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


# AUC analysis of languages data
# self_file_path - test file with english
# non_self_file_path - path of test file with other language
# output_folder - folder to save the results of AUC analysis
def detect_non_english(r,
                       self_file_path,
                       non_self_file_path,
                       output_folder,
                       n=10,
                       training_file_path='english.train'):
    # process non-self (positive)
    non_self_scores = compute_scores(n, r, training_file_path, non_self_file_path, alphabet=training_file_path)
    non_self_data = [(score, 1) for score in non_self_scores]
    positive_num = len(non_self_scores)

    # process self (negative)
    self_scores = compute_scores(n, r, training_file_path, self_file_path, alphabet=training_file_path)
    self_data = [(score, 0) for score in self_scores]
    negative_num = len(self_scores)

    # merge self and non-self data
    data = non_self_data + self_data
    auc = perform_roc_analysis(data, positive_num, negative_num, output_folder, r)
    return auc


# performs AUC analysis for all the possible r values and finds r that maximizes AUC
# self_file - test file with english
# non_self_file - path of test file with other language
def auc_analysis_languages(self_file, non_self_file, output_folder, r_range=range(1, 10)):
    create_dir(output_folder)
    r_performance = []
    results = ''
    for r in r_range:
        auc = detect_non_english(r, self_file, non_self_file, output_folder)
        print('For r = {} AUC = {} \n'.format(r, auc))
        results += 'For r = {} AUC = {} \n'.format(r, auc)
        r_performance.append((r, auc))

    sorted_by_auc = sorted(r_performance, key=lambda tup: tup[1], reverse=True)
    results += 'r = {} discriminates self and non-self best, AUC = {}'.format(sorted_by_auc[0][0], sorted_by_auc[0][1])

    results_file = open("{}/results.txt".format(output_folder), "w")
    results_file.write(results)
    results_file.close()


# performs full analysis for languages data
def task1():
    english_test_file = 'english.test'

    # Task 1.2
    not_english_test_file = 'tagalog.test'
    output_folder = 'task 1.2'
    auc_analysis_languages(english_test_file, not_english_test_file, output_folder)

    # Task 1.3
    create_dir('task 1.3')
    output_folders = ['task 1.3/hiligaynon', 'task 1.3/middle-english', 'task 1.3/plautdietsch', 'task 1.3/xhosa']
    languages_files = ['lang/hiligaynon.txt', 'lang/middle-english.txt', 'lang/plautdietsch.txt', 'lang/xhosa.txt']
    for output_folder, languages_file in zip(output_folders, languages_files):
        auc_analysis_languages(english_test_file, languages_file, output_folder)


def fixed_length_substrings(string, chunk_length):
    substrings = []
    for i in range(len(string)-chunk_length+1):
        substrings.append(string[i:i + chunk_length])

    return substrings


# returns a set of all unique substrings of chunk_length and their occurrences in Counter object
def unique_substrings(string, chunk_length):
    counter = Counter()
    for i in range(len(string)-chunk_length+1):
        substring = string[i:i + chunk_length]
        counter[substring] += 1

    return counter


def write_strings_to_file(strings, file_path):
    with open(file_path, 'w') as file:
        for string in strings:
            file.write(string + "\n")


def get_labes(labels_file_path):
    labels = []
    with open(labels_file_path) as file:
        for label in file:
            labels.append(int(label.rstrip()))

    return labels


# returns the array of Counter objects which contain unique substrings the occurrences
# of chunk_length for each string in test set
def get_unique_substrings(test_file_path, chunk_length):
    counters = []
    with open(test_file_path) as file:
        for sequence in file:
            counters.append(unique_substrings(sequence.rstrip(), chunk_length))

    return counters


# transforms the training set to be compatible with the algorithm
# finds the shortest string in training set
# and extracts all uniques substrings of this length
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


# AUC analysis of system calls data
# self_file_path - test file with english
# non_self_file_path - path of test file with other language
# alpha_file_path - path of alphabet
# output_folder - folder to save the results of AUC analysis
def detect_syscall_intrusion(n,
                             r,
                             test_file_path,
                             labels,
                             training_file_path,
                             alpha_file_path,
                             output_folder):
    positive_num = len([x for x in labels if x == 1])
    negative_num = len([x for x in labels if x == 0])

    test_counters = get_unique_substrings(test_file_path, n)

    scores = compute_scores_optimized(n, r, training_file_path, test_counters, alpha_file_path)
    data = tuple(zip(scores, labels))
    auc = perform_roc_analysis(data, positive_num, negative_num, output_folder, r)
    return auc


# performs AUC analysis for all the possible r values and finds r that maximizes AUC
def auc_analysis_syscalls(test_file_path, labels, training_file_path, alpha_file_path, output_folder):
    create_dir(output_folder)
    r_performance = []
    results = ''

    # define 'n' and 'r' range based on the string length in train set
    with open(training_file_path) as file:
        line = file.readline().rstrip()
        n = len(line)

    r_range = range(1, n+1)
    for r in r_range:
        auc = detect_syscall_intrusion(n, r, test_file_path, labels, training_file_path, alpha_file_path, output_folder)
        print('For r = {} AUC = {} \n'.format(r, auc))
        results += 'For r = {} AUC = {} \n'.format(r, auc)
        r_performance.append((r, auc))

    sorted_by_auc = sorted(r_performance, key=lambda tup: tup[1], reverse=True)
    results += 'r = {} discriminates self and non-self best, AUC = {}'.format(sorted_by_auc[0][0], sorted_by_auc[0][1])

    results_file = open("{}/results.txt".format(output_folder), "w")
    results_file.write(results)
    results_file.close()


def main():
    syscalls_folder = 'syscalls'
    results_folder = 'task 2'
    subfolders = ['snd-cert', 'snd-unm']
    test_files_indexes = ['1', '2', '3']

    train_files_suffix = '.train'
    train_preprocessed_files_suffix = '.train_preprocessed'
    test_files_suffix = '.test'
    labels_files_suffix = '.labels'
    alphabet_files_suffix = '.alpha'

    # pre-process training files
    for subfolder in subfolders:
        train_file_path = os.path.join(syscalls_folder, subfolder, subfolder + train_files_suffix)
        train_preprocessed_file_path = os.path.join(syscalls_folder, subfolder, subfolder + train_preprocessed_files_suffix)
        training_sequences = process_training_set(train_file_path)
        write_strings_to_file(training_sequences, train_preprocessed_file_path)

    # perform AUC analysis for all test sets
    create_dir(results_folder)
    for subfolder in subfolders:
        subfolder_path = os.path.join(results_folder, subfolder)
        create_dir(subfolder_path)
        for index in test_files_indexes:
            test_results_path = os.path.join(results_folder, subfolder, index)
            test_file_path = os.path.join(syscalls_folder, subfolder, subfolder + '.' + index + test_files_suffix)
            training_file_path = os.path.join(syscalls_folder, subfolder, subfolder + train_preprocessed_files_suffix)
            alphabet_file_path = os.path.join(syscalls_folder, subfolder, subfolder + alphabet_files_suffix)
            labels_file_path = os.path.join(syscalls_folder, subfolder, subfolder + '.' + index + labels_files_suffix)
            labels = get_labes(labels_file_path)

            auc_analysis_syscalls(test_file_path, labels, training_file_path, alphabet_file_path, test_results_path)

    print('done')


if __name__ == "__main__":
    main()