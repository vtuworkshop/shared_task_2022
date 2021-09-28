import sys
from sklearn.metrics import precision_recall_fscore_support


def read_conll(path):
    data = open(path).readlines()
    words, labels = [], []
    for line in data:
        line = line.strip()
        if '\t' in line:
            parts = line.split('\t')
            words.append(parts[0])
            labels.append(parts[1])
    return words, labels


if __name__ == '__main__':
    gold_dir = sys.argv[1]
    sys_dir = sys.argv[2]

    gold_is_file = os.isFile(f1)
    sys_is_file = os.isFile(f2)

    if f1_file and f2_file:
        f1_files = [f1]
        f2_files = [f2]
    elif not f1_is_file and not f2_is_file:
        f1_files = glob.glob(f1+'/*.conll')
        f2_files = glob.glob(f2+'/*.conll')

    golden_words, golden_labels = read_conll(f1)
    prediction_words, prediction_labels = read_conll(f2)

    if len(golden_words) != len(prediction_words):
        print('Mismatch: ', len(golden_words), len(prediction_words))
        exit(0)

    ps, rs, fs, ss = precision_recall_fscore_support(golden_labels,
                                                     prediction_labels,
                                                     labels=['COMMA', 'PERIOD', 'QUESTION', 'EXCLAMATION'])
    for p, r, f, s in zip(ps, rs, fs, ss):
        print(f'{p * 100:5.2f} {r * 100:5.2f} {f * 100:5.2f} {s:8d}')
