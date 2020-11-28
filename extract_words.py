def extract_words(path):
    with open(path + '/words.txt', "rt") as f:
        labels_dict = {}
        text = list(map(str.split, f.readlines()))
        words_set = {"the"}
        for line in text:
            if "err" not in line:
                labels_dict[line[0]] = line[-1]
        for key, value in labels_dict.items():
            words_set.add(value + "\n")
    words_set = sorted(words_set)
    with open("wordlist.txt", "wt") as file:
        file.writelines(words_set)

extract_words("./dataset")