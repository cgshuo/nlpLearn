# coding=gbk
def get_trainDicts(train_corpus_path):
    with open(train_corpus_path, 'r', encoding='UTF-8') as f:
        file_content = f.read().split()
    trainDicts = list(set(file_content)) # set无重复无序
    return  trainDicts

#通过迭代匹配的方式，按照字典进行分词
def forward_march(corpus_path, max_len, tranDicts, result_path):
    result = open(result_path, 'w', encoding='UTF-8')
    with open(corpus_path, 'r', encoding='UTF-8' ) as f:
        corpus_lines = f.readlines()
    for line in corpus_lines:
        tokens = []
        while len(line) > 0 :
            tryWord = line[0:max_len]
            while tryWord not in tranDicts:
                if len(tryWord) == 1:
                    break
                tryWord = tryWord[0:len(tryWord)-1]
            tokens.append(tryWord)
            line = line[len(tryWord):] #继续匹配本行剩余部分

        for word in tokens:
            if word == '\n':
                result.write('\n')
            else:
                result.write(word + "  ")
    result.close()


if __name__ == '__main__':
    train_corpus_path = 'file/pku_training.utf8'
    corpus_path = 'file/pku_test.utf8'
    result_path = 'file/forward_match_result'

    maxLen = 5
    trainDicts = get_trainDicts(train_corpus_path)
    forward_march(corpus_path, maxLen, trainDicts, result_path)