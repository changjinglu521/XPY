#nltk.download("stopwords") # 下载停用词
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer # 提取词干

input_file = './data/1967train.csv'
output_file = './data/c_1967train.csv'

writer = open(output_file, 'w')
lines = open(input_file, 'r').readlines()

# 停用词
stop_words = stopwords.words("english")
# 词干
stemmer = SnowballStemmer('english')
# 正则化表达式
text_cleaning_re = '@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+'

def preprocessing(text, stem=False):
    text = re.sub(text_cleaning_re, ' ',str(text).lower()).strip()
    tokens=[]
    for token in text.split():# 把text中每个单词提取出来
        if token not in stop_words: #如果不是停用词
            if stem:
                tokens.append(stemmer.stem(token))#提取词干
            else:
                tokens.append(token)#直接保存单词
    return ' '.join(tokens)

for i, line in enumerate(lines):
    parts = line[:-1].split('\t')
    print(parts)
    label = parts[0]
    sentence = parts[1]
    c_sentence = preprocessing(sentence)
    writer.write(label + "," + c_sentence + '\n')
writer.close()
    #print("generated augmented sentences with eda for " + train_orig + " to " + output_file + " with num_aug=" + str(num_aug))
