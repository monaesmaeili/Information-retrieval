# Mona Esmaeili
from nltk.stem.porter import *
import os
from collections import Counter
import operator
import math

current_path = os.path.abspath(os.path.dirname(__file__))
transcripts_directory = os.path.join(current_path, 'database')
text_files_names = os.listdir(transcripts_directory)
total_documents = len(text_files_names)
doc_to_words = {}

words = {}
stop_words = []
word_tokens_count_before = 0
word_tokens_count_after = 0

with open("stopwords.txt", 'r') as file:
    lines = file.readlines()
    for stop_word in lines:
        stop_words.append(stop_word.replace('\n', ''))

for file_name in text_files_names:
    doc_to_words[file_name] = []
    with open(os.path.join(transcripts_directory, file_name), 'r') as file:
        lines = file.readlines()
        for line in lines:
            special_removed_words = re.sub(r"[^\w\s]", '', line.strip().lower()).split()
            word_tokens_count_before += len(special_removed_words)
            for word in special_removed_words:
                if word not in stop_words:
                    if word in words:
                        words[word] += 1
                    else:
                        words[word] = 1
                    doc_to_words[file_name].append(word)



stemmer = PorterStemmer()
words_after_stemming = {}

for file_name in text_files_names:
    doc_stemmed_words = []
    for word in doc_to_words[file_name]:
        stem = stemmer.stem(word)
        doc_stemmed_words.append(stem)
        if stem in words_after_stemming:
            words_after_stemming[stem] += words[word]
        else:
            words_after_stemming[stem] = words[word]
    doc_to_words[file_name] = doc_stemmed_words
    word_tokens_count_after += len(doc_stemmed_words)

num_of_unique_words = len(words_after_stemming)
only_occurred_once = 0

for word in words_after_stemming:
    if words_after_stemming[word] == 1:
        only_occurred_once += 1
average_per_doc = word_tokens_count_after / total_documents

top30 = sorted(words_after_stemming.items(), key=operator.itemgetter(1), reverse=True)[0:30]

TF_top30 = {}
IDF_top30 = {}
TF_IDF_top30 = {}
DF_top30 = {}
scaled_TF_top30 = {}
p_top30 = {}
for term_count in top30:
    TF_top30[term_count[0]] = term_count[1]
    p_top30[term_count[0]] = TF_top30[term_count[0]] / word_tokens_count_after
    appears_in_doc = 0
    for doc_name in doc_to_words:
        if term_count[0] in doc_to_words[doc_name]:
            appears_in_doc += 1
    IDF_top30[term_count[0]] = math.log(total_documents / appears_in_doc, 2)
    DF_top30[term_count[0]] = appears_in_doc
    scaled_TF_top30[term_count[0]] = math.log(1 + TF_top30[term_count[0]], 2)
    TF_IDF_top30[term_count[0]] = TF_top30[term_count[0]] * IDF_top30[term_count[0]]

print("Number of token in database (before) : " + str(word_tokens_count_before))
print("Number of token in database (after) : " + str(word_tokens_count_after))
print("Number of unique words in database : " + str(num_of_unique_words))
print("The number of words that occur only once in the database : " + str(only_occurred_once))
print("The average number of word tokens per document : " + str(average_per_doc))
print("Term\tTF\tScaled TF\tDF\tIDF\tTF * IDF\tp(term)")

for word in top30:
    print(str(word[0]) + "\t" + str(TF_top30[word[0]]) + "\t" + str(scaled_TF_top30[word[0]]) + "\t" + str(DF_top30[word[0]]) + "\t" + str(IDF_top30[word[0]]) + "\t" + str(TF_IDF_top30[word[0]]) + "\t" + str(p_top30[word[0]]))









