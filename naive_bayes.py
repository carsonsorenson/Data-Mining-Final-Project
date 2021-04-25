import pandas as pd
import nltk
import re
import string
from nltk.stem import WordNetLemmatizer
import uuid
import emoji
import csv

url_uuid = uuid.uuid4()

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
STOPWORDS = set(nltk.corpus.stopwords.words('english'))

url_re = re.compile(r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])?')

def remove_url(text):
    text = url_re.sub('url_uuid', text)
    return text

def convert_emoji(text):
    text = emoji.demojize(text)
    return text

def convert_to_lowercase(text):
    text = text.lower()
    return text

def remove_punctuation(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def remove_stopwords(text):
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

def lemmatize_words(text):
    text = ' '.join(lemmatizer.lemmatize(word, pos="v") for word in text.split())
    text = ' '.join(lemmatizer.lemmatize(word, pos="a") for word in text.split())
    text = ' '.join(lemmatizer.lemmatize(word, pos="n") for word in text.split())
    return text

def fill_empty_keywords(text):
    return str(text)

def apply_all_transformations(df):
    new_df = df.copy()
    new_df['text'] = new_df['text'].apply(remove_url)
    new_df['text'] = new_df['text'].apply(remove_punctuation)
    new_df['text'] = new_df['text'].apply(convert_to_lowercase)
    new_df['text'] = new_df['text'].apply(remove_stopwords)
    new_df['text'] = new_df['text'].apply(convert_emoji)
    new_df['text'] = new_df['text'].apply(lemmatize_words)
    new_df['keyword'] = new_df['keyword'].apply(fill_empty_keywords)
    return new_df

train_df = apply_all_transformations(pd.read_csv("train.csv"))
test_df = apply_all_transformations(pd.read_csv("test.csv"))

words = {}
train_dict = train_df.to_dict()
ids = list(train_dict["id"].keys())
texts = train_dict["text"]
keywords = train_dict["keyword"]

for id in ids:
    try:
        for word in texts[id].split():
            if word not in words.keys():
                words[word] = 1
            else:
                words[word] += 1
    except:
        pass

temp = {}
keyword_likelihood = train_df.groupby(['keyword', 'target']).size().div(len(train_df))

prior = train_df.groupby('target').size().div(len(train_df))
last_len = 0
best = 0.0


with open('submission.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["id", "target"])
# for i in reversed(range(28,29)):
    for word in words.keys():
        if words[word] > 28: #at 20 -> 0.7546302377512151
            temp[word] = words[word]
    word_likelihood = {}
    # if last_len == len(temp):
    #     continue
    last_len = len(temp)
    # print("trying i=" + str(i), flush=True)
    for word in temp.keys():
        try:
            temp_df = train_df.copy()
            temp_df[word] = temp_df.apply(lambda x: word in x['text'].split(), axis=1)
            word_likelihood[word] = temp_df.groupby(['target',word]).size().div(len(temp_df))
        except:
            continue
        # print(word)
        # print(word_likelihood[word])
    right = 0
    total = 0
    for index, row in test_df.iterrows():
        p_yes = 1 # prior[1]
        p_no = 1 #prior[0]
        for word in row["text"].split():
            if word in word_likelihood.keys():
                # print(word, flush=True)
                try:
                    p_yes = p_yes * word_likelihood[word][1][True]
                except:
                    p_yes = p_yes * (1-word_likelihood[word][1][False])
                try:
                    p_no = p_no * word_likelihood[word][0][True]
                except:
                    p_no = p_no * (1-word_likelihood[word][0][False])
        if row["keyword"] != "nan":
            try:
                p_yes = p_yes * keyword_likelihood[row["keyword"]][1]
            except:
                p_yes = p_yes * (1-keyword_likelihood[row["keyword"]][0])
            try:
                p_no = p_no * keyword_likelihood[row["keyword"]][0]
            except:
                p_no = p_no * (1-keyword_likelihood[row["keyword"]][1])
        is_disaster = int(p_yes > p_no)

        writer.writerow([row["id"], is_disaster])
        # if is_disaster==row["target"]:
        #     right +=1
        # total += 1

    # if right/total > best:
    #     best = right/total
    #     print("new best! " + str(best), flush=True)
