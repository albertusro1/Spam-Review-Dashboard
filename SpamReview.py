import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
import string
import nltk
import wordcloud
from nltk.corpus import stopwords
from collections import Counter
nltk.download('stopwords')
nltk.download('punkt')
import warnings
warnings.filterwarnings("ignore")

### import dataset
sms = pd.read_csv("spam.csv", encoding='latin-1')
sms.dropna(how="any", inplace=True, axis=1)
sms.columns = ['label', 'message']

### preprocessing
sms['label_num'] = sms.label.map({'ham':0, 'spam':1})


def text_preprocess(text):
    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']

    # check characters to see if they are in punctuation & join the character again
    nopunc = [char.lower() for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    # remove any stopwords
    return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])

sms['clean_msg'] = sms.message.apply(text_preprocess)

# kata yang paling sering muncul
wordss = sms.clean_msg.apply(lambda x: [word.lower() for word in x.split()])
words = Counter()

for msg in wordss:
    words.update(msg)

# create 3 new columns
sms["no_of_characters"] = sms["message"].apply(len)
sms["no_of_words"]      = sms.apply(lambda row: nltk.word_tokenize(row["message"]), axis=1).apply(len)
sms["no_of_sentence"]   = sms.apply(lambda row: nltk.sent_tokenize(row["message"]), axis=1).apply(len)


st.set_page_config(
    page_title='Dashboard Kelompok 9',
    page_icon='✅',
    layout='wide'
)

# dashboard title
st.title("Real-Time / Live Spam Review Dashboard")

# top-level filters

label_filter = st.selectbox("Select Label", ['All', 'spam', 'ham'])

if "All" in label_filter:
    sms = sms
else:
    sms = sms[sms['label'] == label_filter]

# creating a single-element container.
placeholder = st.empty()

# near real-time / live feed simulation
for seconds in range(200):

    with placeholder.container():

        # create two columns for charts

        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("### Data Count")
            fg = plt.figure(figsize=(12,8))
            sns.countplot(x=sms['label'])
            plt.xlabel('Label')
            plt.ylabel('Data Count')
            st.pyplot(fg)
        with fig_col2:
            st.markdown("### Message Length")
            fg2 = plt.figure(figsize=(12, 8))
            sms[sms.label == 'ham'].no_of_characters.plot(bins=35, kind='hist', color='blue',label='Ham messages', alpha=0.6)
            sms[sms.label == 'spam'].no_of_characters.plot(kind='hist', color='red',label='Spam messages', alpha=0.6)
            plt.legend()
            plt.xlabel("Message Length")
            st.pyplot(fg2)

        fig_col3, fig_col4, fig_col5 = st.columns(3)

        def get_top_text_ngrams(corpus, n, g):
            vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
            bag_of_words = vec.transform(corpus)
            sum_words = bag_of_words.sum(axis=0)
            words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
            return words_freq[:n]

        with fig_col3:
            st.markdown("### Unigram Analysis")
            fg3 = plt.figure(figsize=(15, 10))
            sns.barplot(x=list(dict(words.most_common(5)).values()),
                        y=list(dict(words.most_common(5)).keys()))
            st.pyplot(fg3)
        with fig_col4:
            st.markdown("### Bigram Analysis")
            fg4 = plt.figure(figsize=(15, 10))
            most_common_bi = get_top_text_ngrams(sms.message, 5, 2)
            most_common_bi = dict(most_common_bi)
            sns.barplot(x=list(most_common_bi.values()), y=list(most_common_bi.keys()))
            st.pyplot(fg4)
        with fig_col5:
            st.markdown("### Trigram Analysis")
            fg5 = plt.figure(figsize = (15,10))
            most_common_tri = get_top_text_ngrams(sms.message,5,3)
            most_common_tri = dict(most_common_tri)
            sns.barplot(x=list(most_common_tri.values()),y=list(most_common_tri.keys()))
            st.pyplot(fg5)

        st.markdown("### WordCloud")
        def show_wordcloud(data_spam_or_ham):
            text = ' '.join(data_spam_or_ham['message'].astype(str).tolist())

            fig_wordcloud = wordcloud.WordCloud(background_color='lightgrey',
                                                colormap='viridis', width=800, height=600).generate(text)

            plt.figure(figsize=(8, 4), frameon=False)
            plt.imshow(fig_wordcloud)
            plt.axis('off')
            return plt

        try:
            wordcloud = show_wordcloud(sms)
            st.pyplot(wordcloud)
        except:
            pass
