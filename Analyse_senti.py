import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences
import joblib



df = pd.read_csv("comments.csv")
comments= df['comments']

l_com=list(comments)
c = []
for i in l_com:
    f= i.split('\n')[0]
    h = f[7:]
    c.append(h)

new_df = pd.DataFrame()

new_df['comments'] = c

cc=[]
for w in c:
    w=re.sub('[^0-9a-zA-Z ]','',w)
    cc.append(w)

vectorizer=TfidfVectorizer(max_features=5000,
	lowercase=True,stop_words=set(stopwords.words()),
	tokenizer=word_tokenize,strip_accents='ascii',use_idf=True)

X=vectorizer.fit_transform(cc).toarray()


X_padd = pad_sequences(X, maxlen=1830)

model = joblib.load("random_forest.pkl")

y_predct =model.predict(X_padd)

new_df['senti'] = y_predct


