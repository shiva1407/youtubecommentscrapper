import pandas as pd
import re
import matplotlib.pyplot as plt
from  PIL import Image
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import joblib
import base64
import feedparser
import io
import requests
import seaborn as snb

def preproceesing_fun(df):
    print("in fun")
    #df = pd.read_csv("comment.csv")
    comments= df['comments']
    l_com=list(comments)
    new_df = pd.DataFrame()
    new_df['comments'] = l_com


    stops = set(stopwords.words('english'))

    stops.add("'")
    comments_without_sw=[]
    temp=l_com
    for i in temp:
        val=""
        for v in i.split():
            v=v.lower()
            if v not in stops:
                val=val+" "+v
        comments_without_sw.append(val)

    cc=[]
    for w in comments_without_sw:
        w=re.sub('[^0-9a-zA-Z ]','',w)
        cc.append(w)

    vectorizer =  joblib.load('vectorized.pkl')
    X=vectorizer.transform(cc).toarray()
    model = joblib.load("random_forest.pkl")
    y_predct =model.predict(X)
    new_df['senti'] = y_predct
    
    positive = new_df[new_df.senti==1.0]
    negitive = new_df[new_df.senti== -1.0]
    neutral = new_df[new_df.senti==0.0]
    
    fig = snb.countplot(x='senti',data=new_df).get_figure()
    fig.savefig("output.png")

    print("mahesh")



def generate_wordcloud(df,mask):
    cloud_text = df['comments']
    cloud_text_list =list(cloud_text)
    generate_text =''
    for i in cloud_text_list:
        generate_text += i
    wcw=WordCloud(mask=mask,background_color='white',colormap="rainbow", height=3000,width=3000).generate(generate_text).to_image()
    img = io.BytesIO()
    wcw.save(img, "PNG")
    img.seek(0)
    img_b64 = base64.b64encode(img.getvalue()).decode()
    return img_b64
    
# def wc():
#      #mask images
#     mask_pos=np.array(Image.open('UpVote.png'))
#     mask_neg=np.array(Image.open('DownVote.png'))
#     #mask_neu=np.array(Image.open('neutral_mask.png'))

#     #saving word clouds
#     # generate_wordcloud(positive,mask_pos)
#     # generate_wordcloud(negitive,mask_neg)
#     # #generate_wordcloud(neutral,mask_neu)   


# if __name__ == "__main__":
#     print('main fun')
#     preproceesing_fun(df)