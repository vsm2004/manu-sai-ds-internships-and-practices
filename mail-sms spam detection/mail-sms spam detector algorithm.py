import numpy as np
import pandas as pd
p=pd.read_csv("C:\\Users\\DELL'\\Desktop\\data science projects\\sms spam detector\\spam.csv", encoding='latin-1')
p.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
p.rename(columns={'v1':'target','v2':'text'},inplace=True)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
p['target'] = encoder.fit_transform(p['target'])
# remove duplicates
p = p.drop_duplicates(keep='first')
import matplotlib.pyplot as plt
plt.pie(p['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()
import nltk
nltk.download('punkt')
p['num_characters'] = p['text'].apply(len)
# num of words
nltk.download('punkt_tab')
p['num_words'] = p['text'].apply(lambda x:len(nltk.word_tokenize(x)))
p['num_sentences'] = p['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
import seaborn as sns
plt.figure(figsize=(12,6))
sns.histplot(p[p['target'] == 0]['num_characters'])
sns.histplot(p[p['target'] == 1]['num_characters'],color='red')
plt.figure(figsize=(12,6))
sns.histplot(p[p['target'] == 0]['num_words'])
sns.histplot(p[p['target'] == 1]['num_words'],color='red')
sns.pairplot(p,hue='target')
sns.heatmap(p[['target','num_characters', 'num_words', 'num_sentences']].corr(), annot=True)
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))


    return " ".join(y)
p['transformed_text'] = p['text'].apply(transform_text)
from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
spam_wc = wc.generate(p[p['target'] == 1]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15,6))
plt.imshow(spam_wc)
ham_wc = wc.generate(p[p['target'] == 0]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15,6))
plt.imshow(ham_wc)
spam_corpus = []
for msg in p[p['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
from collections import Counter
sns.barplot(x=pd.DataFrame(Counter(spam_corpus).most_common(30))[0],y=pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()
ham_corpus = []
for msg in p[p['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)
from collections import Counter
sns.barplot(x=pd.DataFrame(Counter(ham_corpus).most_common(30))[0],y=pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)
y = p['target'].values
X = tfidf.fit_transform(p['transformed_text']).toarray()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))
mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))
bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))
import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))
