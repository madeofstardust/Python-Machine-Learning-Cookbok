# Newsgroup classification

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


#two example groups:

NewsClass = ['sci.space', 'sci.crypt']

DataTrain = fetch_20newsgroups(subset='train',
                               categories=NewsClass,
                               shuffle=True,
                               random_state=42)

print(DataTrain.target_names)

print(len(DataTrain.data))
print(len(DataTrain.target))

#extracting features from tezxt:
CountVect = CountVectorizer()
XTrainCounts = CountVect.fit_transform(DataTrain.data)
print(XTrainCounts.shape)

#we neeed to divide all the occcurences of each word by the total number of words in this doc: 
TfTransformer = TfidfTransformer(use_idf=False).fit(XTrainCounts)
XTrainNew = TfTransformer.transform(XTrainCounts)
TfidfTransformer = TfidfTransformer()
XTrainNewidf = TfidfTransformer.fit_transform(XTrainCounts)

#Classifier:
from sklearn.naive_bayes import MultinomialNB
NBMultiClassifier = MultinomialNB().fit(XTrainNewidf, DataTrain.target)

#Accuracy:
NewsClassPred = NBMultiClassifier.predict(XTrainNewidf)
accuracy = 100.0 * (DataTrain.target == NewsClassPred).sum() / XTrainNewidf.shape[0]
print("Accuracy of the classifier =", round(accuracy, 2), "%")

'''To extract features from the text, a tokenization procedure was needed. In 
the tokenization phase, within each single sentence, atomic elements called 
tokens are identified; based on the token identified, it's possible to carry 
out an analysis and evaluation of the sentence itself. Once the characteristics 
of the text had been extracted, a classifier based on the multinomial Naive 
Bayes algorithm was constructed'''
'''The Naive Bayes multinomial algorithm is used for text and images when features
represent the frequency of words (textual or visual) in a document'''



