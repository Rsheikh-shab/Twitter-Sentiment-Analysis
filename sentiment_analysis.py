import collections
import nltk.classify.util, nltk.metrics
from nltk.classify import MaxentClassifier, SklearnClassifier
import csv
from sklearn.svm import LinearSVC, SVC
import random
from nltk.corpus import stopwords
import itertools
from nltk.metrics import precision, recall, f_measure
import numpy as np
import matplotlib.pyplot as plt

positive = []
negative = []
neautral = []

acrcy=[]
prcsn=[]
rcall=[]
fmsr=[]

with open('Clean_Dataset/Positive.csv', 'r') as myfile:    
    reader = csv.reader(myfile, delimiter=',')
    for val in reader:
        positive.append(val[0])

with open('Clean_Dataset/Neutral.csv', 'r') as myfile:    
    reader = csv.reader(myfile, delimiter=',')
    for val in reader:
        neautral.append(val[0])

with open('Clean_Dataset/Negative.csv', 'r') as myfile:    
    reader = csv.reader(myfile, delimiter=',')
    for val in reader:
        negative.append(val[0])

def splitter(d):    
    d_new = []
    for w in d:
        w_filter = [i.lower() for i in w.split()]
        d_new.append(w_filter)
    return d_new

def feats(w):    
    return dict([(w, True) for w in w])

# Calculating Precision, Recall & F-measure
def evaluate_classifier(featx):
    
    negfeats = [(featx(f), 'negative') for f in splitter(negative)]
    posfeats = [(featx(f), 'positive') for f in splitter(positive)]
    neautralfeats = [(featx(f), 'neautral') for f in splitter(neautral)]
    negcutoff = int(len(negfeats)*3/4)
    poscutoff = int(len(posfeats)*3/4)
    neautcutoff = int(len(neautralfeats)*3/4)
 
    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff] + neautralfeats[:neautcutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:] + neautralfeats[neautcutoff:]
    # Max Entropy and SVM classifiers
    classifier_list = ['maxent', 'svm']     
        
    for cl in classifier_list:
        if cl == 'maxent':
            classifierName = 'Maximum Entropy'
            classifier = MaxentClassifier.train(trainfeats, 'GIS', trace=0, encoding=None, labels=None, gaussian_prior_sigma=0, max_iter = 1)
        elif cl == 'svm':
            classifierName = 'SVM'
            classifier = SklearnClassifier(LinearSVC(), sparse=False)
            classifier.train(trainfeats)
            
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)
 
        for i, (feats, label) in enumerate(testfeats):
                refsets[label].add(i)
                observed = classifier.classify(feats)
                testsets[observed].add(i)
 
        accuracy = nltk.classify.util.accuracy(classifier, testfeats)
        
        pos_precision = precision(refsets['positive'], testsets['positive'])
        if pos_precision is None:
            pos_precision = 0.0
        pos_recall = recall(refsets['positive'], testsets['positive'])
        if pos_recall is None:
            pos_recall = 0.0
        pos_fmeasure = f_measure(refsets['positive'], testsets['positive'])
        if pos_fmeasure is None:
            pos_fmeasure = 0.0

        neut_precision = precision(refsets['neautral'], testsets['neautral'])
        if neut_precision is None:
            neut_precision = 0.0
        neut_recall = recall(refsets['neautral'], testsets['neautral'])
        if neut_recall is None:
            neut_recall = 0.0
        neut_fmeasure = f_measure(refsets['neautral'], testsets['neautral'])
        if neut_fmeasure is None:
            neut_fmeasure = 0.0
        
        neg_precision = precision(refsets['negative'], testsets['negative'])
        if neg_precision is None:
            neg_precision = 0.0
        neg_recall = recall(refsets['negative'], testsets['negative'])
        if neg_recall is None:
            neg_recall = 0.0
        neg_fmeasure = f_measure(refsets['negative'], testsets['negative'])
        if neg_fmeasure is None:
            neg_fmeasure = 0.0
        print ('\n')
        print (classifierName)
        print ('accuracy:', accuracy)
        acrcy.append(accuracy)
        print ('precision', (pos_precision + neg_precision + neut_precision) / 3)
        prcsn.append((pos_precision + neg_precision + neut_precision) / 3)
        print ('recall', (pos_recall + neg_recall + neut_recall ) / 3)
        rcall.append((pos_recall + neg_recall + neut_recall ) / 3)
        print ('f-measure', (pos_fmeasure + neg_fmeasure + neut_fmeasure ) / 3)
        fmsr.append((pos_fmeasure + neg_fmeasure + neut_fmeasure ) / 3)

evaluate_classifier(feats)

#Plotting:

msvm=(acrcy[1],prcsn[1],rcall[1],fmsr[1])
mmaxent=(acrcy[0],prcsn[0],rcall[0],fmsr[0])

fig, ax = plt.subplots()
index = np.arange(4)
width = 0.35
err_config = {'ecolor':'0.3'}

r1=plt.bar(index,mmaxent,width,alpha=0.4,color='b',error_kw=err_config, label='MaxEntropy')
r2=plt.bar(index+width,msvm,width,alpha=0.4,color='g',error_kw=err_config, label='SVM')
plt.xlabel('Evaluation Metrics')
plt.ylabel('Score')
plt.title('MaxEntropy VS SVM')
plt.xticks(index+width/2,('Accuracy','Precision','Recall','F-measure'))
plt.legend()
plt.tight_layout()
plt.show()
