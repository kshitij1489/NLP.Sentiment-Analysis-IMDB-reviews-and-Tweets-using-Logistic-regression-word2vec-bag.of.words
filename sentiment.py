import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
import numpy as np
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)
    return train_pos, train_neg, test_pos, test_neg

def remove_stop_words(user_input, stop_words):
    # Remove stopwords.
    purgedList = []
    for item in user_input:
        list = []
        for w in item:
           if w not in stop_words:
            list.append(w)
        purgedList.append(list)
    return purgedList

def createDict(input):
    dict={}
    threshold = int(0.01 * len(input))
    for items in input:
        subitems = set(items)
        for element in subitems:
            if element in dict:
                dict[element] = dict[element] + 1
            else:
                dict[element] = 1
    for k, v in dict.items():
        if v < threshold:
            del dict[k]  	
    return dict 
	
def filter_keys(pos_dict, neg_dict):
    valid_entires=[]
    unique_keys = set(pos_dict.keys() + neg_dict.keys())
    for item in unique_keys:
        pos_count = 0
        neg_count = 0 
        if item in pos_dict:
            pos_count = pos_dict[item]
        if item in neg_dict:
            neg_count = neg_dict[item]
        if pos_count >= 2 * neg_count or pos_count < 2 * neg_count:
            valid_entires.append(item)
    return valid_entires 
	
def createBinaryVector(input, valid_entires):
    binaryVec = []
    for data in input:
        dataSet = set(data)
        temp = []
        for item in valid_entires:
            if item in dataSet:
                temp.append(1)
            else:
                temp.append(0)
        binaryVec.append(temp)  
    return binaryVec  	
	
	
def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE
    train_pos_vec = remove_stop_words(train_pos, stopwords)
    train_neg_vec = remove_stop_words(train_neg, stopwords)	
    pos_dict = createDict(train_pos_vec) #This function also filters based on min threshold
    neg_dict = createDict(train_neg_vec) #This function also filters based on min threshold
    valid_entires = filter_keys(pos_dict, neg_dict)	#This function filters based on 3rd criteria
    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE
    train_pos_vec = createBinaryVector(train_pos, valid_entires)
    train_neg_vec = createBinaryVector(train_neg, valid_entires)
    test_pos_vec = createBinaryVector(test_pos, valid_entires)
    test_neg_vec = createBinaryVector(test_neg, valid_entires)
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE
    labeled_train_pos = []
    for i in range(len(train_pos)):
        labeled_train_pos.append(LabeledSentence(train_pos[i], [u'TRAIN_POS_' + str(i)]))	
    labeled_train_neg = []
    for i in range(len(train_neg)):
        labeled_train_neg.append(LabeledSentence(train_neg[i], [u'TRAIN_NEG_' + str(i)]))
    labeled_test_pos = []
    for i in range(len(test_pos)):
        labeled_test_pos.append(LabeledSentence(test_pos[i], [u'TEST_POS_' + str(i)]))
    labeled_test_neg = []
    for i in range(len(test_neg)):
        labeled_test_neg.append(LabeledSentence(test_neg[i], [u'TEST_NEG_' + str(i)]))
    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)
    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)
    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    train_pos_vec=[]
    for i in range(len(train_pos)):
        train_pos_vec.append(model.docvecs[u'TRAIN_POS_' + str(i)])
    train_neg_vec=[]
    for i in range(len(train_neg)):
        train_neg_vec.append(model.docvecs[u'TRAIN_NEG_' + str(i)])
    test_pos_vec=[]
    for i in range(len(test_pos)):
        test_pos_vec.append(model.docvecs[u'TEST_POS_' + str(i)])
    test_neg_vec=[]
    for i in range(len(test_neg)):
        test_neg_vec.append(model.docvecs[u'TEST_NEG_' + str(i)])
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    nb_model = BernoulliNB(alpha=1.0, binarize=None)
    nb_model.fit(np.array(train_pos_vec + train_neg_vec), np.array(Y))
    lr_model = LogisticRegression()
    lr_model.fit(np.array(train_pos_vec + train_neg_vec), np.array(Y))
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    nb_model = GaussianNB()
    nb_model.fit(np.array(train_pos_vec + train_neg_vec), np.array(Y))
    lr_model = LogisticRegression()
    lr_model.fit(np.array(train_pos_vec + train_neg_vec), np.array(Y))
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    tp = 0
    fn = 0
    p_predict = model.predict(test_pos_vec)
    for item in p_predict:
        if item == "pos":
            tp = tp + 1
        else:
            fn = fn + 1
    n_predict = model.predict(test_neg_vec)
    fp = 0
    tn = 0
    for item in n_predict:
        if item == "neg":
            tn = tn + 1
        else:
            fp = fp + 1            
    accuracy = float(tp + tn) / (tp + tn + fn + fp)
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
