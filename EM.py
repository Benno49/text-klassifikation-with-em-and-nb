from sklearn import preprocessing
from sklearn.preprocessing import normalize
import numpy as np
import scipy.sparse as sparse

from NB_EM_Base import NB_EM_Base

class ExpectationMaximation(NB_EM_Base):
    """
    Expectation Maxiamtion classifier for Multinomial distribution with one multinomial-distribution per class
    
    Parameters:
        log_model_prob : logarithmic model probability of the current model for the data it was fitted with
        class_prob : probability for each class of the model
        word_prob : probability for each word in class of the model
        
    Arguments:
        alpha : smothing parameter for empirical ratio of counts
        tol : mimimum log_model_prob increase for next e- and m-step iteration
        max_iter : maximum number of e- and m-stem iterations
        show_progress : True if the log_model_prob in each iteration should be printed
    
    """
    log_model_prob = -np.inf
    
    def __init__(self, alpha=1e-2, max_iter=40, tol=10, show_progress=False):
        super().__init__(alpha)
        self.tol = tol
        self.max_iter = max_iter
        self.show_progress = show_progress
        
    def fit(self, X_labeled, y_labeled, X_unlabeled):
        """
        Executes the Expectation Maximation Algorithm to trains itself with the given data
    
        Arguments:
            X_labeled : array containing word counts of the labeled texts
            Y_labeled : array of labels as a binary matrix
            X_unlabeled : array containing word counts of the unlabeled texts
        
        Returns:
            self
        """
        #transforming y_labeled to binary matrix
        lb = preprocessing.LabelBinarizer()
        lb.fit(y_labeled)
        Y_labeled = lb.transform(y_labeled)
        if(Y_labeled.shape[1]==1):
            Y_labeled=np.hstack([Y_labeled==0,Y_labeled])
        #computing initial model with just the labeled texts
        self.class_prob, self.word_prob = self.multinomial_parameters(X_labeled,Y_labeled)
        prev_log_model_prob = self.log_model_prob
        self.log_model_prob = self.calc_log_model_prob(X_labeled, Y_labeled, X_unlabeled)
        X = sparse.vstack([X_labeled, X_unlabeled])
        
        iter_ = 0
        if (self.show_progress):
            print("log_model_prob in iteration ", iter_, ":", self.log_model_prob)
        while (prev_log_model_prob + self.tol < self.log_model_prob and iter_ < self.max_iter):
            iter_ += 1
            #e-step
            Y_unlabeled = self.predict_prob(X_unlabeled)
            #m-step
            Y = np.vstack([Y_labeled, Y_unlabeled])
            self.class_prob, self.word_prob = self.multinomial_parameters(X,Y)
            #update log_model_prob
            prev_log_model_prob = self.log_model_prob
            self.log_model_prob = self.calc_log_model_prob(X_labeled, Y_labeled, X_unlabeled)
            if (self.show_progress):
                print("log_model_prob in iteration ", iter_, ":", self.log_model_prob)
        return self
    
    
    def calc_log_model_prob(self, X_labeled, Y_labeled, X_unlabeled):
        """
        Computes the logarithmical model probability given the Texts and Labels
    
        Arguments:
            X_labeled : array containing word counts of the labeled texts
            Y_labeled : array of labels as a binary matrix
            X_unlabeled : array containing word counts of the unlabeled texts
        
        Returns:
            model_prob : logarithmic model probability
        """
        model_prob=0
        #summand for log_model_probability for the unlabeled data
        summand_1 = X_unlabeled.dot(np.log(self.word_prob))
        summand_1 = np.exp(np.array(summand_1))
        summand_1 = np.multiply(summand_1,self.class_prob)
        summand_1 = np.log(np.sum(summand_1, axis=1))
        model_prob += np.sum(summand_1)
        #summand for log_model_probability for the labeled data
        summand_2 = X_labeled.dot(np.log(self.word_prob))
        summand_2 = np.exp(summand_2)
        summand_2 = np.multiply(summand_2,self.class_prob)
        #debug
        if (np.count_nonzero(np.sum(np.multiply(summand_2, Y_labeled), axis=1)==0)>0):
            print(np.sum(np.multiply(summand_2, Y_labeled), axis=1))
            print(Y_labeled.shape)
        summand_2 = np.log(np.sum(np.multiply(summand_2, Y_labeled), axis=1))
        model_prob += np.sum(summand_2)
        return model_prob
    
    
