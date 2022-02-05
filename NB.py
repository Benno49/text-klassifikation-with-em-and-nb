from sklearn import preprocessing
from sklearn.preprocessing import normalize
import numpy as np
import scipy.sparse as sparse

from NB_EM_Base import NB_EM_Base


class NaiveBayes(NB_EM_Base):
    """
    Naive Bayes classifier for Multinomial distribution with one multinomial-distribution per class
    
    Parameters:
        log_model_prob : logarithmic model probability of the current model for the data it was fitted with
        class_prob : probability for each class of the model
        word_prob : probability for each word in class of the model
        
    Arguments:
        alpha : smothing parameter for empirical ratio of counts
    """
    log_model_prob = -np.inf
    
    def __init__(self, alpha=1e-2):
        super().__init__(alpha)
        
    def fit(self, X_labeled, y_labeled):
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
        self.log_model_prob = self.calc_log_model_prob(X_labeled, Y_labeled)
        
        return self
    
   
    def calc_log_model_prob(self, X_labeled, Y_labeled):
        """
        Computes the logarithmical model probability given the Texts and Labels
    
        Arguments:
            X_labeled : array containing word counts of the labeled texts
            Y_labeled : array of labels as a binary matrix
        
        Returns:
            model_prob : logarithmic model probability
        """
        model_prob = X_labeled.dot(np.log(self.word_prob))
        model_prob = np.exp(model_prob)
        model_prob = np.multiply(model_prob,self.class_prob)
        model_prob = np.log(np.sum(np.multiply(model_prob, Y_labeled), axis=1))
        model_prob = np.sum(model_prob)
        return model_prob
    
    
