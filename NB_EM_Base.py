from sklearn import preprocessing
from sklearn.preprocessing import normalize
import numpy as np
import scipy.sparse as sparse

import abc

class NB_EM_Base(metaclass=abc.ABCMeta):
    """
    NB-based classifier for Multinomial distribution with one multinomial-distribution per class
    
    Parameters:
        log_model_prob : logarithmic model probability of the current model for the data it was fitted with
        class_prob : probability for each class of the model
        word_prob : probability for each word in class of the model
        
    Arguments:
        alpha : smothing parameter for empirical ratio of counts
    
    """
    log_model_prob = -np.inf
    
    def __init__(self, alpha=1e-2):
        self.alpha=alpha
        
    @abc.abstractmethod
    def fit(self):
        """
        Fits the model given the data
        """
    
    def predict_prob(self, X):
        """
        Predicts the probability of each text belonging to each class
    
        Arguments:
            X : array containing word counts of texts
        
        Returns:
            Y : array of probabilities for ech text beloning to each class
        """
        Y_prob = X.dot(np.log(self.word_prob))
        Y_prob = np.exp(Y_prob)
        Y_prob = np.multiply(Y_prob,self.class_prob)
        Y_prob = normalize(Y_prob, axis=1, norm='l1')
        return Y_prob
    
    def predict(self, X):
        """
        Predicts the most likely class label
    
        Arguments:
            X : array containing word counts of texts
        
        Returns:
            y : class labels with the highes probability for ech text
        """
        Y_prob = self.predict_prob(X)
        y = np.argmax(Y_prob,axis=1)
        return y
    
    def multinomial_parameters(self, X, Y):
        """
        Computes parameters for a Multinomial Distribution 
    
        Arguments:
            X : array containing word counts of texts
            Y : array with the likelyhood of each text beloning to each class
        
        Returns:
            class_prob : smothened empirical ratios for each class
            word_prob : smothened empirical ratios for each word for each class
        """
        n_dataset, n_features = X.shape
        n_classes = Y.shape[1]
        #compute class probabilities (smothened ratio of counts)
        class_prob_pred = np.linalg.norm(Y, ord=1, axis=0)+self.alpha
        class_prob_pred = class_prob_pred/(self.alpha*n_classes+n_dataset)
        #compute word probabilities for each class (smothened ratio of counts)
        zaehler = (X.T.dot(Y)+self.alpha)
        nenner = Y.T.dot(sparse.linalg.norm(X, ord=1, axis=1))+n_features*self.alpha
        word_prob_pred = np.divide(zaehler,nenner)
        return class_prob_pred, word_prob_pred
    
    @abc.abstractmethod
    def calc_log_model_prob(self):
        """
        Computes the logarithmical model probability given the Texts and Labels
        """
        
    
    