from sklearn import preprocessing
from sklearn.preprocessing import normalize
import numpy as np
import scipy.sparse as sparse

from NB_EM_extended_Base import NB_EM_extended_Base


class NaiveBayes_extended(NB_EM_extended_Base):
    """
    Naive Bayes classifier for Multinomial distribution with more than one distribution(subclass) per class
    
    Parameters:
        log_model_prob : logarithmic model probability of the current model for the data it was fitted with
        class_prob : probability for each class of the model
        subclass_prob : probability for each subclass of the model when it is known that a text belongs to the corresponding class
        word_prob : probability for each word in class of the model
        
    Arguments:
        alpha : smothing parameter for empirical ratio of counts
    """
    log_model_prob = -np.inf
    
    def __init__(self, alpha=1e-2):
        super().__init__(alpha)
        
    def fit(self, X_labeled, y_labeled, X_unlabeled, subclasses=1):
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
        Y_class_labeled = lb.transform(y_labeled)
        #for just 2 classes LabelBinarizer makes just one row with 0 and 1
        if(Y_class_labeled.shape[1]==1):
            Y_class_labeled=np.hstack([Y_class_labeled==0,Y_class_labeled])
        #initialize Y_subclass_labeled and Y_labeled_possible
        Y_subclass_labeled, self.Y_labeled_possible, self.subclasses_to_class, self.class_subclass_list = self.create_Y(lb.classes_, Y_class_labeled, subclasses)
        #computing initial model with just the labeled texts
        self.class_prob, self.subclass_prob, self.word_prob = self.multinomial_parameters(X_labeled, Y_class_labeled, Y_subclass_labeled)
        prev_log_model_prob = self.log_model_prob
        self.log_model_prob = self.calc_log_model_prob(X_labeled, Y_subclass_labeled)
        return self
    
    def calc_log_model_prob(self, X_labeled, Y_labeled_possible):
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
        #summand for log_model_probability for the labeled data
        summand_2 = X_labeled.dot(np.log(self.word_prob))
        summand_2 = np.exp(summand_2)
        summand_2 = np.multiply(summand_2,np.multiply(self.subclass_prob,self.class_prob[self.subclasses_to_class]))
        summand_2 = np.log(np.sum(np.multiply(summand_2, self.Y_labeled_possible), axis=1))
        model_prob += np.sum(summand_2)
        return model_prob
    
    
