from sklearn import preprocessing
from sklearn.preprocessing import normalize
import numpy as np
import scipy.sparse as sparse
from scipy.stats import truncnorm

import abc


class NB_EM_extended_Base(metaclass=abc.ABCMeta):
    """
    NB-based classifier for Multinomial distribution with more than one distribution(subclass) per class
    
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
        self.alpha=alpha
        
                
        
    def create_Y(self, classes, Y_labeled_simple, subclasses=1):
        """
        Creates a list with the containing the subclasses that belong to each class
        
        Arguments:
            subclasses : number of subclasses for each class (int or [int])
            classes : array containing all classes
            
            
            
        
        Returns:
            class_subclass_list : list of subclasses belonging to each class
            
            
            
            
        """
        classes_range = range(classes.size)
        class_subclass_list = list()
        subclasses_to_class = np.array(list(),dtype=int)
        Y_labeled = Y_labeled_simple[:,:0]
        Y_labeled_possible = Y_labeled_simple[:,:0]
        if (isinstance(subclasses, int)):
            for class_id in classes_range:
                #add subclasses of class class_id to position class_id of class_subclass_list
                class_subclass_list.append(range(class_id*subclasses,(class_id+1)*subclasses))
                #add class_id for new subclasses to subclass_id position of subclasses_to_class
                subclasses_to_class = np.concatenate([subclasses_to_class, np.tile(np.array([class_id]),(subclasses,))])
                #add 1 for each document that can belong to subclass to Y_labeled_possible
                Y_labeled_possible = np.hstack([Y_labeled_possible,np.tile(Y_labeled_simple[:,[class_id]],(1,subclasses))])
                #add distorted subclass-probabilitys of subclass to Y_labeled
                normal_number_generator = truncnorm(-1,1,loc=0, scale=0.1)
                Y_subclass_part = np.tile(Y_labeled_simple[:,[class_id]],(1,subclasses))
                Y_subclass_distort = normal_number_generator.rvs(Y_subclass_part.shape)
                Y_subclass_distort = np.multiply(Y_subclass_distort,Y_subclass_part)
                Y_subclass_part = np.add(Y_subclass_part,Y_subclass_distort)/subclasses
                Y_labeled = np.hstack([Y_labeled,Y_subclass_part])
        elif(isinstance(subclasses, list)):
            if(len(subclasses)<classes.size):
                print("Amount of classes and subclasses does not match")
            next_index = 0
            for class_id in classes_range:
                #add subclasses of class class_id to position class_id of class_subclass_list
                class_subclass_list.append(range(next_index,next_index+subclasses[class_id]))
                next_index+=subclasses[class_id]
                #add class_id for new subclasses to subclass_id position of subclasses_to_class
                subclasses_to_class = np.concatenate([subclasses_to_class, np.tile(np.array([class_id]),(subclasses[class_id],))])
                #add 1 for each document that can belong to subclass to Y_labeled_possible
                Y_labeled_possible = np.hstack([Y_labeled_possible,np.tile(Y_labeled_simple[:,[class_id]],(1,subclasses[class_id]))])
                #add distorted subclass-probabilitys of subclass to Y_labeled
                normal_number_generator = truncnorm(-1,1,loc=0, scale=0.1)
                Y_subclass_part = np.tile(Y_labeled_simple[:,[class_id]],(1,subclasses[class_id]))
                Y_subclass_distort = normal_number_generator.rvs(Y_subclass_part.shape)
                Y_subclass_distort = np.multiply(Y_subclass_distort,Y_subclass_part)
                Y_subclass_part = np.add(Y_subclass_part,Y_subclass_distort)/subclasses[class_id]
                Y_labeled = np.hstack([Y_labeled,Y_subclass_part])
        else:
            print("subclasses must be an int or a list of int")
        #ensure that each row adds up to 1
        Y_labeled = normalize(Y_labeled, axis=1, norm='l1')
        return Y_labeled, Y_labeled_possible, subclasses_to_class, class_subclass_list
    
    @abc.abstractmethod
    def fit(self):
        """
        Fits the model given the data
        """
    
    def predict_subclass_prob_unlabeled(self, X):
        """
        Predicts the probability of each text belonging to each subclass
    
        Arguments:
            X : array containing word counts of texts
        
        Returns:
            Y_subclass_prob : array of probabilities for ech text beloning to each subclass
        """
        Y_subclass_prob = X.dot(np.log(self.word_prob))
        Y_subclass_prob = np.exp(Y_subclass_prob)
        Y_subclass_prob = np.multiply(Y_subclass_prob,np.multiply(self.subclass_prob,self.class_prob[self.subclasses_to_class]))
        Y_subclass_prob = normalize(Y_subclass_prob, axis=1, norm='l1')
        return Y_subclass_prob
    
    def predict_subclass_prob_labeled(self, X):
        """
        Predicts the probability of each text belonging to each subclass
    
        Arguments:
            X : array containing word counts of texts for the labeled data
        
        Returns:
            Y_subclass_prob : array of probabilities for ech text beloning to each subclass
        """
        Y_subclass_prob = X.dot(np.log(self.word_prob))
        Y_subclass_prob = np.exp(Y_subclass_prob)
        Y_subclass_prob = np.multiply(Y_subclass_prob,self.subclass_prob)
        #setting probabilitys for impossible class assignments to 0
        Y_subclass_prob = np.multiply(Y_subclass_prob, self.Y_labeled_possible)
        Y_subclass_prob = normalize(Y_subclass_prob, axis=1, norm='l1')
        return Y_subclass_prob
    
    def predict_prob_unlabeled(self, X):
        """
        Predicts the probability of each text belonging to each class
    
        Arguments:
            X : array containing word counts of texts
        
        Returns:
            Y_class_prob : array of probabilities for ech text beloning to each class
            Y_subclass_prob : array of probabilities for ech text beloning to each subclass
        """
        Y_subclass_prob = self.predict_subclass_prob_unlabeled(X)
        Y_class_prob = Y_subclass_prob[:,:0]
        for class_id in range(len(self.class_subclass_list)):
            Y_class_prob = np.hstack([Y_class_prob, np.sum(Y_subclass_prob[:,self.class_subclass_list[class_id]], axis=1).reshape(-1,1)])
        return Y_class_prob, Y_subclass_prob
    
    def predict(self, X):
        """
        Predicts the most likely class label
    
        Arguments:
            X : array containing word counts of texts
        
        Returns:
            y : class labels with the highes probability for ech text
        """
        Y_class_prob, _ = self.predict_prob_unlabeled(X)
        y = np.argmax(Y_class_prob,axis=1)
        return y
    
    def multinomial_parameters(self, X, Y_class, Y_subclass):
        """
        Computes parameters for a Multinomial Distribution 
    
        Arguments:
            X : array containing word counts of texts
            Y_class : array with the likelyhood of each text beloning to each class
            Y_subclass : array with the likelyhood of each text beloning to each subclass
        Returns:
            class_prob : smothened empirical ratios for each class
            subclass_prob : smothened empirical ratios for each subclass under the prerequisit that the text belongs to the corresponding class
            word_prob : smothened empirical ratios for each word for each subclass
        """
        n_dataset, n_features = X.shape
        n_classes = Y_class.shape[1]
        #compute class probabilities (smothened ratio of counts)
        documents_in_class = np.linalg.norm(Y_class, ord=1, axis=0)
        class_prob_pred = documents_in_class+self.alpha
        class_prob_pred = class_prob_pred/(self.alpha*n_classes+n_dataset)
        #compute subclass probabilities (smothened ratio of counts)
        subclass_prob_pred = np.linalg.norm(Y_subclass, ord=1, axis=0)+self.alpha
        for class_id in range(n_classes):
            subclass_prob_pred[self.class_subclass_list[class_id]] = \
                    subclass_prob_pred[self.class_subclass_list[class_id]]/(self.alpha*len(self.class_subclass_list[class_id])+documents_in_class[class_id])
        #compute word probabilities for each class (smothened ratio of counts)
        zaehler = (X.T.dot(Y_subclass)+self.alpha)
        nenner = Y_subclass.T.dot(sparse.linalg.norm(X, ord=1, axis=1))+n_features*self.alpha
        word_prob_pred = np.divide(zaehler,nenner)
        return class_prob_pred, subclass_prob_pred, word_prob_pred
    
    @abc.abstractmethod
    def calc_log_model_prob(self):
        """
        Computes the logarithmical model probability given the Texts and Labels
        """
        
    
    