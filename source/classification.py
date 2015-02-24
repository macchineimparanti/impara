'''
Created on Nov 21, 2013

@author: alessandro
'''

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.metrics.pairwise import chi2_kernel
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import RFECV
import numpy as np
from joblib import Parallel, delayed
import os
from multiprocessing import cpu_count

os.system("taskset -p 0xff %d" % os.getpid())

SVM_RBF = 0
SVM_linear = 1
SVM_RBF_Chi2_squared = 2

def classes_balance(labels):
    
    num_labels = int(np.max(labels)) + 1
    
    tot = float(labels.shape[0])
    parts = []
    for i in xrange(num_labels):
        part = np.sum((labels == i).astype(np.int))
        parts.append(part/tot)
        
    return parts
        

def dataset_scaling(X):
    
    scaler=preprocessing.Scaler().fit(X)
    X=scaler.transform(X)
    
    return X , scaler


def mse_errors(classifier,X_tr,y_tr,X_cv,y_cv):
    
    assert hasattr(classifier,"predict_proba")
    
    #it does work with logistic regression and with SVM,
    #however data training has to be performed with
    #probability estimation.
    #MSE measure may not make a lot of sense for a SVM
    prob_tr = classifier.predict_proba(X_tr)[:,1].T
    prob_cv = classifier.predict_proba(X_cv)[:,1].T
        
    tr_err=metrics.mean_squared_error(y_tr,prob_tr)
    cv_err=metrics.mean_squared_error(y_cv,prob_cv)
    
    return tr_err, cv_err


def misclassification_errors(classifier, X_tr, y_tr, X_cv, y_cv):
        
    predicted_tr = classifier.predict(X_tr)
    predicted_cv = classifier.predict(X_cv)
    
    misclassified_tr = predicted_tr!=y_tr
    misclassified_cv = predicted_cv!=y_cv
        
    tr_err=np.float(np.sum(misclassified_tr.astype("float")))/len(misclassified_tr)
    cv_err=np.float(np.sum(misclassified_cv.astype("float")))/len(misclassified_cv)
        
    return tr_err, cv_err
     
def C_and_gamma_evaluation(X_tr, y_tr, X_cv, y_cv, classifier_by_C_and_gamma_function, 
                          error_measure_function, C, idx_C, gamma, idx_gamma):
    
    classifier = classifier_by_C_and_gamma_function(X_tr, y_tr, C=C, gamma=gamma)
                      
    tr_err, cv_err = error_measure_function(classifier,X_tr,y_tr,X_cv,y_cv)
                
    y_pred=classifier.predict(X_cv)

    if hasattr(metrics,"accuracy_score"):
        acc = metrics.accuracy_score(y_cv,y_pred)
    else:
        assert hasattr(metrics,"zero_one_score")
        acc = metrics.zero_one_score(y_cv, y_pred)
    prec=metrics.precision_score(y_cv,y_pred)
    recall=metrics.recall_score(y_cv,y_pred)
    f1_score=metrics.f1_score(y_cv,y_pred)

    return idx_C, idx_gamma, tr_err, cv_err, acc, prec, recall, f1_score


def C_evaluation(X_tr, y_tr, X_cv, y_cv, classifier_by_C_function, 
                          error_measure_function, C, idx_C):
    
    classifier = classifier_by_C_function(X_tr, y_tr,C=C)
                          
    tr_err, cv_err = error_measure_function(classifier,X_tr,y_tr,X_cv,y_cv)
    
    #it is assumed that we are dealing with a sklearn classifier...
    y_pred = classifier.predict(X_cv)

    if hasattr(metrics,"accuracy_score"):
        acc = metrics.accuracy_score(y_cv,y_pred)
    else:
        assert hasattr(metrics,"zero_one_score")
        acc = metrics.zero_one_score(y_cv, y_pred)
    prec=metrics.precision_score(y_cv,y_pred)
    recall=metrics.recall_score(y_cv,y_pred)
    f1_score=metrics.f1_score(y_cv,y_pred)
    
    return idx_C, tr_err, cv_err, acc, prec, recall, f1_score


class ModelSelection(object):
    
    def __init__(self, C_list = None, gamma_list = None):
        
        assert C_list is None or isinstance(C_list, list)
        assert gamma_list is None or isinstance(gamma_list,list)
        
        if C_list is None:
            #regularization parameters
            self.C_list=[0.0000001,0.000001,0.00001,
                    0.0001,0.001,0.01,0.1,1,10,
                    100,1000,10000,100000,1000000]
        else:
            self.C_list = C_list
            
        if gamma_list is None:
            self.gamma_list = [0.0000001,0.000001,0.00001,
                          0.0001,0.001,0.01,0.1,1,10,
                          100,1000,10000,100000,1000000]
        else:
            self.gamma_list = gamma_list
            
            
    def C_selection(self,X,y,C_list, 
                    classifier_by_C_function, error_measure_function,
                    classifier_by_C_function_params = None, 
                    test_size = 0.3, n_iterations = 20, max_num_cpus = -1 ):
        
        if C_list is not None:
            self.C_list = C_list
        
        results_dict = dict()
        
        results_dict["C_list"] = self.C_list
        
        results_dict["tr_err_by_C"] = np.zeros(len(self.C_list),dtype=np.float)
        results_dict["cv_err_by_C"] =np.zeros(len(self.C_list),dtype=np.float)
        results_dict["acc_by_C"] = np.zeros(len(self.C_list),dtype=np.float)
        results_dict["prec_by_C"] = np.zeros(len(self.C_list),dtype=np.float)
        results_dict["recall_by_C"] = np.zeros(len(self.C_list),dtype=np.float)
        results_dict["f1_by_C"] = np.zeros(len(self.C_list),dtype=np.float)
        
        params = zip( range(len(self.C_list)), self.C_list )
        
        set_ripartitions = StratifiedShuffleSplit(y, n_iter = n_iterations, 
                                                  test_size = test_size, indices = False)
                
        if max_num_cpus < 0:
            num_processes = cpu_count()
        else:
            num_processes = min(cpu_count(),max_num_cpus)
                
        n_iter=len(set_ripartitions)
        for train,test in set_ripartitions:
            X_tr,X_cv,y_tr,y_cv =X[train],X[test],y[train],y[test]
            
            results = Parallel(n_jobs=num_processes)(delayed(C_evaluation)(X_tr,y_tr,X_cv,y_cv,classifier_by_C_function, 
                          error_measure_function, C, idx_C) for idx_C, C in params)
            
            for idx_C, tr_err, cv_err, acc, prec, recall, f1_score in results:
                results_dict["tr_err_by_C"][idx_C] = results_dict["tr_err_by_C"][idx_C]+tr_err/n_iter
                results_dict["cv_err_by_C"][idx_C] = results_dict["cv_err_by_C"][idx_C]+cv_err/n_iter
                results_dict["acc_by_C"][idx_C] = results_dict["acc_by_C"][idx_C] + acc/n_iter
                results_dict["prec_by_C"][idx_C] = results_dict["prec_by_C"][idx_C] + prec/n_iter
                results_dict["recall_by_C"][idx_C] = results_dict["recall_by_C"][idx_C] + recall/n_iter
                results_dict["f1_by_C"][idx_C] = results_dict["f1_by_C"][idx_C] + f1_score/n_iter
        
        return results_dict

    
    def C_gamma_selection(self,X, y, classifier_by_C_and_gamma_function, 
                          error_measure_function, C_list = None, gamma_list = None,
                          classifier_by_C_function_params = None, 
                          test_size = 0.3, n_iterations = 20, max_num_cpus = -1):
        
        if C_list is not None:
            self.C_list = C_list
            
        if gamma_list is not None:
            self.gamma_list = gamma_list
        
        results_dict = dict()
        
        results_dict["C_list"]=self.C_list
        results_dict["gamma_list"]=self.gamma_list
        
        results_dict["tr_err_by_C_and_gamma"]=np.zeros((len(self.C_list),len(self.gamma_list)), dtype=np.float)
        results_dict["cv_err_by_C_and_gamma"]=np.zeros((len(self.C_list),len(self.gamma_list)), dtype=np.float)
        
        results_dict["acc_by_C_and_gamma"]=np.zeros((len(self.C_list),len(self.gamma_list)),dtype=np.float)
        results_dict["prec_by_C_and_gamma"]=np.zeros((len(self.C_list),len(self.gamma_list)), dtype=np.float)
        results_dict["recall_by_C_and_gamma"]=np.zeros((len(self.C_list),len(self.gamma_list)), dtype=np.float)
        results_dict["f1_by_C_and_gamma"]=np.zeros((len(self.C_list),len(self.gamma_list)), dtype=np.float)
        
        set_ripartitions = StratifiedShuffleSplit(y, n_iter = n_iterations, 
                                                  test_size = test_size, indices=False)
    
        if max_num_cpus < 0:
            num_processes = cpu_count()
        else:
            num_processes = min(cpu_count(),max_num_cpus)
    
        n_iter=len(set_ripartitions)
        
        enumerated_C_list = zip( range(len(self.C_list)), self.C_list )
        enumerated_gamma_list = zip( range(len(self.gamma_list)), self.gamma_list )
        params = []
        for C_item in enumerated_C_list:
            for gamma_item in enumerated_gamma_list:
                params.append((C_item,gamma_item))
        
        for train,test in set_ripartitions:
            
            X_tr,X_cv,y_tr,y_cv =X[train],X[test],y[train],y[test]          
            
            results = Parallel(n_jobs=num_processes)(delayed(C_and_gamma_evaluation)(X_tr,y_tr,X_cv,y_cv,classifier_by_C_and_gamma_function, 
                          error_measure_function, C, idx_C, gamma, idx_gamma) for (idx_C, C),(idx_gamma, gamma) in params)
            
            for idx_C, idx_gamma, tr_err, cv_err, acc, prec, recall, f1_score in results:
                results_dict["tr_err_by_C_and_gamma"][idx_C, idx_gamma]=results_dict["tr_err_by_C_and_gamma"][idx_C, idx_gamma]+tr_err/n_iter
                results_dict["cv_err_by_C_and_gamma"][idx_C, idx_gamma]=results_dict["cv_err_by_C_and_gamma"][idx_C, idx_gamma]+cv_err/n_iter
                results_dict["acc_by_C_and_gamma"][idx_C, idx_gamma] = results_dict["acc_by_C_and_gamma"][idx_C, idx_gamma] + acc/n_iter
                results_dict["prec_by_C_and_gamma"][idx_C, idx_gamma]=results_dict["prec_by_C_and_gamma"][idx_C, idx_gamma]+prec/n_iter
                results_dict["recall_by_C_and_gamma"][idx_C, idx_gamma]=results_dict["recall_by_C_and_gamma"][idx_C, idx_gamma]+recall/n_iter
                results_dict["f1_by_C_and_gamma"][idx_C, idx_gamma]=results_dict["f1_by_C_and_gamma"][idx_C, idx_gamma]+f1_score/n_iter
                    
        return results_dict
         
            
def SVM_RBF_by_C_and_gamma_function(X,y,C,gamma):
    
    classifier = SVC(kernel="rbf",C=C,gamma=gamma, class_weight = 'auto')
    classifier.fit(X,y)
        
    return classifier
    
    
def linear_SVM_by_C_function(X,y,C):
    
    classifier = LinearSVC(C=C, class_weight = 'auto')
    classifier.fit(X,y)
       
    return classifier


def SVM_RBF_Chi2_squared_by_C_function(X,y,C):
        
    classifier = SVC(kernel=chi2_kernel, C = C, class_weight = 'auto')
    classifier.fit(X,y)
    
    return classifier
    
    
class RecursiveFeaturesElimination(object):
    
    
    def __init__(self, C = None, gamma = None, kernel = SVM_RBF, test_size = 0.3, n_iterations = 10 ):
        
        if kernel == SVM_RBF or kernel == SVM_linear or kernel == SVM_RBF_Chi2_squared:
            assert isinstance(C,(int,long,float))
            self.C = C
        
        if not kernel == SVM_RBF and not kernel == SVM_linear and not kernel == SVM_RBF_Chi2_squared:
            raise Exception("Classification type not supported!")
        
        if kernel == SVM_RBF:
            assert isinstance(gamma,(int,long,float))
            self.gamma = gamma
        
        self.kernel = kernel
        
        self.test_size = test_size
        self.n_iterations = n_iterations
    
    
    def rfe_curves(self, X, y):
        
        num_samples,num_features = X.shape
        
        tr_err_rfe = np.zeros(num_features)
        cv_err_rfe = np.zeros(num_features)
        accuracy_rfe = np.zeros(num_features)
        recall_rfe = np.zeros(num_features)
        precision_rfe = np.zeros(num_features)
        f1_score_rfe = np.zeros(num_features)
        
        for i in xrange(num_features):
            
            mask = np.zeros(num_features)
            mask[:i+1] = 1
            
            new_mask = np.tile(mask==1,(num_samples,1))
            
            extracted_X = X[new_mask]
            
            extracted_X = np.reshape(extracted_X,(num_samples,i+1))
            
            set_ripartitions = StratifiedShuffleSplit(y, n_iter = self.n_iterations, 
                                                  test_size = self.test_size, indices=False)
    
            n_iter = len(set_ripartitions)
        
            for train,test in set_ripartitions:
                
                X_tr,X_cv,y_tr,y_cv =extracted_X[train],extracted_X[test],y[train],y[test]
                
                if self.kernel == SVM_RBF:
                
                    classifier = SVM_RBF_by_C_and_gamma_function(X_tr, y_tr, C=self.C, gamma=self.gamma)      
                    tr_err, cv_err = misclassification_errors(classifier,X_tr,y_tr,X_cv,y_cv)
                                
                elif self.kernel == SVM_linear:
                    
                    classifier = linear_SVM_by_C_function(X_tr, y_tr, C=self.C)      
                    tr_err, cv_err = misclassification_errors(classifier,X_tr,y_tr,X_cv,y_cv)
                
                elif self.kernel == SVM_RBF_Chi2_squared:
                       
                    classifier = SVM_RBF_Chi2_squared_by_C_function(X_tr, y_tr, C = self.C)
                    tr_err, cv_err = misclassification_errors(classifier,X_tr,y_tr,X_cv,y_cv)
                    
                y_pred=classifier.predict(X_cv)
                
                if hasattr(metrics,"accuracy_score"):
                    acc = metrics.accuracy_score(y_cv,y_pred)
                else:
                    assert hasattr(metrics,"zero_one_score")
                    acc = metrics.zero_one_score(y_cv, y_pred)
                prec=metrics.precision_score(y_cv,y_pred)
                recall=metrics.recall_score(y_cv,y_pred)
                f1_score=metrics.f1_score(y_cv,y_pred)
                
                tr_err_rfe[i] = tr_err_rfe[i] + tr_err / n_iter
                cv_err_rfe[i] = cv_err_rfe[i] + cv_err / n_iter
                accuracy_rfe[i] = accuracy_rfe[i] + acc / n_iter
                recall_rfe[i] = recall_rfe[i] + recall / n_iter
                precision_rfe[i] = precision_rfe[i] + prec / n_iter
                f1_score_rfe[i] = f1_score_rfe[i] + f1_score / n_iter
                
        return tr_err_rfe, cv_err_rfe,accuracy_rfe,recall_rfe, precision_rfe, f1_score_rfe
                
    
    def select_features(self, features, rfe_curve):
            
        n_features = rfe_curve.shape
            
        t_1_shifted = np.zeros(n_features)
        
        t_1_shifted[1:] = rfe_curve[:-1]
        
        diff = rfe_curve - t_1_shifted
        
        self.mask = diff > 0
        
        new_features = features[:, self.mask]
        
        return new_features, self.mask
    
    
    def apply_features_selection(self, features):
        
        return features[:, self.mask]
        
            
class SVM(object):
       
    
    def __init__(self, kernel = None):
        
        if kernel == SVM_linear or kernel == SVM_RBF_Chi2_squared or kernel == SVM_RBF:
            self.kernel = kernel
    
    
    def model_selection(self,X,y,kernel=SVM_RBF, C_list = None, gamma_list = None, test_size = 0.3, n_iterations = 20,
                        show_accuracy_flag = True, show_precision_flag = True, show_recall_flag = True,
                        show_f1_score_flag = True, max_num_cpus = -1):
        
        assert X!=None and y!=None     
        assert len(X)==len(y)
        assert kernel==SVM_RBF or kernel==SVM_linear or kernel == SVM_RBF_Chi2_squared
        
        if not hasattr(self,"kernel"):
            self.kernel = kernel
            
        if kernel==SVM_linear:
        
            model_selection = ModelSelection(C_list=C_list)
            parameters_result = model_selection.C_selection(X, y, C_list, classifier_by_C_function = linear_SVM_by_C_function, 
                                        error_measure_function = misclassification_errors, 
                                        test_size = test_size, n_iterations = n_iterations, max_num_cpus = max_num_cpus)
        
            self.print_linear_SVM_results(parameters_result, show_accuracy_flag = show_accuracy_flag, 
                                          show_precision_flag = show_precision_flag, 
                                          show_recall_flag = show_recall_flag, 
                                          show_f1_score_flag = show_f1_score_flag)
        
        elif kernel==SVM_RBF:
            
            model_selection = ModelSelection(C_list=C_list,gamma_list=gamma_list)
            parameters_result = model_selection.C_gamma_selection(X, y, 
                                                                  classifier_by_C_and_gamma_function = SVM_RBF_by_C_and_gamma_function, 
                                                                  error_measure_function = misclassification_errors, 
                                                                  test_size = 0.3, n_iterations = n_iterations, max_num_cpus = max_num_cpus)
            
            self.print_SVM_RBF_results(parameters_result, show_accuracy_flag = show_accuracy_flag, 
                                       show_precision_flag = show_precision_flag, 
                                       show_recall_flag = show_recall_flag, 
                                       show_f1_score_flag = show_f1_score_flag )
                    
        elif self.kernel == SVM_RBF_Chi2_squared:
            
            model_selection = ModelSelection(C_list = C_list)
            parameters_result = model_selection.C_selection(X, y, C_list,
                                                                classifier_by_C_function = SVM_RBF_Chi2_squared_by_C_function, 
                                                                error_measure_function = misclassification_errors, 
                                                                test_size = 0.3, n_iterations = n_iterations, max_num_cpus = max_num_cpus)
            
            self.print_linear_SVM_results(parameters_result, show_accuracy_flag = show_accuracy_flag, 
                                       show_precision_flag = show_precision_flag, 
                                       show_recall_flag = show_recall_flag, 
                                       show_f1_score_flag = show_f1_score_flag )
        
        else:
            raise Exception("Unsupported kernel!")
         
        return parameters_result
    
    
    def best_accuracy_C(self, parameters_result):
    
        C_max = None
        acc_max = 0
                    
        for C_cur in xrange(len(parameters_result["C_list"])):
            C_ = parameters_result["C_list"][C_cur]
            acc = parameters_result["acc_by_C"][C_cur]
            if acc>=acc_max:
                acc_max = acc
                C_max = C_
                    
        assert C_max is not None

        return C_max, acc_max
    
    
    def best_accuracy_C_and_gamma(self, parameters_result):
    
        C_max = None
        gamma_max = None
        acc_max = 0
                    
        for C_cur in xrange(len(parameters_result["C_list"])):
            for gamma_cur in xrange(len(parameters_result["gamma_list"])):
                C_ = parameters_result["C_list"][C_cur]
                gamma_ = parameters_result["gamma_list"][gamma_cur]
                acc = parameters_result["acc_by_C_and_gamma"][C_cur, gamma_cur]
                if acc>=acc_max:
                    acc_max = acc
                    C_max = C_
                    gamma_max = gamma_
                    
        assert C_max is not None
        assert gamma_max is not None
        
        return C_max, gamma_max, acc_max
    
    
    def best_accuracy_gamma(self, parameters_result):
    
        gamma_max = None
        acc_max = 0
                    
        for gamma_cur in xrange(len(parameters_result["gamma_list"])):
            gamma_ = parameters_result["gamma_list"][gamma_cur]
            acc = parameters_result["acc_by_C"][gamma_cur]
            if acc>=acc_max:
                acc_max = acc
                gamma_max = gamma_
                    
        assert gamma_max is not None

        return gamma_max, acc_max
    
    
    def print_linear_SVM_results(self, parameters_result, show_accuracy_flag = True, 
                              show_precision_flag = True, show_recall_flag = True,
                              show_f1_score_flag = True):
    
        if show_accuracy_flag:
    
            C_max = None
            acc_max = 0
            
            print "\nAccuracy results:"
            
            for C_cur in xrange(len(parameters_result["C_list"])):
                C_ = parameters_result["C_list"][C_cur]
                acc = parameters_result["acc_by_C"][C_cur]
                print "C = {0}, accuracy = {1}".format(C_,acc)
                if acc>=acc_max:
                    acc_max = acc
                    C_max = C_
                        
            assert C_max is not None
                        
            print "Optimal C = {0}. Accuracy: {1}.".format(C_max, acc_max)
        
        if show_precision_flag:
        
            C_max = None
            prec_max = 0
            
            print "\nPrecision results:"
            
            for C_cur in xrange(len(parameters_result["C_list"])):
                C_ = parameters_result["C_list"][C_cur]
                prec = parameters_result["prec_by_C"][C_cur]
                print "C = {0}, precision = {1}".format(C_,prec)
                if prec>=prec_max:
                    prec_max = prec
                    C_max = C_
                   
            assert C_max is not None
                         
            print "Optimal C = {0}. Precision: {1}.".format(C_max, prec_max)
        
        if show_recall_flag:
        
            C_max = None
            rec_max = 0
            
            print "\nRecall results:"
            
            for C_cur in xrange(len(parameters_result["C_list"])):
                C_ = parameters_result["C_list"][C_cur]
                rec = parameters_result["recall_by_C"][C_cur]
                print "C = {0}, recall = {1}".format(C_, rec)
                if rec>=rec_max:
                    rec_max = rec
                    C_max = C_
                    
            assert C_max is not None
                        
            print "Optimal C = {0}. Recall: {1}.".format(C_max, rec_max)
        
        if show_f1_score_flag:
        
            C_max = None
            f1_max = 0
            
            print "\nf1 score results:"
            
            for C_cur in xrange(len(parameters_result["C_list"])):
                C_ = parameters_result["C_list"][C_cur]
                f1 = parameters_result["f1_by_C"][C_cur]
                print "C = {0}, f1 = {1}".format(C_,f1)
                if f1>=f1_max:
                    f1_max = f1
                    C_max = C_
                        
            assert C_max is not None
                        
            print "Optimal C = {0}. f1 score: {1}.".format(C_max, f1_max)
    
    
    def print_SVM_RBF_results(self, parameters_result, show_accuracy_flag = True, 
                              show_precision_flag = True, show_recall_flag = True,
                              show_f1_score_flag = True):
    
        if show_accuracy_flag:
    
            C_max = None
            gamma_max = None
            acc_max = 0
            
            print "\nAccuracy results:"
            
            for C_cur in xrange(len(parameters_result["C_list"])):
                for gamma_cur in xrange(len(parameters_result["gamma_list"])):
                    C_ = parameters_result["C_list"][C_cur]
                    gamma_ = parameters_result["gamma_list"][gamma_cur]
                    acc = parameters_result["acc_by_C_and_gamma"][C_cur, gamma_cur]
                    print "C = {0}, gamma = {1}, accuracy = {2}".format(C_, gamma_,acc)
                    if acc>=acc_max:
                        acc_max = acc
                        C_max = C_
                        gamma_max = gamma_
                        
            assert C_max is not None
            assert gamma_max is not None
                        
            print "Optimal C = {0} and gamma = {1}. Accuracy: {2}.".format(C_max, gamma_max, acc_max)
        
        if show_precision_flag:
        
            C_max = None
            gamma_max = None
            prec_max = 0
            
            print "\nPrecision results:"
            
            for C_cur in xrange(len(parameters_result["C_list"])):
                for gamma_cur in xrange(len(parameters_result["gamma_list"])):
                    C_ = parameters_result["C_list"][C_cur]
                    gamma_ = parameters_result["gamma_list"][gamma_cur]
                    prec = parameters_result["prec_by_C_and_gamma"][C_cur, gamma_cur]
                    print "C = {0}, gamma = {1}, precision = {2}".format(C_, gamma_,prec)
                    if prec>=prec_max:
                        prec_max = prec
                        C_max = C_
                        gamma_max = gamma_
                        
            assert C_max is not None
            assert gamma_max is not None
                        
            print "Optimal C = {0} and gamma = {1}. Precision: {2}.".format(C_max, gamma_max, prec_max)
        
        if show_recall_flag:
        
            C_max = None
            gamma_max = None
            rec_max = 0
            
            print "\nRecall results:"
            
            for C_cur in xrange(len(parameters_result["C_list"])):
                for gamma_cur in xrange(len(parameters_result["gamma_list"])):
                    C_ = parameters_result["C_list"][C_cur]
                    gamma_ = parameters_result["gamma_list"][gamma_cur]
                    rec = parameters_result["recall_by_C_and_gamma"][C_cur, gamma_cur]
                    print "C = {0}, gamma = {1}, recall = {2}".format(C_, gamma_,rec)
                    if rec>=rec_max:
                        rec_max = rec
                        C_max = C_
                        gamma_max = gamma_
                        
            assert C_max is not None
            assert gamma_max is not None
                        
            print "Optimal C = {0} and gamma = {1}. Recall: {2}.".format(C_max, gamma_max, rec_max)
        
        if show_f1_score_flag:
        
            C_max = None
            gamma_max = None
            f1_max = 0
            
            print "\nf1 score results:"
            
            for C_cur in xrange(len(parameters_result["C_list"])):
                for gamma_cur in xrange(len(parameters_result["gamma_list"])):
                    C_ = parameters_result["C_list"][C_cur]
                    gamma_ = parameters_result["gamma_list"][gamma_cur]
                    f1 = parameters_result["f1_by_C_and_gamma"][C_cur, gamma_cur]
                    print "C = {0}, gamma = {1}, f1 = {2}".format(C_, gamma_,f1)
                    if f1>=f1_max:
                        f1_max = f1
                        C_max = C_
                        gamma_max = gamma_
                        
            assert C_max is not None
            assert gamma_max is not None
                        
            print "Optimal C = {0} and gamma = {1}. f1 score: {2}.".format(C_max, gamma_max, f1_max)
    
    
    def print_SVM_RBF_Chi2_squared_results(self, parameters_result, show_accuracy_flag = True, 
                              show_precision_flag = True, show_recall_flag = True,
                              show_f1_score_flag = True):
    
        self.print_linear_SVM_results(parameters_result, show_accuracy_flag, show_precision_flag, show_recall_flag, show_f1_score_flag)
    
    
    # return tr_err_rfe, cv_err_rfe,accuracy_rfe,recall_rfe, precision_rfe, f1_score_rfe
    def recursive_features_elimination_curves(self, X, y, C = None, gamma = None, kernel=SVM_RBF, n_iterations = 10, test_size = 0.3):
        
        rfe = RecursiveFeaturesElimination(C=C,gamma=gamma,kernel=kernel,
                                            n_iterations=n_iterations,
                                            test_size=test_size)
        return rfe.rfe_curves(X, y)
    
    
    def training(self,X,y,kernel=SVM_RBF,C=None,gamma=None):
        
        assert isinstance(C,(int,float))
        
        if not hasattr(self,"kernel"):
            self.kernel = kernel
        
        if kernel == SVM_RBF:
            assert isinstance(gamma,(int,float))
            self.classifier = SVC(kernel="rbf",C=C,gamma=gamma, class_weight = 'auto')
            self.classifier.fit(X,y)
        elif kernel == SVM_linear:
            self.classifier = LinearSVC(C=C, class_weight = 'auto')
            self.classifier.fit(X,y)
        elif kernel == SVM_RBF_Chi2_squared:
            self.classifier = SVC(kernel=chi2_kernel,C=C, class_weight = 'auto')
            self.classifier.fit(X,y)
        else:
            raise Exception("Classification kernel not supported!")

        return self.classifier
        
    
    def classify(self, X):
        
        assert hasattr(self,"classifier")
                
        return self.classifier.predict(X)
    
    
    def performance_estimation(self, X, y, kernel = SVM_RBF, C = 1.0, gamma = None, n_iterations = 20, test_size = 0.3):
        
        assert isinstance(C,(int,float))
        
        set_ripartitions = StratifiedShuffleSplit(y, n_iter = n_iterations, 
                                                  test_size = test_size, indices = False)
        
        if kernel == SVM_linear:
            classifier = LinearSVC(C=C, class_weight = 'auto')
        elif kernel == SVM_RBF:
            assert isinstance(gamma,(int,float))
            classifier = SVC(kernel="rbf", C=C, gamma=gamma, class_weight = 'auto')
        elif kernel == SVM_RBF_Chi2_squared:
            classifier = SVC(kernel=chi2_kernel,C=C, class_weight = 'auto')
            
        accuracy_avg = 0.0
        precision_avg = 0.0
        recall_avg = 0.0
        f1_score_avg = 0.0
        
        for train,test in set_ripartitions:
            X_tr,X_cv,y_tr,y_cv =X[train],X[test],y[train],y[test]
            classifier.fit(X_tr, y_tr)
            y_pred=classifier.predict(X_cv)
            if hasattr(metrics,"accuracy_score"):
                acc = metrics.accuracy_score(y_cv,y_pred)
            else:
                assert hasattr(metrics,"zero_one_score")
                acc = metrics.zero_one_score(y_cv, y_pred)
            prec=metrics.precision_score(y_cv,y_pred)
            recall=metrics.recall_score(y_cv,y_pred)
            f1_score=metrics.f1_score(y_cv,y_pred)
            
            accuracy_avg = accuracy_avg + acc / n_iterations
            precision_avg = precision_avg + prec / n_iterations
            recall_avg = recall_avg + recall / n_iterations
            f1_score_avg = f1_score_avg + f1_score / n_iterations
            
        return accuracy_avg, precision_avg, recall_avg, f1_score_avg
