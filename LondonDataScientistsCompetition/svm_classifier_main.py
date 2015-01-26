'''
Created on Dec 8, 2013

@author: Alessandro Ferrari

This script performs classification using scikit-learn, numpy and matplotlib on the London Data Scientists Competition on Kaggle.
Classification is perfomed by using an SVM with radial basis functions as kernel. 
The method allows to perform features scaling, pca, model selection and recursive features elimination.
All these settings can be managed as options of the script. Type "python london_data_scientists_competition.py -h" for more informations.

Examples of usage:

First, add in the python path the path of the library. 
For linux just types in the terminal:
>>PYTHONPATH=$PYTHONPATH:/path/to/folder/impara
>>export PYTHONPATH

Move the terminal current working directory to the library root folder:
>>cd /path/to/folder/impara

By typing:
>>python LondonDataScientistsCompetition/svm_classifier_main.py -h

in the root folder of the project you will have a complete description and synopsys of the
features currently available. The data concerning the competition are available in the folder 
LondonDataScientistsCompetion/data. 

Example 1, first attempt (default C_list and gamma_list values, overnight mode), it can take a while:
>>python LondonDataScientistsCompetition/svm_classifier_main.py --pca-enabled --features-scaling-enabled --n-iterations-ms=20 --sparse-filtering --rf-features-selection --rfe-features-selection --n-iterations-rfe=20 --pca-variance-retain=0.85 --overnight-simulation ./LondonDataScientistsCompetition/data ./LondonDataScientistsCompetition/data

Example 2, first attempt (default C_list and gamma_list values, interactive mode):
>>python LondonDataScientistsCompetition/svm_classifier_main.py --pca-enabled --features-scaling-enabled --n-iterations-ms=20 --sparse-filtering --rf-features-selection --rfe-features-selection --n-iterations-rfe=20 --pca-variance-retain=0.85 ./LondonDataScientistsCompetition/data ./LondonDataScientistsCompetition/data

Example 3, refining model selection (refining model selection):
>>python LondonDataScientistsCompetition/svm_classifier_main.py --pca-enabled --features-scaling-enabled --n-iterations-ms=20 --sparse-filtering --rf-features-selection --rfe-features-selection --n-iterations-rfe=20 --pca-variance-retain=0.85 --C-list=[300,500,700,800,900,1000,1300,2000,3000,5000,7000] --gamma-list=[0.005,0.007,0.008,0.01,0.02,0.03,0.05,0.07,0.09] ./LondonDataScientistsCompetition/data ./LondonDataScientistsCompetition/data

Example 4, skipping model selection
>>python LondonDataScientistsCompetition/svm_classifier_main.py --pca-enabled --features-scaling-enabled --sparse-filtering --rf-features-selection --pca-variance-retain=0.85 --overnight-simulation --skip-model-selection --C=3000 --gamma=0.008 ./LondonDataScientistsCompetition/data ./LondonDataScientistsCompetition/data
'''

import os
import sys
import json
from datetime import date
import pickle
import getopt
import random
import classification
import numpy as np
from classification import SVM, dataset_scaling, RecursiveFeaturesElimination, SVM_RBF, SVM_linear, SVM_RBF_Chi2_squared, classes_balance
import matplotlib.pyplot as plt
from utilities import plot_3d, plot_2d, save_csv_submitted_labels, plot_features, plot_rfe_curve, load_sf_features, save_sf_features, print_model_selection_results
from features_preprocessing import Scaler
from dimensionality_reduction import PCA
from sparse_filtering import SparseFilter
from features_filtering import FeaturesSelectionRandomForests


def parse_list_string(strlist):
    
    assert isinstance(strlist,basestring)
    
    assert strlist[0] == "["
    assert strlist[-1] == "]"
    
    strvalues = strlist[1:-1].split(",")
    
    vlist = []
    for vstr in strvalues:
        try:
            v = float(vstr)
        except Exception as e:
            raise ValueError("Error while converting {0} to float. List in input {1} has to be in the format {2}.".format(vstr,strlist,e))
        vlist.append(v)
        
    return vlist


def parse_option():

    params_dict = dict()

    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help", "C=", "gamma=", 
                                                          "pca-enabled", "pca-disabled",
                                                          "pca-variance-retain=",
                                                          "features-scaling-enabled",
                                                          "features-scaling-disabled",
                                                          "C-list=","gamma-list=",
                                                          "rbf-svm", "linear-svm",
                                                          "rbf-chi2-svm",
                                                          "skip-model-selection",
                                                          "n-iterations-ms=", #number iterations model selection
                                                          "n-iterations-performance-estimation=",
                                                          "sparse-filtering",
                                                          "save-sf-features=",
                                                          "load-sf-features=",
                                                          "n-layers-sf=",
                                                          "n-iterations-sf=",
                                                          "n-features-sf=",
                                                          "rf-features-selection", #random forests features selection
                                                          "rfe-features-selection", 
                                                          "n-iterations-rfe=", #number iterations recursive features elimination
                                                          "overnight-simulation",
                                                          "show-precision-enabled",
                                                          "show-precision-disabled",
                                                          "show-accuracy-enabled",
                                                          "show-accuracy-disabled",
                                                          "show-recall-enabled",
                                                          "show-recall-disabled",
                                                          "show-f1score-enabled",
                                                          "show-f1score-disabled",
                                                          "show-trerr-enabled",
                                                          "show-trerr-disabled",
                                                          "show-cverr-enabled",
                                                          "show-cverr-disabled",
                                                          "dataset-name=",
                                                          "test-set-enabled",
                                                          "test-set-disabled",
                                                          "training-set-fn=",
                                                          "target-set-fn=",
                                                          "test-set-fn="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        sys.exit(2)
    output = None
    verbose = False
    
    options_list = []
    for o, a in opts:
        options_list.append(o)
    
    for o, a in opts:
        
        if o == "--C":
            
            if not "--skip-model-selection" in options_list:
                print "Warning: C specified even if model selection is enabled! Specify --skip-model-selection option to force C and gamma parameters"
                sys.exit(2)
            try:
                params_dict["C"] = float(a)
            except Exception as e:
                print "Invalid value of C: {0}. Exception: {1}.".format(a,e)
                sys.exit(2)
        
        elif o == "--gamma":
            
            if not "--skip-model-selection" in options_list:
                print "Warning: gamma specified even if model selection is enabled! Specify --skip-model-selection option to force C and gamma parameters"
                sys.exit(2)
                
            if "--linear-svm" in options_list:
                print "--gamma specified even if --linear-svm is enabled."
                sys.exit(2)
                
            try:
                params_dict["gamma"] = float(a)
            except Exception as e:
                print "Invalid value of gamma: {0}. Exception: {1}.".format(a,e)
                sys.exit(2)
                
        elif o == "--pca-enabled":
            
            if "--pca-disabled" in options_list:
                print "--pca-enabled and --pca-disabled specified at the same time."
                sys.exit(2)
                
            params_dict["pca_flag"] = True
        
        elif o == "--pca-disabled":
            
            if "--pca-enabled" in options_list:
                print "--pca-enabled and --pca-disabled specified at the same time."
                sys.exit(2)
                
            params_dict["pca_flag"] = False
            
        elif o == "--pca-variance-retain":
            
            if not "--pca-enabled" in options_list:
                print "--pca-variance-retain specified while --pca-enable not specified."
                sys.exit(2)
                
            try:
                params_dict["pca_variance_retain"] = float(a)
            except Exception as e:
                print "Error while converting --pca-variance-retain to float. Exception: {0}".format(e)
                sys.exit(2)
                
            if not 0 <= params_dict["pca_variance_retain"] <= 1:
                print "--pca-variance-retain={0} has to be between 0 and 1.".format(params_dict["pca_variance_retain"])
                sys.exit(2)
                        
        elif o == "--features-scaling-enabled":
            
            if "--features-scaling-disabled" in options_list:
                print "--features-scaling-enabled and --features-scaling-disabled specified at the same time."
                sys.exit(2)
                
            params_dict["scaling_flag"] = True
            
        elif o == "--features-scaling-disabled":
            
            if "--features-scaling-enabled" in options_list:
                print "--features-scaling-enabled and --features-scaling-disabled specified at the same time."
                sys.exit(2)
                
            params_dict["scaling_flag"] = False
            
        elif o == "--skip-model-selection" in options_list:
            
            params_dict["skip_model_selection"] = True
            
        elif o == "--rbf-svm" in options_list:
            
            if "--linear-svm" in options_list or "--rbf-chi2-svm" in options_list:
                print "--linear-svm  or --rbf-chi2-svm is enabled when --rbf-svm is enabled."
                sys.exit(2)
            
            params_dict["kernel"] = SVM_RBF
            
        elif o == "--linear-svm" in options_list:
            
            if "--rbf-chi2-svm" in options_list or "--rbf-svm" in options_list:
                print "--rbf-chi2-svm  or --rbf-svm is enabled when --linear-svm is enabled."
                sys.exit(2)
            
            params_dict["kernel"] = SVM_linear
            
        elif o == "--rbf-chi2-svm" in options_list:
            
            if "--linear-svm" in options_list or "--rbf-svm" in options_list:
                print "--linear-svm  or --rbf-svm is enabled when --rbf-chi2-svm is enabled."
                sys.exit(2)
            
            params_dict["kernel"] = SVM_RBF_Chi2_squared

        elif o == "--n-iterations-ms":
            
            if "--skip-model-selection" in options_list:
                print "--n-iterations-ms specified even if --skip-model-selection specified."
                sys.exit(2)
            try:
                params_dict["n_iterations_ms"] = int(a)
            except Exception as e:
                print "Error while converting --n-iterations-ms={0} to int.".format(a)
                sys.exit(2)
            if not 0<params_dict["n_iterations_ms"]<100:
                print "Error: 0 < --n-iterations-ms < 100."
                sys.exit(2)

        elif o == "n-iterations-performance-estimation=":
            
            try:
                params_dict["n_iterations_performance_estimation"] = int(float(a))
            except Exception as e:
                print "Error while converting --n-iterations-performance-estimation={0} to int.".format(a)
                sys.exit(2)
                
        elif o == "--C-list":
            
            if "--skip-model-selection" in options_list:
                print "C_list specified even if --skip-model-selection is enabled."
                sys.exit(2)
            
            try:
                params_dict["C_list"] = parse_list_string( a )
            except Exception as e:
                print "--C-list argument expressed wrong. {0}".format(e)
                sys.exit(2)
            
        elif o == "--gamma-list":
        
            if "--skip-model-selection" in options_list:
                print "gamma_list specified even if --skip-model-selection is enabled."
                sys.exit(2)
                
            if "--linear-svm" in options_list:
                print "gamma_list specified even if --linear-svm is enabled."
                sys.exit(2)
        
            try:
                params_dict["gamma_list"] = parse_list_string( a )
            except Exception as e:
                print "--gamma-list argument expressed wrong. {0}".format(e)
                sys.exit(2)
                
        elif o == "--sparse-filtering":
            
            params_dict["sparse_filtering_flag"] = True
            
        elif o == "--load-sf-features":
            
            if not "--sparse-filtering" in options_list:
                print "Cannot enable --load-sf-features if --sparse-filtering is not specified!"
                sys.exit(2)
            
            if a[0]=='.':
                a = os.path.join(os.getcwd(),a[2:])
            if not os.path.exists(a):
                print "Invalid path specified for loading sparse filtering features."
                sys.exit(2)
                
            params_dict["load_sf_flag"] = True
            params_dict["load_sf_path"] = a
            
        elif o == "--save-sf-features":
            
            if not "--sparse-filtering" in options_list:
                print "Cannot enable --save-sf-features if --sparse-filtering is not specified!"
            
            if a[0]=='.':
                a = os.path.join(os.getcwd(),a[2:])
            if not os.path.exists(os.path.dirname(a)):
                print "Invalid path specified for saving sparse filtering features.{0}".format(os.path.dirname(a))
                sys.exit(2)
            
            params_dict["save_sf_flag"] = True
            params_dict["save_sf_path"] = a
            
        elif o == "--n-iterations-sf":
            
            if not "--sparse-filtering" in options_list:
                print "--n-iterations-sf specified even if --sparse-filtering is not enabled."
                sys.exit(2)
                
            try:
                params_dict["n_iterations_sf"] = int(a)
            except Exception as e:
                print "Error while converint --n-iterations-fs={0} to int. Exception: {1}".format(a,e)
                sys.exit(2)
            
            if not 0<params_dict["n_iterations_sf"]<=100000:
                print "Error: --n-iterations-sf has to be greater than 0 and smaller than 100000."
                sys.exit(2)
            
        elif o == "--n-features-sf":
            
            if not "--n-features-sf" in options_list:
                print "--n-features-sf specified even if --sparse-filtering is not enabled."
                sys.exit(2)
                
            try:
                params_dict["n_features_sf"] = int(a)
            except Exception as e:
                print "Error while converting --n-features-fs={0} to int. Exception: {1}".format(a,e)
                sys.exit(2)
                
            if not 0<params_dict["n_features_sf"]<=100000:
                print "Error: --n-features-sf has to be greater than 0 and smaller than 100000."
                sys.exit(2)

            try:
                params_dict["n_layers_sf"] = int(a)
            except Exception as e:
                print "Error while converting --n-layers-fs={0} to int. Exception: {1}".format(a,e)
                sys.exit(2)
                
            if not 0<params_dict["n_layers_sf"]<=100:
                print "Error: --n-layers-sf has to be greater than 0 and smaller than 100."
                sys.exit(2)

        elif o == "--n-layers-sf":

            if not "--n-layers-sf" in options_list:
                print "--n-layers-sf specified even if --sparse-filtering is not enabled."
                sys.exit(2)	
                
        elif o == "--rf-features-selection":
            
            params_dict["rf_features_selection_flag"] = True
            
        elif o == "--rfe-features-selection":
            
            params_dict["rfe_features_selection_flag"] = True
            
        elif o == "--n-iterations-rfe":
        
            if not "--rfe-features-selection" in options_list:
                print "--n-iterations-rfe specified even if --rfe-features-selection is not specified."
                sys.exit(2)
        
            try:
                params_dict["n_iterations_rfe"] = int(a)
            except Exception as e:
                print "Error while converting --n-iterations-rfe={0} to int.".format(a)
                sys.exit(2)
                
            if not 0<params_dict["n_iterations_rfe"]<100:
                print "Error: 0 < --n-iterations-rfe < 100."
                sys.exit(2)
        
        elif o == "--overnight-simulation":
            
            params_dict["overnight_simulation"] = True
        
        elif o == "--show-precision-enabled":
            
            if "--show-precision-disabled" in options_list:
                print "--show-precision-enabled specified meanwhile --show-precision-disabled specified."
                sys.exit(2)
                
            params_dict["show_precision_flag"] = True
            
        elif o == "--show-precision-disabled":
        
            if "--show-precision-enabled" in options_list:
                print "--show-precision-disabled specified meanwhile -show-precision-enabled specified."
                sys.exit(2)
                
            params_dict["show_precision_flag"] = False
            
        elif o == "--show-accuracy-enabled":
            
            if "--show-accuracy-disabled" in options_list:
                print "--show-accuracy-enabled specified meanwhile --show-accuracy-disabled specified."
                sys.exit(2)
                
            params_dict["accuracy_flag"] = True
            
        elif o == "--show-accuracy-disabled":
        
            if "--show-accuracy-enabled" in options_list:
                print "--show-accuracy-disabled specified meanwhile -show-accuracy-enabled specified."
                sys.exit(2)
                
            params_dict["show_accuracy_flag"] = False
            
        elif o == "--show-recall-enabled":
            
            if "--show-recall-disabled" in options_list:
                print "--show-recall-enabled specified meanwhile --show-recall-disabled specified."
                sys.exit(2)
                
            params_dict["show_recall_flag"] = True
            
        elif o == "--show-recall-disabled":
        
            if "--show-recall-enabled" in options_list:
                print "--show-recall-disabled specified meanwhile --show-recall-enabled specified."
                sys.exit(2)
                
            params_dict["show_recall_flag"] = False
        
        elif o == "--show-f1score-enabled":
            
            if "--show-f1score-disabled" in options_list:
                print "--show-f1score-enabled specified meanwhile --show-f1score-disabled specified."
                sys.exit(2)
                
            params_dict["show_f1score_flag"] = True
            
        elif o == "--show-f1score-disabled":
        
            if "--show-f1score-enabled" in options_list:
                print "--show-f1score-disabled specified meanwhile --show-f1score-enabled specified."
                sys.exit(2)
                
            params_dict["show_f1score_flag"] = False
                    
        elif o == "--show-trerr-enabled":
            
            if "--show-trerr-disabled" in options_list:
                print "--show-trerr-enabled specified meanwhile --show-trerr-disabled specified."
                sys.exit(2)
                
            params_dict["show_trerr_flag"] = True
            
        elif o == "--show-trerr-disabled":
        
            if "--show-trerr-enabled" in options_list:
                print "--show-trerr-disabled specified meanwhile --show-trerr-enabled specified."
                sys.exit(2)
                
            params_dict["show_trerr_flag"] = False

        elif o == "--show-cverr-enabled":
            
            if "--show-cverr-disabled" in options_list:
                print "--show-cverr-enabled specified meanwhile --show-cverr-disabled specified."
                sys.exit(2)
                
            params_dict["show_cverr_flag"] = True
            
        elif o == "--show-cverr-disabled":
        
            if "--show-cverr-enabled" in options_list:
                print "--show-cverr-disabled specified meanwhile --show-cverr-enabled specified."
                sys.exit(2)
                
            params_dict["show_cverr_flag"] = False
            
        elif o == "--dataset-name":
            
            params_dict["dataset_name"] = a
            
        elif o == "--training-set-fn":
            
            params_dict["training_set_fn"] = a
            
        elif o == "--target-set-fn":
            
            params_dict["target_set_fn"] = a
            
        elif o == "--test-set-fn":
            
            params_dict["test_set_fn"] = a

        elif o == "--test-set-enabled":

            params_dict["test_set_flag"] = True
            
        elif o == "--test-set-disabled":
            
            params_dict["test_set_flag"] = False
    
        elif o == "-h" or o == "--help":
            
            description = "This script performs classification using scikit-learn, numpy and matplotlib on the London Data Scientists Competition on Kaggle."+\
                            " Classification is perfomed by using an SVM with radial basis functions as kernel. The method allows to perform features scaling,"+\
                            " pca, model selection and recursive features elimination. All these settings can be managed as options of the script.\n"

            print description
            
            cmd_synopsys = "python london_data_scientists_competition.py [ { --rbf-svm | --linear-svm | --rbf-chi2-svm } ] [--C=numeric] [--gamma=numeric] [ { --pca-enabled | --pca-disabled } ]"+\
                            " [ --pca-variance-retain=[0.0..1.0] ] [ { --features-scaling-enabled | --features-scaling-disabled } ]"+\
                            " [ --C-list=[0.01,0.1,1,..] ] [ --gamma-list=[0.01,0.1,1,..] ] [ --skip-model-selection ] [ --n-iterations-ms=[1..100] ]"+\
                            " [ { --show-precision-enabled | --show-precision-disabled } ] [ { --show-accuracy-enabled | --show-accuracy-disabled } ]"+\
                            " [ { --show-recall-enabled | --show-recall-disabled } ] [ { --show-f1score-enabled | --show-f1score-disabled] }"+\
                            " [ { --show-trerr-enabled | --show-trerr-disabled } ] [ { --show-cverr-enabled | --show-cverr-disabled] }"+\
                            " [--sparse-filtering] [--n-iterations-sf=[0..100000]] [--n-features-sf=[0..10000]] [--rf-features-selection] [--rfe-features-selection] [--n-iterations-rfe=[0..100]]"+\
                            " [ --training-set-fn features.txt ] [ --target-set-fn labels.txt ] [ --test-set-fn test.txt ] [ { --test-set-enabled | --test-set-disabled } ]"+\
                            " source_path destination_path\n"
            print cmd_synopsys
            print "--rbf-svm : specified to use a radial basis function kernel. It requires the definition of the parameters C and gamma, that may be defined by model selection, or they may be indicated directly. It is mutual exclusive respect with --linear-svm."
            print "--rbf-chi2-svm : specified to use a Chi squared kernel. It requires the definition of the parameter C and gamma, that may be defined by model selection, or they may be indicated directly. It is mutual exclusive respect with --linear-svm."
            print "--linear-svm : specified to use a linear kernel svm. It is enabled by default. If it is enabled, --gamma and --gamma-list cannot be specified. It is mutual exclusive to --rbf-svm."
            print "--C : specifies the regularization parameter of the SVM."
            print "It has to be specified only if --skip-model-selection, otherwise C should be indicate after the model selection process.\n"
            print "--gamma : specifies the gamma of the rbf kernel of the SVM."
            print "It has to be specified only if --skip-model-selection is specified, otherwise gamma should be indicate after the model selection process.\n"
            print "--pca-enabled : it enables linear principal component analysis on the features of the dataset." 
            print "It is mutually exclusive to --pca-disabled.\n"
            print "--pca-disabled: it disables linear principal component analysis on the features of the dataset." 
            print "It is mutually exclusive to --pca-disabled.\n"
            print "--pca-variance-retain : specifies the variance retained by the pca. It has to be float. It has to be between 0.0 and 1.0."
            print "Its default value is 1.0. It has to be specified together with --pca-enabled.\n"
            print "--features-scaling-enabled : it enables features whitening. It is mutually exclusive to --features-scaling-disabled.\n"
            print "--features-scaling-disabled : it disables features whitening. It is mutually exclusive to --features-scaling-enabled.\n"
            print "--C-list : specifies the list of C values used in model selection in the format [0.001,0.01,0.1,1,10]. --skip-model-selection must not be activated.\n"
            print "--gamma-list : specifies the list of gamma values used in model selection in the format [0.001,0.01,0.1,1,10]. --skip-model-selection must not be activated.\n"
            print "--skip-model-selection : this flag allows to skip model selection. In this case, C and gamma parameters can be expressed as command line parameters. --C-list and --gamma-list must not be specified.\n"
            print "--n-iterations-ms: this flag allows to specify the number of iterations to use for averaging in model selection for parameters selection. The number of iterations specified has to be an integer greater than 0 and smaller than 100."
            print "--n-iterations-performance-estimation: it allows to specify the number of iterations for estimating performances of the classifier. By default is 20."
            print "It has to be specified only if --skip-model-selection is not specified. Its default value is 6.\n"
            print "--sparse-filtering: this flag enables sparse filtering for strong features generation.\n"
            print "--n-iterations-sf : this allows to specify the number of iteration for sparse filtering optimization. By default 1000. It has to be specified when --sparse-filtering is enabled.\n"
            print "--n-features-sf : this allows to specify the number of features for sparse filtering. By default 50. It has to be specified when --sparse-filtering is enabled.\n"
            print "--n-layers-sf : this allows to specify the number of layers trained as greedy layer-wise network with sparse filtering. By default 50. It has to be specified when --sparse-filtering is enabled.\n"
            print "--save-sf-features: since sparse filtering is time consuming, it is possible to save the calculated features and re-load them in the nexts run. It has to specified with --sparse filtering enabled.\n"
            print "--load-sf-features: since sparse filtering is time consuming, it is possible to load pre-calculated features. It has to specified with --sparse filtering enabled.\n"
            print "--rf-features-selection: this flag enables strong features selection by means of random forests.\n"
            print "--rfe-features-selection: this flag enables strong features selection by means of recursive features elimination.\n"
            print "--n-iterations-rfe : this flag allows to specify the number of iterations for averaging results of recursive features elimination.The number of iterations specified has to be an integer greater than 0 and smaller than 100."
            print "It has to be specified only if --rfe-features-selection is specified. Its default value is 5.\n"
            print "--overnight-simulation : the simulation runs without plottings graph or asking input at the user, selecting automatically parameters.\n"
            print "--show-accuracy-enabled : this flag enables plotting graph and statistics about accuracy results. Default is activated.\n"
            print "--show-accuracy-disabled : this flag disables plotting graph and statistics about accuracy results. Default is activated.\n"
            print "--show-precision-enabled : this flag enables plotting graph and statistics about precision results. Default is disactivated.\n"
            print "--show-precision-disabled : this flag disables plotting graph and statistics about precision results. Default is disactivated.\n"
            print "--show-recall-enabled : this flag enables plotting graph and statistics about recall results. Default is disactivated.\n"
            print "--show-recall-disabled : this flag disables plotting graph and statistics about recall results. Default is disactivated.\n"
            print "--show-f1score-enabled : this flag enables plotting graph and statistics about f1 score results. Default is activated.\n"
            print "--show-f1score-disabled : this flag disables plotting graph and statistics about f1 score results. Default is activated.\n"
            print "--show-trerr-enabled : this flag enables plotting graph and statistics about training error results. Default is disactivated.\n"
            print "--show-trerr-disabled : this flag disables plotting graph and statistics about training error results. Default is disactivated.\n"
            print "--show-cverr-enabled : this flag enables plotting graph and statistics about cross-validation error results. Default is disactivated.\n"
            print "--show-cverr-disabled : this flag disables plotting graph and statistics about cross-validation error results. Default is disactivated.\n"
            print "--datast-name: name of the dataset to be used when saving the model and the results."
            print "--training-set-fn : specifies the filename of the training set."
            print "--target-set-fn : specifies the filename of the labels set."
            print "--test-set-fn: specifies the filename of the test set, whether enabled."
            print "--test-set-enabled: it enables the use of the test set. By default is enabled."
            print "--test-set-disabled: it disabled the use of the test set. By default is enabled."
            
            sys.exit(0)
            
        else:
            print "Unrecognized option: {0}. For details: python svm_classifier_main.py -h ".format(o)
            sys.exit(2)
    
    if len(sys.argv) < 3:
        print "Error while specifying the arguments. python svm_classifier_main.py <sourcedir> <destdir>"
    
    params_dict["source_path"] = os.path.expanduser(args[-2])
    params_dict["source_path"] = params_dict["source_path"].replace(".",os.getcwd())
    params_dict["dest_path"] = os.path.expanduser(args[-1])
    params_dict["dest_path"] = params_dict["dest_path"].replace(".",os.getcwd())
    
    if not os.path.exists(params_dict["source_path"]):
        raise Exception("The specified source path does not exist! Source path: {0}.".format(params_dict["source_path"]))
    
    if not os.path.exists(params_dict["dest_path"]):
        raise Exception("The specified destination path does not exist! Destination path: {0}.").format(params_dict["dest_path"])
        
    if not params_dict.has_key("training_set_fn"):
        params_dict["training_set_fn"] = "train.csv"
    
    if not params_dict.has_key("target_set_fn"):
        params_dict["target_set_fn"] = "trainLabels.csv"
    
    if not params_dict.has_key("test_set_fn"):
        params_dict["test_set_fn"] = "test.csv"
    
    if not params_dict.has_key("predicted_set_fn"):
        params_dict["predicted_set_fn"] = "predicted_y.csv"
            
    if not params_dict.has_key("scaling_flag"):
        params_dict["scaling_flag"] = True
    
    if not params_dict.has_key("pca_flag"):
        params_dict["pca_flag"] = True
        
    if not params_dict.has_key("pca_variance_retain"):
        params_dict["pca_variance_retain"] = 1.0
        
    if not params_dict.has_key("kernel"):
        params_dict["kernel"] = SVM_linear
        
    if not params_dict.has_key("skip_model_selection"):
        params_dict["skip_model_selection"] = False
 
    if not params_dict.has_key("sparse_filtering_flag"):
        params_dict["sparse_filtering_flag"] = False
        
    if not params_dict.has_key("n_iterations_sf"):
        params_dict["n_iterations_sf"] = 1000
        
    if not params_dict.has_key("n_iterations_performance_estimation"):
        params_dict["n_iterations_performance_estimation"] = 20
        
    if not params_dict.has_key("n_features_sf"):
        params_dict["n_features_sf"] = 50

    if not params_dict.has_key("n_layers_sf"):
        params_dict["n_layers_sf"] = 1
        
    if not params_dict.has_key("load_sf_flag"):
        params_dict["load_sf_flag"] = False

    if not params_dict.has_key("save_sf_flag"):
        params_dict["save_sf_flag"] = False
            
    if not params_dict.has_key("rf_features_selection_flag"):
        params_dict["rf_features_selection_flag"] = False
        
    if not params_dict.has_key("rfe_features_selection_flag"):
        params_dict["rfe_features_selection_flag"] = False
        
    if not params_dict.has_key("n_iterations_ms"):
        params_dict["n_iterations_ms"] = 6
        
    if not params_dict.has_key("n_iterations_rfe"):
        params_dict["n_iterations_rfe"] = 5
        
    if not params_dict.has_key("overnight_simulation"):
        params_dict["overnight_simulation"] = False
        
    if not params_dict.has_key("show_accuracy_flag"):
        params_dict["show_accuracy_flag"] = True

    if not params_dict.has_key("show_precision_flag"):
        params_dict["show_precision_flag"] = False

    if not params_dict.has_key("show_recall_flag"):
        params_dict["show_recall_flag"] = False

    if not params_dict.has_key("show_f1score_flag"):
        params_dict["show_f1score_flag"] = True

    if not params_dict.has_key("show_trerr_flag"):
        params_dict["show_trerr_flag"] = False

    if not params_dict.has_key("show_cverr_flag"):
        params_dict["show_cverr_flag"] = False

    if not params_dict.has_key("test_set_flag"):
        params_dict["test_set_flag"] = True
        
    if not params_dict.has_key("dataset_name"):
        params_dict["dataset_name"] = "generic"

    return params_dict


def main():

    params_dict = parse_option()

    testdir = params_dict["source_path"]
    
    # Read data
    train = np.genfromtxt(open(os.path.join(testdir,params_dict['training_set_fn']),'rb'), delimiter=',')
    print "Number of training samples: {0}.".format(train.shape[0])
    print "Number of features: {0}.".format(train.shape[1])
    target = np.genfromtxt(open(os.path.join(testdir,params_dict['target_set_fn']),'rb'), delimiter=',')
    len_train_set = train.shape[0]
    if params_dict["test_set_flag"]:
        test = np.genfromtxt(open(os.path.join(testdir, params_dict['test_set_fn']),'rb'), delimiter=',')
        len_test_set = test.shape[0]
        
    if not params_dict["overnight_simulation"]:
        print "Visualizing features for understanding the most suitable scaling type."
        if params_dict["test_set_flag"]:
            plot_features(np.vstack((train,test)))
        else:
            plot_features(train)
        plt.show()
    
    balances = classes_balance(target)
    counter = 0
    for b in balances:
        print "For class {0} the balance is {1:.4f}.".format(counter, b)
        counter += 1
    
    n_feat = train.shape[1]
    num_samples = train.shape[0] 
    
    #features scaling
    print "Starting features preprocessing ..."
    
    if params_dict["sparse_filtering_flag"]:
        
        print "Performing sparse filtering..."
        
        if params_dict["load_sf_flag"]:
            sf, train_sf, test_sf = load_sf_features(params_dict["load_sf_path"])
        else:
            sf = SparseFilter(n_layers=params_dict["n_layers_sf"],n_features=params_dict["n_features_sf"], n_iterations=params_dict["n_iterations_sf"])
            sf.fit(np.r_[train,test])
            train_sf = sf.transform(train)
            test_sf = sf.transform(test)
        
        if params_dict["save_sf_flag"]:
            save_sf_features(sf, train_sf, test_sf, params_dict["save_sf_path"])
        
        print "Features sparse filtering performed!"
        
        print train_sf.shape
    
    if params_dict["test_set_flag"]:
        dataset = np.r_[train, test]
    else:
        dataset = train
        
    if params_dict["pca_flag"]:
        
        print "Performing PCA..."
        
        pca = PCA(variance_retain = params_dict["pca_variance_retain"])
        pca.fit(dataset)
        dataset_pca = pca.transform(dataset)
        if params_dict["test_set_flag"]:
            train_pca = dataset_pca[:len_train_set,:]
            test_pca = dataset_pca[len_train_set:,:]
        else:
            train_pca = dataset_pca
    
        n_feat_pca = dataset_pca.shape[1]
        print "Number of features after PCA: {0}.".format(n_feat_pca)
    
    else:
    
        dataset_pca = dataset
        train_pca = train
        if params_dict["test_set_flag"]:
            test_pca = test
            
        n_feat_pca = dataset_pca.shape[1]
        print "Number of features after PCA: {0}.".format(n_feat_pca)
    
    if params_dict["pca_flag"]:
        
        if not params_dict["overnight_simulation"]:
            print "Visualizing features after PCA..."
        
            plot_features(dataset_pca)
            plt.show()
            
    if params_dict["scaling_flag"]:
        scaler = Scaler(bias_and_variance_flag = True, log10_flag = False, log2_flag = False, log1p_flag = False)
        if params_dict["test_set_flag"]:
            dataset_scaled = scaler.fit(np.r_[train_pca,test_pca])
            train_scaled = dataset_scaled[:len_train_set,:]
            test_scaled = dataset_scaled[len_train_set:,:]
        else:
            dataset_scaled = scaler.fit(train_pca)
            train_scaled = dataset_scaled
    else:
        train_scaled = train_pca
        if params_dict["test_set_flag"]:
            test_scaled = test_pca
    
    if params_dict["scaling_flag"]:
        
        if not params_dict["overnight_simulation"]:
            print "Visualizing features after features preprocessing.."
    
            plot_features(dataset_scaled)
            plt.show()
    
    if params_dict["sparse_filtering_flag"]:
        
        train_data = np.c_[train_scaled, train_sf]
        if params_dict["test_set_flag"]:
            test_data = np.c_[test_scaled, test_sf]
    
    else:
        
        train_data = train_scaled
        if params_dict["test_set_flag"]:
            test_data = test_scaled
        
    print "Features preprocessing done!"
    print train_data.shape
    if params_dict["rf_features_selection_flag"]:
        
        print "Starting features selection by means of random forests..."
        
        fsrf = FeaturesSelectionRandomForests()
        fsrf.fit(train_data, target)
        
        if not params_dict["overnight_simulation"]:
            fsrf.plot_features_importance()
        
        fsrf_mask = fsrf.features_mask
        
        train_data = fsrf.transform(train_data)
        if params_dict["test_set_flag"]:
            test_data = fsrf.transform(test_data)
        
        n_feat_fsrf = train_data.shape[1]
        
        print "Random forests features selection done!"
    
    classification_obj=SVM()
    
    if not params_dict["skip_model_selection"]:
    
        print "Starting model selection ..."
        
        if not params_dict.has_key("C_list"):
            C_list = [0.0001, 0.001,0.01,0.1,1,10,100,1000,10000]
        else:
            C_list = params_dict["C_list"]
        
        if params_dict["kernel"] == SVM_RBF: 
            if not params_dict.has_key("gamma_list"):
                gamma_list = [0.0001, 0.001,0.01,0.1,1,10,100,1000,10000]
            else:
                gamma_list = params_dict["gamma_list"]
        else:
            gamma_list = None
            
        #performing model selection
        ms_result = classification_obj.model_selection(train_data,target, kernel = params_dict["kernel"], 
                                                       n_iterations=params_dict["n_iterations_ms"],
                                                       C_list = C_list,
                                                       gamma_list = gamma_list,
                                                       show_accuracy_flag = params_dict["show_accuracy_flag"], 
                                                       show_precision_flag = params_dict["show_precision_flag"], 
                                                       show_recall_flag = params_dict["show_recall_flag"], 
                                                       show_f1_score_flag = params_dict["show_f1score_flag"])
        
        if not params_dict["overnight_simulation"]:
            #displaying model selection
            if params_dict["kernel"] == SVM_RBF:
                if params_dict["show_accuracy_flag"]:
                    plot_3d(x=ms_result["gamma_list"], y=ms_result["C_list"], z=ms_result["acc_by_C_and_gamma"], zlabel="accuracy", title="Accuracy by C and gamma")
                if params_dict["show_precision_flag"]:
                    plot_3d(x=ms_result["gamma_list"], y=ms_result["C_list"], z=ms_result["recall_by_C_and_gamma"], zlabel="recall", title="Recall by C and gamma")
                if params_dict["show_recall_flag"]:
                    plot_3d(x=ms_result["gamma_list"], y=ms_result["C_list"], z=ms_result["prec_by_C_and_gamma"], zlabel="precision", title="Precision by C and gamma")
                if params_dict["show_f1score_flag"]:
                    plot_3d(x=ms_result["gamma_list"], y=ms_result["C_list"], z=ms_result["f1_by_C_and_gamma"], zlabel="accuracy", title="f1 score by C and gamma")
                if params_dict["show_trerr_flag"]:
                    plot_3d(x=ms_result["gamma_list"], y=ms_result["C_list"], z=ms_result["tr_err_by_C_and_gamma"], zlabel="training error", title="Training error score by C and gamma")
                if params_dict["show_cverr_flag"]:
                    plot_3d(x=ms_result["gamma_list"], y=ms_result["C_list"], z=ms_result["cv_err_by_C_and_gamma"], zlabel="cross-validation error", title="Cross-validation error score by C and gamma")
            elif params_dict["kernel"] == SVM_linear  or params_dict["kernel"] == SVM_RBF_Chi2_squared:
                if params_dict["show_accuracy_flag"]:
                    plot_2d(x=ms_result["C_list"], y=ms_result["acc_by_C"], ylabel="accuracy", title="Accuracy by C")
                if params_dict["show_precision_flag"]:
                    plot_2d(x=ms_result["C_list"], y=ms_result["recall_by_C"], ylabel="recall", title="Recall by C")
                if params_dict["show_recall_flag"]:
                    plot_2d(x=ms_result["C_list"], y=ms_result["prec_by_C"], ylabel="precision", title="Precision by C and gamma")
                if params_dict["show_f1score_flag"]:
                    plot_2d(x=ms_result["C_list"], y=ms_result["f1_by_C"], ylabel="accuracy", title="f1 score by C")
                if params_dict["show_trerr_flag"]:
                    plot_2d(x=ms_result["C_list"], y=ms_result["tr_err_by_C"], ylabel="training error", title="Training error score by C")
                if params_dict["show_cverr_flag"]:
                    plot_2d(x=ms_result["C_list"], y=ms_result["cv_err_by_C"], ylabel="cross-validation error", title="Cross-validation error score by C")
            else:
                raise Exception("Unsupported kernel type!")
            
            plt.show()
        
        if not params_dict["overnight_simulation"]:
            #entering the C and gamma chosen
            print "Plotted graphics for model selection. Choose the best C and gamma ..."
                        
            while True:
                C_str = raw_input("Enter the C value suggested by model selection:")
                try:
                    C = float(C_str)
                except Exception as e:
                    print "Invalid C inserted. C has to be numeric. Exception: {0}".format(e)
                    continue
                break
            
            if params_dict["kernel"] == SVM_RBF:
                while True:
                    gamma_str = raw_input("Enter the gamma value suggested by model selection:")
                    try:
                        gamma = float(gamma_str)
                    except Exception as e:
                        print "Invalid gamma inserted. gamma has to be numeric. Exception: {0}".format(e)
                        continue
                    break
            
            if params_dict["kernel"] == SVM_linear or params_dict["kernel"] == SVM_RBF_Chi2_squared:
                print "Parameters selection performed! C = {0}.".format(C)
            else:
                print "Parameters selection performed! C = {0}, gamma = {1}".format(C, gamma)    
    
        else:
            
            if params_dict["kernel"] == SVM_linear  or params_dict["kernel"] == SVM_RBF_Chi2_squared:
                C,accuracy = classification_obj.best_accuracy_C(ms_result)
            elif params_dict["kernel"] == SVM_RBF:
                C,gamma,accuracy = classification_obj.best_accuracy_C_and_gamma(ms_result)
            else:
                raise Exception("Unsupported kernel type!")
                
            print "C automatically selected equals to {0}.".format(C)
            if params_dict["kernel"] == SVM_RBF:
                print "gamma automatically selected equals to {0}.".format(gamma)
            print "The accuracy attained by those parameters during model selection is {0}.".format(accuracy)
    
    else:
    
        if params_dict.has_key("C"):
            C = params_dict["C"]
            print "C specified by the user: {0}.".format(C)
        if params_dict.has_key("gamma"):
            gamma = params_dict["gamma"]
            print "gamma specified by the user: {0}".format(gamma)
    
    if params_dict["rfe_features_selection_flag"]:
        print "Performing recursive features elimination..."
        
        if params_dict["kernel"] == SVM_linear or params_dict["kernel"] == SVM_RBF_Chi2_squared:
            rfe = RecursiveFeaturesElimination(C=C,kernel=SVM_linear,
                                               n_iterations=params_dict["n_iterations_rfe"],
                                               test_size=0.3)
        elif params_dict["kernel"] == SVM_RBF:
            rfe = RecursiveFeaturesElimination(C=C,gamma=gamma,kernel=params_dict["kernel"],
                                               n_iterations=params_dict["n_iterations_rfe"],
                                               test_size=0.3)
        else:
                raise Exception("Unsupported kernel type!")
            
        tr_err_rfe, cv_err_rfe, accuracy_rfe,recall_rfe, precision_rfe, f1_score_rfe = rfe.rfe_curves(train_data, target) 
    
        if not params_dict["overnight_simulation"]:
            if params_dict["show_accuracy_flag"]:
                plot_rfe_curve(accuracy_rfe,"accuracy")
            if params_dict["show_precision_flag"]:
                plot_rfe_curve(precision_rfe,"precision")
            if params_dict["show_recall_flag"]:
                plot_rfe_curve(recall_rfe,"recall")
            if params_dict["show_f1score_flag"]:
                plot_rfe_curve(f1_score_rfe,"f1 score")
            if params_dict["show_trerr_flag"]:
                plot_rfe_curve(tr_err_rfe,"training error")
            if params_dict["show_cverr_flag"]:
                plot_rfe_curve(cv_err_rfe,"cross-validation error")
            plt.show()
        
        train_data, rfe_mask = rfe.select_features(train_data, accuracy_rfe)
        if params_dict["test_set_flag"]:
            test_data = rfe.apply_features_selection(test_data)
            
        n_feat_rfe = train_data.shape[1]
        print "Number of features after Recursive Features Elimination: {0}.".format(n_feat_rfe)
    
        print "Recursive features elimination done!."
       
    #training
    print "Performing training..."
    
    if params_dict["kernel"] == SVM_linear or params_dict["kernel"] == SVM_RBF_Chi2_squared:
        model = classification_obj.training(train_data, target, kernel = SVM_linear, C=C)
    elif params_dict["kernel"] == SVM_RBF:
        model = classification_obj.training(train_data, target, kernel = params_dict["kernel"], C=C, gamma=gamma)
    else:
        raise Exception("Unsupported kernel type!")
    
    print "Training performed!"
    
    if params_dict["test_set_flag"]:
        
        #prediction on kaggle test set
        print "Performing classification on the test set..."
        
        predicted = classification_obj.classify(test_data)
        
        print "Classification performed on the test set!"
        
        #save data in the submission format
        save_csv_submitted_labels(predicted, os.path.join(testdir,params_dict["predicted_set_fn"]))
        
    if params_dict["kernel"] == SVM_linear or params_dict["kernel"] == SVM_RBF_Chi2_squared:
        acc, prec, rec, f1 = classification_obj.performance_estimation(train_data, target, kernel = params_dict["kernel"], C = C, n_iterations = params_dict["n_iterations_performance_estimation"])
    elif params_dict["kernel"] == SVM_RBF: 
        acc, prec, rec, f1 = classification_obj.performance_estimation(train_data, target, kernel = params_dict["kernel"], C = C, gamma = gamma, n_iterations = params_dict["n_iterations_performance_estimation"])
    print "Estimated performances:\nAccuracy: {0}\nPrecision: {1}\nRecall: {2}\nf1 Score: {3}".format(acc, prec, rec, f1)

    seedid = random.randint(0,100)
    today = date.today()
    if today.day < 10:
        day = "0%s" % today.day
    else:
        day = "%s" % today.day
    if today.month < 10:
        month = "0%s" % today.month
    else:
        month = "%s" % today.month
    bn = "{name}_{year}_{month}_{day}_rand{seed}_acc{acc:.4f}_prec{prec:4f}_rec{rec:4f}".format(name=params_dict["dataset_name"],seed=seedid,year=today.year, month=month, day=day, acc=acc, prec=prec, rec=rec)
    bn = bn.replace(".","")

    """
        FILLING MODEL DICT
        Making the predicted model persistent!
    """

    model_dict = dict()

    dumped_model = pickle.dumps(model)
    model_dict["classifier"] = dumped_model

    model_dict["scaling_flag"] = params_dict["scaling_flag"]
    if model_dict["scaling_flag"]:
        dumped_scaler = pickle.dumps(scaler)
        model_dict["scaler"] = dumped_scaler

    model_dict["pca_flag"] = params_dict["pca_flag"]
    if params_dict["pca_flag"]:
        dumped_pca = pickle.dumps(pca)
        model_dict["pca"] = dumped_pca   

    model_dict["fsrf_flag"] = params_dict["rf_features_selection_flag"]
    if params_dict["rf_features_selection_flag"]:
        dumped_fsrf_mask = pickle.dumps(fsrf_mask)
        model_dict["fsrf_mask"] = dumped_fsrf_mask
    else:
        model_dict["fsrf_mask"] = None

    model_dict["rfe_flag"] = params_dict["rfe_features_selection_flag"]
    if params_dict["rfe_features_selection_flag"]:
        dumped_rfe_mask = pickle.dumps(rfe_mask)
        model_dict["rfe_mask"] = dumped_rfe_mask
    else:
        model_dict["rfe_mask"] = None

    json_model = json.dumps(model_dict, sort_keys=True, indent=4, separators=(',', ': '))
    models_path = os.path.join(testdir,"models")
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    
    fn = "model_%s.json" % bn

    f = open(os.path.join(models_path, fn),"w")
    f.write(json_model)
    f.close()
    
    """
        FILLING EXPERIMENT DICT
        Saving a summary of the experiment, useful for the data scientist.
    """

    experiment_dict = dict()

    experiment_dict["01) number of samples dataset"] = num_samples
    experiment_dict["02) number of features dataset"] = n_feat
    balances_dict = dict()
    for i in xrange(len(balances)):
        balances_dict["{0}".format(i)] = balances[i]
    experiment_dict["01b) Balance of the classes of the dataset"] = balances_dict     

    if params_dict["kernel"] == SVM_linear:
        experiment_dict["03) classifier type"] = "SVM linear"
    elif params_dict["kernel"] == SVM_RBF:
        experiment_dict["03) classifier type"] = "SVM RBF"
    elif params_dict["kernel"] == SVM_RBF_Chi2_squared:
        experiment_dict["03) classifier type"] = "SVM RBF Chi2"
    else:
        experiment_dict["03) classifier type"] = "Not specified"
    
    if not params_dict["skip_model_selection"]:
        experiment_dict["04) C list"] = C_list
    if params_dict["kernel"] == SVM_RBF and not params_dict["skip_model_selection"]:
        experiment_dict["05) gamma list"] = gamma_list

    experiment_dict["06) selected_C"] = C
    if params_dict["kernel"] == SVM_RBF:
        experiment_dict["07) selected gamma"] = gamma
    
    experiment_dict["08) accuracy"] = acc
    experiment_dict["09) precision"] = prec
    experiment_dict["10) recall"] = rec
    experiment_dict["11) f1 score"] = f1  

    experiment_dict["12) number iterations in model selection"] = params_dict["n_iterations_ms"]
    
    experiment_dict["13) pca flag"] = params_dict["pca_flag"]
    if params_dict["pca_flag"]:
        experiment_dict["14) pca retain"] = params_dict["pca_variance_retain"]
        experiment_dict["16) number of features after pca"] = n_feat_pca

    experiment_dict["17) features scaling"] = params_dict["scaling_flag"]

    experiment_dict["18) random forests features selection"] = params_dict["rf_features_selection_flag"]
    if params_dict["rf_features_selection_flag"]:
        experiment_dict["19) number of features after random forests features selection"] = n_feat_fsrf

    experiment_dict["20) recursive features elimination"] = params_dict["rfe_features_selection_flag"]
    if params_dict["rfe_features_selection_flag"]:
        experiment_dict["21) number of iterations in rfe"] = params_dict["n_iterations_rfe"]

    json_experiment = json.dumps(experiment_dict, sort_keys=True, indent=4, separators=(',', ': '))
    experiments_path = os.path.join(testdir,"experiments")
    if not os.path.exists(experiments_path):
        os.makedirs(experiments_path)
    fn = "experiment_%s.json" % bn
    f = open(os.path.join(experiments_path, fn),"w")
    f.write(json_experiment)
    f.close()
    
    if not params_dict["skip_model_selection"]:
        if params_dict["kernel"] == SVM_RBF:
            acc_table = print_model_selection_results(results = ms_result["acc_by_C_and_gamma"], 
                                          C_list = ms_result["C_list"], 
                                          gamma_list = ms_result["gamma_list"] ) 
            prec_table = print_model_selection_results(results = ms_result["prec_by_C_and_gamma"], 
                                          C_list = ms_result["C_list"], 
                                          gamma_list = ms_result["gamma_list"] ) 
            recall_table = print_model_selection_results(results = ms_result["recall_by_C_and_gamma"], 
                                          C_list = ms_result["C_list"], 
                                          gamma_list = ms_result["gamma_list"] ) 
            f1_table = print_model_selection_results(results = ms_result["f1_by_C_and_gamma"], 
                                          C_list = ms_result["C_list"], 
                                          gamma_list = ms_result["gamma_list"] ) 
        else:
            acc_table = print_model_selection_results(results = ms_result["acc_by_C"], 
                                          C_list = ms_result["C_list"], 
                                          gamma_list = None )
            prec_table = print_model_selection_results(results = ms_result["prec_by_C"], 
                                          C_list = ms_result["C_list"], 
                                          gamma_list = None )
            recall_table = print_model_selection_results(results = ms_result["recall_by_C"], 
                                          C_list = ms_result["C_list"], 
                                          gamma_list = None )
            f1_table = print_model_selection_results(results = ms_result["f1_by_C"], 
                                          C_list = ms_result["C_list"], 
                                          gamma_list = None )
        acc_str = "Accuracy:\n{0}\n".format(acc_table)        
        prec_str = "Precision:\n{0}\n".format(prec_table)
        recall_str = "Recall:\n{0}\n".format(recall_table)
        f1_str = "f1_score:\n{0}\n".format(f1_table)
        
        fn = "experiment_%s_results.txt" % bn
        f = open(os.path.join(experiments_path, fn),"w")
        f.write(acc_str)
        f.write(prec_str)
        f.write(recall_str)
        f.write(f1_str)
        f.close()    

        basename = os.path.basename(bn)

        print "Results saved in %s." % basename

if __name__ == "__main__":
    main()
