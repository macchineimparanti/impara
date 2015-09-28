'''
Created on Dec 8, 2013

@author: Alessandro Ferrari

This script performs classification using scikit-learn, numpy and matplotlib.
Classification is perfomed by using an SVM with radial basis functions as kernel, chi-squared kernels and linear kernel. 
The method allows to perform features scaling, pca, model selection, random forests features selection and recursive features elimination.
All these settings can be managed by specifying them in a configuration file. Type "python __main__.py -h" for more informations.

Examples of usage:

First, add in the python path the path of the library. 
For linux just types in the terminal:
>>PYTHONPATH=$PYTHONPATH:/path/to/folder/impara/source
>>export PYTHONPATH

Move the terminal current working directory to the library root folder:
>>cd /path/to/folder/impara/source

By typing:
>>python __main__.py -h
you get help about the command line parameters inputs.

Example of usage:
python __main__.py --configuration_filename /path/to/script_params.json

Example of script_params.json (you cannot keep comments in the deployed one):

{
    "C":  null ,  #this has to be set to a float number if model selection 
    #is disabled, while if there is model selection enabled, 
    #it can be left to null
    
    "gamma":  null , #this has to be set to a float number if model selection 
    #is disabled, while if there is model selection enabled, 
    #it can be left to null
    
    "pca_flag": true, #if true, pca is enabled, otherwise is disabled
    
    "pca_variance_retain": 0.99,
    
    "scaling_flag": true, #if true, features scaling is performed
    
    "C_list": null, #list of C parameters to cross-validate. 
    #If not set, it is left to the default one.
    
    "gamma_list": null, #list of gamma parameters to cross-validate. 
    #Only for rbf kernel. If not set, it is left to the default one.
    
    "svm_type": "rbf",  #selection among ('rbf','rbf_chi2','linear')
    
    "skip_model_selection": false, #if true, there is model selection, 
    #"C" and "gamma" are determined by this configuration file
    
    "n_iterations_ms": 6, #number of iterations in model selection for having more reliable cross-validation
    
    "n_iterations_performance_estimation": 20, #number of iterations for having more reliable performances estimation
    
    "sparse_filtering_flag": false, #determines if sparse filtering is enabled
    
    "save_sf_fn": null,
    
    "load_sf_fn": null,
    
    "n_layers_sf": null,
    
    "n_iterations_sf": 1000,
    
    "n_features_sf": 50,
    
    "rf_features_selection_flag": true, #if true, random forests features selection is performed
    
    "rfe_features_selection_flag": false, #if true, recursive features elimination is enabled, useful for cross-validate number of pca components
    
    "n_iterations_rfe": 5, #number iterations recursive features elimination, for having more robust cross-validation
    
    "overnight_simulation": true, #if true, there is not need of user interaction during the training
    
    "show_precision_flag": true, #show accurate precision statistics
    
    "show_accuracy_flag": true, #show accurate accuracy statistics
    
    "show_recall_flag": true, #show accurate recall statistics
    
    "show_f1score_flag": true, #show accurate f1-score statistics
    
    "show_trerr_flag": false, #show accurate training error statistics
    
    "show_cverr_flag": false, #show accurate cross-validation error statistics
    
    "max_num_cpus": 4, #maximum number of processes for executing model selection. -1 to leave it as max as possible
    
    "dataset_name": "generic", #identification name for saving the model, useful for organizing data
    
    "test_set_flag": false,  #this does not exclude cross-validation, in this case test_set are samples without labels, useful in competitions where you have unlabeled data to submit.
    
    "training_set_fn": "/path/to/training/set.csv",
    
    "target_set_fn": "/path/to/labels.csv",
    
    "test_set_fn": "/path/to/test/set.csv", #this has to be specified only if there is test_set_unlabeled data to classify
    
    "predicted_set_fn": "/path/to/predicted/set.csv", #this is done only if there is test_set_unlabeled data to classify
    
    "dest_path": "/destination/path" #directory where to save produced data
}

'''

import os
import sys
import json
import pickle
import getopt
import random
import numpy as np
import classification
from datetime import date
import matplotlib.pyplot as plt
from dimensionality_reduction import PCA
from features_preprocessing import Scaler
try:
    from sparse_filtering import SparseFilter
    sparse_filtering_available = True
except Exception as e:
    sparse_filtering_available = False
    sparse_filtering_import_exc = str(e)
from features_filtering import FeaturesSelectionRandomForests
from classification import SVM, dataset_scaling, RecursiveFeaturesElimination, SVM_RBF, SVM_linear, SVM_RBF_Chi2_squared, classes_balance
from utilities import plot_3d, plot_2d, save_csv_submitted_labels, plot_features, plot_rfe_curve, load_sf_features, save_sf_features, print_model_selection_results


def parse_option():

    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help","configuration_filename="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        sys.exit(2)
    
    options_list = []
    for o, a in opts:
        options_list.append(o)
    
    for o, a in opts:
        
        if o == "--configuration_filename":
            configuration_fn = os.path.abspath(a)
            if not os.path.exists(configuration_fn):
                print "The specified configuration filename %s does not exist."%configuration_fn
                sys.exit(2)
            try:
                f = open(configuration_fn,"r")
                params_str = f.read()
                f.close()
            except Exception as e:
                print "Error while reading configuration file %s from disk. Exception: %s."%(configuration_fn,e)
                sys.exit(2)
            try:
                params_dict = json.loads(params_str)
            except Exception as e:
                print "Error while json-parsing parameters. Exception: %s."%e
                sys.exit(2)
        
        elif o == "-h" or o == "--help":
                        
            help = """\n\n\nThis script performs classification using scikit-learn, numpy and matplotlib.
Classification is perfomed by using an SVM with radial basis functions as kernel, chi-squared kernels and linear kernel. 
The method allows to perform features scaling, pca, model selection, random forests features selection and recursive features elimination.
All these settings can be managed by specifying them in a configuration file. Type "python __main__.py -h" for more informations.

Examples of usage:

First, add in the python path the path of the library. 
For linux just types in the terminal:
>>PYTHONPATH=$PYTHONPATH:/path/to/folder/impara/source
>>export PYTHONPATH

Move the terminal current working directory to the library root folder:
>>cd /path/to/folder/impara/source

By typing:
>>python __main__.py -h
you get help about the command line parameters inputs.

Example of usage:
python __main__.py --configuration_filename /path/to/script_params.json

Example of script_params.json (you cannot keep comments in the deployed one):

{
    "C":  null ,  #this has to be set to a float number if model selection 
    #is disabled, while if there is model selection enabled, 
    #it can be left to null
    
    "gamma":  null , #this has to be set to a float number if model selection 
    #is disabled, while if there is model selection enabled, 
    #it can be left to null
    
    "pca_flag": true, #if true, pca is enabled, otherwise is disabled
    
    "pca_variance_retain": 0.99,
    
    "scaling_flag": true, #if true, features scaling is performed
    
    "C_list": null, #list of C parameters to cross-validate. 
    #If not set, it is left to the default one.
    
    "gamma_list": null, #list of gamma parameters to cross-validate. 
    #Only for rbf kernel. If not set, it is left to the default one.
    
    "svm_type": "rbf",  #selection among ('rbf','rbf_chi2','linear')
    
    "skip_model_selection": false, #if true, there is model selection, 
    #"C" and "gamma" are determined by this configuration file
    
    "n_iterations_ms": 6, #number of iterations in model selection for having more reliable cross-validation
    
    "n_iterations_performance_estimation": 20, #number of iterations for having more reliable performances estimation
    
    "sparse_filtering_flag": false, #determines if sparse filtering is enabled
    
    "save_sf_fn": null,
    
    "load_sf_fn": null,
    
    "n_layers_sf": null,
    
    "n_iterations_sf": 1000,
    
    "n_features_sf": 50,
    
    "rf_features_selection_flag": true, #if true, random forests features selection is performed
    
    "rfe_features_selection_flag": false, #if true, recursive features elimination is enabled, useful for cross-validate number of pca components
    
    "n_iterations_rfe": 5, #number iterations recursive features elimination, for having more robust cross-validation
    
    "overnight_simulation": true, #if true, there is not need of user interaction during the training
    
    "show_precision_flag": true, #show accurate precision statistics
    
    "show_accuracy_flag": true, #show accurate accuracy statistics
    
    "show_recall_flag": true, #show accurate recall statistics
    
    "show_f1score_flag": true, #show accurate f1-score statistics
    
    "show_trerr_flag": false, #show accurate training error statistics
    
    "show_cverr_flag": false, #show accurate cross-validation error statistics
    
    "max_num_cpus": 4, #maximum number of processes for executing model selection. -1 to leave it as max as possible
    
    "dataset_name": "generic", #identification name for saving the model, useful for organizing data
    
    "test_set_flag": false,  #this does not exclude cross-validation, in this case test_set are samples without labels, useful in competitions where you have unlabeled data to submit.
    
    "training_set_fn": "/path/to/training/set.csv",
    
    "target_set_fn": "/path/to/labels.csv",
    
    "test_set_fn": "/path/to/test/set.csv", #this has to be specified only if there is test_set_unlabeled data to classify
    
    "predicted_set_fn": "/path/to/predicted/set.csv", #this is done only if there is test_set_unlabeled data to classify
    
    "dest_path": "/destination/path" #directory where to save produced data
}"""

            print help
            
            sys.exit(0)
            
        else:
            print "Unrecognized option: {0}. For details: python svm_classifier_main.py -h ".format(o)
            sys.exit(2)
    
    if len(sys.argv) < 3:
        print "Error while specifying the arguments. python svm_classifier_main.py <sourcedir> <destdir>"
            
    if not params_dict.has_key("dest_path"):
        print "The destination path is not specified in the configuration file %s."%configuration_fn
        sys.exit(2)
                
    if not os.path.exists(params_dict["dest_path"]):
        os.makedirs(params_dict["dest_path"])
        
    if not params_dict.has_key("training_set_fn"):
        print "Training set not specified in %s."%configuration_fn
        sys.exit(2)
    
    if not params_dict.has_key("test_set_flag"):
        params_dict["test_set_flag"] = False
    
    if params_dict["test_set_flag"]:
        if not params_dict.has_key("test_set_fn"):
            print "The test set filename is not specified in configuration file %s."%configuration_fn
            sys.exit(2)
            
    if not params_dict.has_key("target_set_fn"):
        print "The target set filename is not specified in configuration file %s."%configuration_fn
        sys.exit(2)
        
    if not params_dict.has_key("predicted_set_fn"):
        params_dict["predicted_set_fn"] = os.path.join(params_dict["dest_path"],"predicted_y.csv")
            
    if not params_dict.has_key("scaling_flag"):
        params_dict["scaling_flag"] = True
    
    if not params_dict.has_key("pca_flag"):
        params_dict["pca_flag"] = True
        
    if not params_dict.has_key("pca_variance_retain"):
        params_dict["pca_variance_retain"] = 1.0
        
    if not params_dict.has_key("svm_type"):
        params_dict["svm_type"] = "rbf"
        
    if params_dict["svm_type"]=="linear":
        params_dict["kernel"] = SVM_linear
    elif params_dict["svm_type"]=="rbf":
        params_dict["kernel"] = SVM_RBF
    elif params_dict["svm_type"]=="rbf_chi2":
        params_dict["kernel"] = SVM_RBF_Chi2_squared
    else:
        print "The svm_type %s specified is not supported!"%params_dict["svm_type"]
        sys.exit(2)
        
    if not params_dict.has_key("skip_model_selection"):
        params_dict["skip_model_selection"] = False
 
    if params_dict["skip_model_selection"]:
        if not params_dict.has_key("C"):
            print "Skipping model selection is enabled but C parameter is not specified!"
            sys.exit(2)
        if params_dict["kernel"] == SVM_RBF and not params_dict.has_key("gamma"):
            print "Skipping model selection is enabled but gamma parameter is not specified!"
            sys.exit(2)
 
    if not params_dict.has_key("n_iterations_performance_estimation"):
            params_dict["n_iterations_performance_estimation"] = 20
 
    if not params_dict.has_key("sparse_filtering_flag"):
        params_dict["sparse_filtering_flag"] = False
    
    if params_dict["sparse_filtering_flag"]:
        
        if not sparse_filtering_available:
            print "Sparse filtering enabled. Error while importing the sparse filtering module: %s."%sparse_filtering_import_exc
            sys.exit(2)
        
        if not params_dict.has_key("n_iterations_sf"):
            params_dict["n_iterations_sf"] = 1000 
            
        if not params_dict.has_key("n_features_sf"):
            params_dict["n_features_sf"] = 50
    
        if not params_dict.has_key("n_layers_sf"):
            params_dict["n_layers_sf"] = 1
            
        if not params_dict.has_key("load_sf_fn") or params_dict["load_sf_fn"] is None:
            params_dict["load_sf_flag"] = False
        elif not os.path.exists(params_dict["load_sf_fn"]):
            print "The file %s specified for loading sparse filtering features does not exist! Not loading sparse filtering."%params_dict["load_sf_fn"]
            params_dict["load_sf_flag"] = False
            
        if not params_dict.has_key("save_sf_fn") or params_dict["save_sf_fn"] is None:
            params_dict["save_sf_flag"] = False
        elif not os.path.exists(os.path.dirname(params_dict["save_sf_fn"])):
            os.makedirs(os.path.dirname(params_dict["save_sf_fn"]))
    else:
        params_dict["load_sf_flag"] = False
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
        params_dict["show_precision_flag"] = True

    if not params_dict.has_key("show_recall_flag"):
        params_dict["show_recall_flag"] = True

    if not params_dict.has_key("show_f1score_flag"):
        params_dict["show_f1score_flag"] = True

    if not params_dict.has_key("show_trerr_flag"):
        params_dict["show_trerr_flag"] = False

    if not params_dict.has_key("show_cverr_flag"):
        params_dict["show_cverr_flag"] = False
        
    if not params_dict.has_key("max_num_cpus"):
        params_dict["max_num_cpus"] = -1
        
    if not params_dict.has_key("dataset_name"):
        params_dict["dataset_name"] = "generic"

    return params_dict


def get_model_name(params):
    
    name = params["dataset_name"]
    if params["scaling_flag"]:
        scaling = "_with_features_scaling"
    else:
        scaling = ""
    if params["pca_flag"]:
        pca = ("_with_pca_retain_%s"%(params["pca_variance_retain"])).replace(".","")
    else:
        pca = ""
    svm_type = "_%s"%params["svm_type"]
    if params["rf_features_selection_flag"]:
        rf = "_rf_features_selection"
    else:
        rf = ""
    if params["rfe_features_selection_flag"]:
        rfe = "_rfe_features_selection_flag"
    else:
        rfe = ""
    if params["sparse_filtering_flag"]:
        sf = "_with_sf_nfeatures_%s_nlayers_%s"%(params["n_features_sf"],params["n_layers_sf"])
    else:
        sf = ""
        
    model_name = "%s%s%s%s%s%s%s"%(name,scaling,pca,svm_type,rf,rfe,sf)
    
    return model_name
    

def main():

    params_dict = parse_option()
    
    # Read data
    train = np.genfromtxt(open(params_dict['training_set_fn'],'rb'), delimiter=',')
    print "Number of training samples: {0}.".format(train.shape[0])
    print "Number of features: {0}.".format(train.shape[1])
    target = np.genfromtxt(open(params_dict['target_set_fn'],'rb'), delimiter=',')
    len_train_set = train.shape[0]
    if params_dict["test_set_flag"]:
        test = np.genfromtxt(open(params_dict['test_set_fn'],"rb"), delimiter=',')
        
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
            if params_dict["test_set_flag"]:
                sf.fit(np.r_[train,test])
                train_sf = sf.transform(train)
                test_sf = sf.transform(test)
            else:
                sf.fit(train)
                train_sf = sf.transform(train)        

        if params_dict["save_sf_flag"]:
            if params_dict["test_set_flag"]:
                save_sf_features(sf, train_sf, test_sf, params_dict["save_sf_path"])
            else:
                save_sf_features(sf, train_sf, None, params_dict["save_sf_path"])
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
                                                       show_f1_score_flag = params_dict["show_f1score_flag"],
                                                       max_num_cpus = params_dict["max_num_cpus"])
        
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
        save_csv_submitted_labels(predicted, os.path.join(params_dict["dest_path"],params_dict["predicted_set_fn"]))
        
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
    bn = "{name}_{year}_{month}_{day}_rand{seed}_acc{acc:.4f}_prec{prec:4f}_rec{rec:4f}".format(name=get_model_name(params_dict),seed=seedid,year=today.year, month=month, day=day, acc=acc, prec=prec, rec=rec)
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
    models_path = os.path.join(params_dict["dest_path"],"models")
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
    experiments_path = os.path.join(params_dict["dest_path"],"experiments")
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
