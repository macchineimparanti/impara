'''
Created on Dec 8, 2013

@author: Alessandro Ferrari

This script performs classification using scikit-learn, numpy and matplotlib on the London Data Scientists Competition on Kaggle.
Classification is perfomed by using an SVM with radial basis functions as kernel. 
The method allows to perform features scaling, pca, model selection and recursive features elimination.
All these settings can be managed as options of the script. Type "python london_data_scientists_competition.py -h" for more informations.

Examples of usage:

Example 1, first attempt (default C_list and gamma_list values), it can take a while:
python main.py --pca-enabled --features-scaling-disabled . .

Example 2, refining model selection (improving results):
python main.py --pca-enabled --features-scaling-disabled --C-list=[10,30,50,70,90] --gamma-list=[0.01,0.02,0.03,0.04,0.05] . .

Example 3, skipping model selection (straightaway to the results):
python london_data_scientists_competition.py --pca-enabled --features-scaling-disabled --skip-model-selection --C=50 --gamma=0.023 . .
'''

import os
import sys
import getopt
import classification
import numpy as np
from classification import SVM, dataset_scaling
import matplotlib.pyplot as plt
from utilities import plot_3d, save_csv_submitted_labels, plot_features, plot_rfe_curve
from features_preprocessing import Scaler
from dimensionality_reduction import PCA


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
                                                          "features-scaling-enabled",
                                                          "features-scaling-disabled",
                                                          "C-list=","gamma-list=",
                                                          "skip-model-selection",
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
                                                          "show-cverr-disabled"])
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
        
            try:
                params_dict["gamma_list"] = parse_list_string( a )
            except Exception as e:
                print "--gamma-list argument expressed wrong. {0}".format(e)
                sys.exit(2)
        
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
    
        elif o == "-h" or o == "--help":
            
            description = "This script performs classification using scikit-learn, numpy and matplotlib on the London Data Scientists Competition on Kaggle."+\
                            " Classification is perfomed by using an SVM with radial basis functions as kernel. The method allows to perform features scaling,"+\
                            " pca, model selection and recursive features elimination. All these settings can be managed as options of the script.\n"

            print description
            
            cmd_synopsys = "python london_data_scientists_competition.py [--C=numeric] [--gamma=numeric] [--pca-enabled|--pca-disabled]"+\
                            " [--features-scaling-enabled|--features-scaling-disabled] [--C-list=[0.01,0.1,1,..]]"+\
                            " [--gamma-list=[0.01,0.1,1,..]] [--skip-model-selection]"+\
                            " [--show-precision-enabled|--show-precision-disabled] [--show-accuracy-enabled|--show-accuracy-disabled]"+\
                            " [--show-recall-enabled|--show-recall-disabled] [--show-f1score-enabled|--show-f1score-disabled]"+\
                            " [--show-trerr-enabled|--show-trerr-disabled] [--show-cverr-enabled|--show-cverr-disabled]"+\
                            " source_path destination_path\n"
            print cmd_synopsys
            print "--C : specifies the regularization parameter of the SVM."
            print "It has to be specified only if --skip-model-selection, otherwise C should be indicate after the model selection process.\n"
            print "--gamma : specifies the gamma of the rbf kernel of the SVM."
            print "It has to be specified only if --skip-model-selection, otherwise gamma should be indicate after the model selection process.\n"
            print "--pca-enabled : it enables linear principal component analysis on the features of the dataset." 
            print "It is mutually exclusive to --pca-disabled.\n"
            print "--pca-disabled: it disables linear principal component analysis on the features of the dataset." 
            print "It is mutually exclusive to --pca-disabled.\n"
            print "--features-scaling-enabled : it enables features whitening. It is mutually exclusive to --features-scaling-disabled.\n"
            print "--features-scaling-disabled : it disables features whitening. It is mutually exclusive to --features-scaling-enabled.\n"
            print "--C-list : specifies the list of C values used in model selection in the format [0.001,0.01,0.1,1,10]. --skip-model-selection must not be activated.\n"
            print "--gamma-list : specifies the list of gamma values used in model selection in the format [0.001,0.01,0.1,1,10]. --skip-model-selection must not be activated.\n"
            print "--skip-model-selection : this flag allows to skip model selection. In this case, C and gamma parameters can be expressed as command line parameters. --C-list and --gamma-list must not be specified.\n"
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
            
            sys.exit(0)
            
        else:
            print "Unrecognized option: {0}. For details: python london_data_scientists_competition.py -h ".format(0)
            sys.exit(2)
    
    if len(sys.argv) < 3:
        print "Error while specifying the arguments. python london_data_scientists_competition.py <sourcedir> <destdir>"
    
    params_dict["source_path"] = os.path.expanduser(args[-2])
    params_dict["source_path"] = params_dict["source_path"].replace(".",os.getcwd())
    params_dict["dest_path"] = os.path.expanduser(args[-1])
    params_dict["dest_path"] = params_dict["dest_path"].replace(".",os.getcwd())
    
    if not os.path.exists(params_dict["source_path"]):
        raise Exception("The specified source path does not exist! Source path: {0}.".format(params_dict["source_path"]))
    
    if not os.path.exists(params_dict["dest_path"]):
        raise Exception("The specified destination path does not exist! Destination path: {0}.").format(params_dict["dest_path"])
        
    if not params_dict.has_key("scaling_flag"):
        params_dict["scaling_flag"] = True
    
    if not params_dict.has_key("pca_flag"):
        params_dict["pca_flag"] = True
        
    if not params_dict.has_key("skip_model_selection"):
        params_dict["skip_model_selection"] = False
        
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

    return params_dict


def main():

    params_dict = parse_option()

    testdir = params_dict["source_path"]
    
    # Read data
    train = np.genfromtxt(open(os.path.join(testdir, "LondonDataScientistsCompetition_data",'train.csv'),'rb'), delimiter=',')
    target = np.genfromtxt(open(os.path.join(testdir, "LondonDataScientistsCompetition_data",'trainLabels.csv'),'rb'), delimiter=',')
    test = np.genfromtxt(open(os.path.join(testdir, "LondonDataScientistsCompetition_data",'test.csv'),'rb'), delimiter=',')
    
    print "Visualizing features for understanding the most suitable scaling type."
    
    plot_features(np.vstack((train,test)))
    plt.show()
    
    #features scaling
    print "Starting features preprocessing ..."
    
    if params_dict["scaling_flag"]:
        b_v_flag = True
    else:
        b_v_flag = False
    
    scaler = Scaler(bias_and_variance_flag = b_v_flag, log10_flag = False, log2_flag = False, log1p_flag = False)
    dataset_scaled = scaler.fit(np.vstack((train,test)))
    train_scaled = dataset_scaled[:1000]
    test_scaled = dataset_scaled[1000:]
    
    if params_dict["scaling_flag"]:
        
        print "Visualizing features after features preprocessing.."
    
        plot_features(dataset_scaled)
        plt.show()
        
    if params_dict["pca_flag"]:
        
        print "Performing PCA..."
        
        pca = PCA(variance_retain = 1.0)
        pca.fit(dataset_scaled)
        dataset_pca = pca.transform(dataset_scaled)
        train_pca = dataset_pca[:1000]
        test_pca = dataset_pca[1000:]
    
    else:
    
        dataset_pca = dataset_scaled
        train_pca = train_scaled
        test_pca = test_scaled
    
    if params_dict["pca_flag"]:
        
        print "Visualizing features after PCA..."
        
        plot_features(dataset_pca)
        plt.show()
    
    print "Features preprocessing done!"
    
    classification_obj=SVM()
    
    if not params_dict["skip_model_selection"]:
    
        print "Starting model selection ..."
        
        if not params_dict.has_key("C_list"):
            C_list = [0.001,0.01,0.1,1,10,100,1000,10000]
        else:
            C_list = params_dict["C_list"]
        
        if not params_dict.has_key("gamma_list"):
            gamma_list = [0.001,0.01,0.1,1,10,100,1000,10000]
        else:
            gamma_list = params_dict["gamma_list"]
                    
        #performing model selection
        ms_result = classification_obj.model_selection(train_pca,target,n_iterations=6,
                                                       C_list = C_list,
                                                       gamma_list = gamma_list,
                                                       show_accuracy_flag = params_dict["show_accuracy_flag"], 
                                                       show_precision_flag = params_dict["show_precision_flag"], 
                                                       show_recall_flag = params_dict["show_recall_flag"], 
                                                       show_f1_score_flag = params_dict["show_f1score_flag"])
        
        #displaying model selection
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
        plt.show()
        
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
        
        while True:
            gamma_str = raw_input("Enter the gamma value suggested by model selection:")
            try:
                gamma = float(gamma_str)
            except Exception as e:
                print "Invalid gamma inserted. gamma has to be numeric. Exception: {0}".format(e)
                continue
            break
        
        print "Parameters selection performed! C = {0}, gamma = {1}".format(C, gamma)    

    if params_dict.has_key("C"):
        C = params_dict["C"]
        print "C specified by the user: {0}.".format(C)
    if params_dict.has_key("gamma"):
        gamma = params_dict["gamma"]
        print "gamma specified by the user: {0}".format(gamma)
    
    tr_err_rfe, cv_err_rfe, accuracy_rfe,recall_rfe, precision_rfe, f1_score_rfe = classification_obj.recursive_features_elimination(train_pca,target, C=C, gamma = gamma, n_iterations = 1, test_size = 0.3) 
    
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
    
    print "Recursive features elimination done!."
    
    num_samples, num_features = train_pca.shape
    
    while True:
        K_str = raw_input("Enter the number of components to keep:")
        try:
            K = float(K_str)
        except Exception as e:
            print "Invalid number of components inserted. number of components has to be numeric. Exception: {0}".format(e)
            continue
        if not 0 < K < num_features:
            print "Number of components has to be smaller than {0} and greater than 0.".format(num_features)
            continue
        break
    
    #training
    print "Performing training..."
    
    classifier = classification_obj.training(train_pca, target, C=C, gamma=gamma)
    
    print "Training performed!"
    
    #prediction on kaggle test set
    print "Performing classification on the test set..."
    
    predicted = classification_obj.classify(test_pca)
    
    print "Classification performed on the test set!"
    
    #save data in the submission format
    save_csv_submitted_labels(predicted, os.path.join(testdir,"LondonDataScientistsCompetition_data","predicted_y.csv"))

if __name__ == "__main__":
    main()