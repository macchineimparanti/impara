'''
Created on Mar 8, 2014

@author: alessandro
'''
import os
import sys
import getopt
from sparse_filtering import SparseFilter
import numpy as np
from utilities import load_sf_features, save_sf_features


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
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help", "training-set-fn=",
                                                       "target-set-fn=", "test-set-fn=",
                                                       "n-layers=","n-features-list=",
                                                       "n-iterations=","output-name=",
                                                       "n-repetitions="])
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
            
        if o == "--training-set-fn":
            
            params_dict["training_set_fn"] = a
            
        elif o == "--target-set-fn":
            
            params_dict["target_set_fn"] = a
            
        elif o == "--test-set-fn":
            
            params_dict["test_set_fn"] = a

        elif o == "--n-layers":
                        
            try:
                params_dict["n_layers"] = int( a )
            except Exception as e:
                print "Error while converting to integer --n-layers {0} argument. {1}".format(a,e)
                sys.exit(2)
            
        elif o == "--n-features-list":
                        
            try:
                params_dict["n_features_list"] = parse_list_string( a )
            except Exception as e:
                print "--n-features-list argument expressed wrong. {0}".format(e)
                sys.exit(2)
                
        elif o == "--n-iterations":
                        
            try:
                params_dict["n_iterations"] = int( a )
            except Exception as e:
                print "Error while converting to integer --n-iterations {0} argument. {1}".format(a)
                sys.exit(2)
                
        elif o == "--output-name":
            
            params_dict["output_name"] = a
            
            #remove extension if exists
            idx = params_dict["output_name"].find('.')
            if idx!=-1:
                params_dict["output_name"] = params_dict["output_name"][:idx]
                print "Warnings: Extension removed from --output-name. New --output-name = {0}.".format(params_dict["output_name"])
            
        elif o == "--n-repetitions":
                            
            try:
                params_dict["n_repetitions"] = int(a)
            except Exception as e:
                print "Error while converting --n-repetitions={0} to int. Exception: {1}".format(a,e)
                sys.exit(2)
            
            if not 0<params_dict["n_repetitions"]<=20:
                print "Error: --n-repetitions has to be greater than 0 and smaller or equal than 20."
                sys.exit(2)
            
        elif o == "-h" or o == "--help":
            
            description = "This script generates sparse filters stacked in layerwise greedy networks using scikit-learn, numpy and matplotlib on the dataset of the London Data Scientists Competition on Kaggle."+\
                            "This script allows to select a list of number of features for the network layers, number of layers, number of iterations to stop optimization and number of repetitions for avoiding bad initializations in the convergence process."
            print description
            
            cmd_synopsys = "python sparse_filtering_generator.py [--training-set-fn training_set_basename] "+\
                            "[--target-set-fn target_set_basename] [--test-set-fn test_set_basename] [--n-layers 2] "+\
                                "[--n-features-list [50,60,70,80,90,100,110,120]] [--n-iterations 10000] [--n-repetitions 1] "+\
                                "[--output-name output_basename] source_path destination_path\n"
            print cmd_synopsys
            print "--training-set-fn: It allows to specify custom names for the training set. By default is 'train.csv'."
            print "--test-set-fn: It allows to specify custom names for the test set. By default is 'test.csv'"
            print "--target-set-fn: It allows to specify custom names for the target set containing labels. By default is set to 'trainLabels.csv'"
            print "--n-layers: It allows to specify the number of layers of sparse filters stacked as a greedy layerwise network. By default is set to 2. If the number of layers specified is n, the sparse filters generated will have number of layers 1,2,3,...,n-2,n-1,n."
            print "--n-features-list: It allows to specify a list containing the number of features. For each number of features a group of sparse filters network is generated. By default is set to [50,60,70,80,90,100,110,120]"
            print "--n-iterations: It allows to specify the number of iterations during the optimization process for calculating a layer. By default is set to 10000."
            print "--n-repetitions: It allows to specify how many times to repeat the design of a sparse filters network with the same characteristics. Repetitions may reduce the risk of bad initializations in the optimization process. By default is set to 1."
            print "--output-name: It allows to specify the output basename for the generated sparse filters networks and their relative transformed data."
            sys.exit(0)
            
        else:
            print "Unrecognized option: {0}. For details: python sparse_filtering_generator.py -h ".format(o)
            sys.exit(2)
    
    if len(sys.argv) < 3:
        print "Error while specifying the arguments. python sparse_filtering_generator.py <sourcedir> <destdir>"
    
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
        
    if not params_dict.has_key("n_layers"):
        params_dict["n_layers"] = 2
        
    if not params_dict.has_key("n_features_list"):
        params_dict["n_features_list"] = [50,60,70,80,90,100,110,120]
    
    if not params_dict.has_key("n_repetitions"):
        params_dict["n_repetitions"] = 1
        
    if not params_dict.has_key("n_iterations"):
        params_dict["n_iterations"] = 10000
        
    if not params_dict.has_key("output_name"):
        params_dict["output_name"] = "sparse_filters_network"
        
    
    return params_dict


def main():

    params_dict = parse_option()

    testdir = params_dict["source_path"]
    
    # Read data
    train = np.genfromtxt(open(os.path.join(testdir,params_dict['training_set_fn']),'rb'), delimiter=',')
    target = np.genfromtxt(open(os.path.join(testdir,params_dict['target_set_fn']),'rb'), delimiter=',')
    test = np.genfromtxt(open(os.path.join(testdir, params_dict['test_set_fn']),'rb'), delimiter=',')
    
    print "Started to perform sparse filtering..."
        
    for n_features in params_dict["n_features_list"]:
        for i in xrange(params_dict["n_repetitions"]):
            print "Number of layers: {0}. Number of features: {1}. Number of iterations: {2}. Number of repetitions: {3}.".format(params_dict["n_layers"], n_features, params_dict["n_iterations"], i+1)
            sf = SparseFilter(n_layers=params_dict["n_layers"],n_features=n_features, n_iterations=params_dict["n_iterations"])
            sf.fit(np.r_[train,test])
            
            sparse_filters_list = sf.layers_ripartition()
            
            for sub_sf in sparse_filters_list: 
                
                train_sf = sub_sf.transform(train)
                test_sf = sub_sf.transform(test)
                output_fn = params_dict["output_name"]
                #remove extension if exists
                idx = output_fn.find('.')
                if idx!=-1:
                    output_fn = output_fn[:idx]
                filename = output_fn + "_nlayers{0}_nfeat{1}_niter{2}_nrep{3}.dat".format(sub_sf.get_layers_counter(), sub_sf.get_features_counter(), params_dict["n_iterations"], i+1)
                path = os.path.join(params_dict["dest_path"],filename)
                save_sf_features(sub_sf, train_sf, test_sf, path)
        
    print "Features sparse filtering performed!"
    
main()