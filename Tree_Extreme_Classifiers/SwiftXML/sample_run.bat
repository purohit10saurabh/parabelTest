@echo off

set dataset=EUR-Lex
set data_dir=..\Sandbox\Data\%dataset%
set results_dir=..\Sandbox\Results\%dataset%
set model_dir=..\Sandbox\Results\%dataset%\model

::training points' user features and ground truth labels
set trn_file=%data_dir%\eurlex_train.txt
:: test points' user features and ground truth labels
set tst_file=%data_dir%\eurlex_test.txt
:: label or item features for all the labels	
set Y_Yf_file=%data_dir%\Y_Yf.txt

:: training points' user features
set trn_X_Xf_file=%results_dir%\trn_X_Xf.txt
:: training points' item-set features
set trn_item_X_Xf_file=%results_dir%\trn_item_X_Xf.txt
:: training points' ground truth labels
set trn_X_Y_file=%results_dir%\trn_X_Y.txt
:: test points' user features
set tst_X_Xf_file=%results_dir%\tst_X_Xf.txt
:: test points' item-set features
set tst_item_X_Xf_file=%results_dir%\tst_item_X_Xf.txt
:: test points' revealed labels	
set inc_tst_X_Y_file=%results_dir%\inc_tst_X_Y.txt
:: test points' unrevealed, held-out labels for evaluation
set exc_tst_X_Y_file=%results_dir%\exc_tst_X_Y.txt
:: labels' inverse propensity scores
set inv_prop_file=%results_dir%\inv_prop.txt
:: predicted label scores for the test points
set score_file=%results_dir%\score_mat.txt

:: 'frac' percentage of ground truth labels will be revealed per each test point, while rest of the points are held-out for evaluation purposes
set frac=0.8
:: 'A' parameter in the inverse propensity model. Refer to README.txt	
set A=0.55
:: 'B' parameter in the inverse propensity model. Refer to README.txt
set B=1.5
:: Number of points in %trn_file%
set N=15539

::matlab -nodesktop -nodisplay -r "addpath(genpath('..\Tools')); swiftXML_preprocess_data( '%trn_file%', '%tst_file%', '%Y_Yf_file%', '%trn_X_Xf_file%', '%trn_item_X_Xf_file%', '%trn_X_Y_file%', '%tst_X_Xf_file%', '%tst_item_X_Xf_file%', '%inc_tst_X_Y_file%', '%exc_tst_X_Y_file%', '%inv_prop_file%', %frac%, %A%, %B% ); exit;"

:: training
:: Reads training features in user space (in %trn_X_Xf_file%) and item space (in %trn_item_X_Xf_file%), training labels (in %trn_X_Y_file%), label propensity scores (in %inv_prop_file%) and writes SwiftXML model to %model_dir%. Fine-tuned hyperparameter settings for the benchmark datasets used in the SwiftXML (WSDM'18) paper are available from "hyperparameters.txt" file.
swiftXML_train %trn_X_Xf_file% %trn_item_X_Xf_file% %trn_X_Y_file% %inv_prop_file% %model_dir% -S 0 -T 5 -s 0 -t 50 -b 1.0 -c 0.5 -ic 32.0 -m 10 -a 0.9 -f 0.5 -N %N%

:: testing
:: Reads test features in user space (in %tst_X_Xf_file%) and item space (in %tst_item_X_Xf_file%), SwiftXML model (in %model_dir%), and writes test label scores to %score_file%
swiftXML_predict %tst_X_Xf_file% %tst_item_X_Xf_file% %score_file% %model_dir%

:: performance evaluation 
matlab -nodesktop -nodisplay -r "addpath(genpath('..\Tools')); wts = csvread('%inv_prop_file%'); inc_tst_X_Y = read_text_mat('%inc_tst_X_Y_file%'); exc_tst_X_Y = read_text_mat('%exc_tst_X_Y_file%'); score_mat = read_text_mat('%score_file%'); swiftXML_evaluate_predictions( score_mat, inc_tst_X_Y, exc_tst_X_Y, wts );"
