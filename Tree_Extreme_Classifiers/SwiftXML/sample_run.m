addpath( genpath( '../Tools' ) );

dataset = 'EUR-Lex';
data_dir = fullfile( '..', 'Sandbox', 'Data', dataset );
results_dir = fullfile( '..', 'Sandbox', 'Results', dataset );
model_dir = fullfile( '..', 'Sandbox', 'Results', dataset, 'model' );
create_recur_dir( model_dir );

trn_file = fullfile( data_dir, 'eurlex_train.txt' );  % training points' user features and ground truth labels
tst_file = fullfile( data_dir, 'eurlex_test.txt' );	% test points' user features and ground truth labels
Y_Yf_file = fullfile( data_dir, 'Y_Yf.txt' );	% label or item features for all the labels

trn_X_Xf_file = fullfile( results_dir, 'trn_X_Xf.txt' );	% training points' user features
trn_item_X_Xf_file = fullfile( results_dir, 'trn_item_X_Xf.txt' );	% training points' item-set features
trn_X_Y_file = fullfile( results_dir, 'trn_X_Y.txt' );	% training points' ground truth labels
tst_X_Xf_file = fullfile( results_dir, 'tst_X_Xf.txt' );	% test points' user features
tst_item_X_Xf_file = fullfile( results_dir, 'tst_item_X_Xf.txt' );	% test points' item-set features
inc_tst_X_Y_file = fullfile( results_dir, 'inc_tst_X_Y.txt' );	% test points' revealed labels
exc_tst_X_Y_file = fullfile( results_dir, 'exc_tst_X_Y.txt' );	% test points' unrevealed, held-out labels for evaluation
inv_prop_file = fullfile( results_dir, 'inv_prop.txt' );	% labels' inverse propensity scores
score_file = fullfile( results_dir, 'score_mat.txt' );	% predicted label scores for the test points

frac=0.8;	% 'frac' percentage of ground truth labels will be revealed per each test point, while rest of the points are held-out for evaluation purposes
A=0.55;	% 'A' parameter in the inverse propensity model. Refer to README.txt
B=1.5;	% 'B' parameter in the inverse propensity model. Refer to README.txt
N=15539; % Number of points in 'trn_file'

swiftXML_preprocess_data( trn_file, tst_file, Y_Yf_file, trn_X_Xf_file, trn_item_X_Xf_file, trn_X_Y_file, tst_X_Xf_file, tst_item_X_Xf_file, inc_tst_X_Y_file, exc_tst_X_Y_file, inv_prop_file, frac, A, B );
trn_X_Xf = read_text_mat( trn_X_Xf_file );
trn_item_X_Xf = read_text_mat( trn_item_X_Xf_file );
trn_X_Y = read_text_mat( trn_X_Y_file );
inv_prop = csvread( inv_prop_file );
tst_X_Xf = read_text_mat( tst_X_Xf_file );
tst_item_X_Xf = read_text_mat( tst_item_X_Xf_file );
inc_tst_X_Y = read_text_mat( inc_tst_X_Y_file );
exc_tst_X_Y = read_text_mat( exc_tst_X_Y_file );

% training
% Reads training features in user space (in trn_X_Xf) and item space (in trn_item_X_Xf), training labels (in trn_X_Y), label propensity scores (in inv_prop) and writes SwiftXML model to 'model_dir'. Fine-tuned hyperparameter settings for the benchmark datasets used in the SwiftXML (WSDM'18) paper are available from "hyperparameters.txt" file.
param = [];
param.pfswitch = 0;
param.num_thread = 5;
param.start_tree = 0;
param.num_tree = 50;
param.bias = 1.0;
param.log_loss_coeff = 0.5;
param.item_log_loss_coeff = 32.0;
param.max_leaf = 10;
param.alpha = 0.9;
param.feat_imp = 0.5;
param.num_trn_X = N;

swiftXML_train( trn_X_Xf, trn_item_X_Xf, trn_X_Y, inv_prop, model_dir, param );

% testing
% Reads test features in user space (in tst_X_Xf) and item space (in tst_item_X_Xf), SwiftXML model (in model_dir), and writes test label scores to score_mat
score_mat = swiftXML_predict( tst_X_Xf, tst_item_X_Xf, model_dir, param );

% performance evaluation 
swiftXML_evaluate_predictions( score_mat, inc_tst_X_Y, exc_tst_X_Y, inv_prop );

