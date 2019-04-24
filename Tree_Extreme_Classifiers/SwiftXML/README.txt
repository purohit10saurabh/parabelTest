** Please make sure that you read and agree to the terms of license (License.pdf) and copyright (liblinear_COPYRIGHT) before using this software. **

This is the code for the algorithm proposed in our research paper "Extreme Multi-label Learning with Label Features for Warm-start Tagging, Ranking & Recommendation" authored by Yashoteja Prabhu, Anil Kag, Shilpa Gopinath, Kunal Dahiya, Shrutendra Harsola, Rahul Agrawal and Manik Varma and published at The International Conference on Web Search and Data Mining (WSDM), 2018. The code is authored by Yashoteja Prabhu (yashoteja.prabhu@gmail.com).

About SwiftXML
==============
The objective in extreme multi-label learning is to build classifiers that can annotate a data point with the subset of relevant labels from an extremely large label set. Extreme classification has, thus far, only been studied in the context of predicting labels for novel test points. SwiftXML is useful for solving extreme classification problem when predictions need to be made on training points with partially revealed labels. This allows the reformulation of warm-start tagging, ranking and recommendation problems as extreme multi-label learning with each item to be ranked/recommended being mapped onto a separate label. SwiftXML can be significantly more accurate as compared to leading extreme classifiers as well as classical recommendation algorithms on warm-start extreme classification tasks. Please refer to the research paper for more details.

This code is made available as is for non-commercial research purposes. Please make sure that you have read the license agreement in LICENSE.doc/pdf. Please do not install or use SwiftXML unless you agree to the terms of the license.

The code for SwiftXML is written in C++ and should compile on 64 bit Windows/Linux machines using a C++11 enabled compiler. Matlab wrappers have also been provided with the code. Installation and usage instructions are provided below. The default parameters provided in the Usage Section work reasonably on the benchmark datasets in the Extreme Classification Repository (http://manikvarma.org/downloads/XC/XMLRepository.html). 

Please contact Yashoteja Prabhu (yashoteja.prabhu@gmail.com) and Manik Varma (manik@microsoft.com) if you have any questions or feedback.

Experimental Results and Datasets
=================================
Please visit the Extreme Classification Repository (http://manikvarma.org/downloads/XC/XMLRepository.html) to download the benchmark datasets and compare SwiftXML's performance to baseline algorithms.
Please download the label (item) features for the benchmark datasets on the Repository as well as user features and labels for new benchmark datasets from https://www.dropbox.com/s/jp208zhmq5trmxh/Item_Features.zip?dl=0. For more information about item features, please refer to the research paper.

Usage
=====
Linux/Windows makefiles for compiling SwiftXML have been provided with the source code. To compile, run "make" (Linux) or "nmake -f Makefile.win" (Windows) in the topmost folder. Run the following commands from inside SwiftXML folder for training and testing. To use Matlab scripts, compile the mex files in 'Tools/matlab' folder by running "make".

Data Preprocessing
------------------
Prior to training and prediction, the data needs to be preprocessed into format accepted by SwiftXML. Preprocessing comprises the following steps

* Download the data point (user) features and the ground truth labels from the Extreme Classification Repository
* Download the label (item) features from here and place it in the dataset's folder. For new datasets, user features and labels are also provided with the zip file
* Set the file names, fraction of revealed labels ("frac") and the parameters of the inverse propensity model for the given dataset ("A" and "B") appropriately in sample_run.sh
* Run swiftXML_preprocess_data.m in Matlab as shown in the example in sample_run.sh. This script creates user and item features and labels in the format expected by SwiftXML. Kag, S. Gopinath, K. Dahiya, S. Harsola, R. Agrawal and M. Varma. Extreme Multi-label Learning with Label Features for Warm-start Tagging, Ranking & Recommendation. In Proceedings of the ACM International Conference on Web Search and Data Mining, Los Angeles, United States, February 2018.
Bibtex source | Abstract | Download in pdf format 
* Run swiftXML_preprocess_data.m in Matlab as shown in the example in sample_run.sh. This script creates user and item features and labels in the at expected by SwiftXML

Following files are expected by SwiftXML and are created by swiftXML_preprocess_data.m:
trn_X_Xf.txt, trn_X_Y.txt, trn_item_X_Xf.txt, tst_X_Xf.txt, inc_tst_X_Y.txt, exc_tst_X_Y.txt, tst_item_X_Xf.txt, inv_prop.txt

Preprocessing requires Perl (https://www.perl.org/get.html) and Matlab (https://www.mathworks.com/downloads). Please refer to sample_run.sh/sample_run.bat for better understanding.


Training
--------

C++:
	./swiftXML_train [user feature file name] [item feature file name] [label file name] [inverse propensity file name] [model folder name] -S 0 -T 1 -s 0 -t 50 -b 1.0 -c 1.0 -m 10 -l 100 -g 30 -a 0.8 -q 1 -N [number of original training points (not counting test ponts which are also used during trianing)]


Matlab:
	swiftXML_train([user feature matrix], [item feature matrix], [input label matrix], [inverse propensity score vector], [output model folder name], param)

where:
	-S = param.pfswitch				: PfastXML switch, setting this to 1 omits tail classifiers, thus leading to PfastXML algorithm. default=0
	-T = param.num_thread			: Number of threads to use. default=1
	-s = param.start_tree			: Starting tree index. default=0
	-t = param.num_tree				: Number of trees to be grown. default=50
	-b = param.bias					: Feature bias value, extre feature value to be appended. default=1.0
	-c = param.log_loss_coeff		: log-loss weight co-efficient for separator in user feature space. default=1.0
	-m = param.max_leaf				: Maximum allowed instances in a leaf node. Larger nodes are attempted to be split, and on failure converted to leaves. default=10
	-l = param.lbl_per_leaf			: Number of label-probability pairs to retain in a leaf. default=100
	-g = param.gamma				: gamma parameter appearing in tail label classifiers. default=30
	-a = param.alpha				: Trade-off parameter between PfastXML and tail label classifiers. default=0.8
	-q = param.quiet				: Quiet option (0/1). default=0
	-ic = param.item_log_loss_coeff	: Log-loss weight co-efficient for separator in item feature space. default=1.0
	-f = param.feat_imp				: Relative importance of user and item feature separators during classification. default=0.5
	-N = param.num_trn_X			: Number of original training instances in dataset. Note that test points are also used during training and are not counted here.

 The fine-tuned hyperparameter settings for the benchmark datasets used in the [1] are available from "hyperparameters.txt" file in the SwiftXML's code folder. For C++, the feature and label input files are expected to be in sparse matrix text format (refer to Miscellaneous section). For Matlab, the feature and label matrices are Matlab's sparse matrices. 

Prediction
----------

C++:
	./swiftXML_predict [user feature file name] [item feature file name] [score file name] [model folder name] -S 0 -T 1 -s 0 -t 50 -n 1000 -q 1

	
Matlab:
	output_score_mat = swiftXML_predict( [user feature file name], [item feature file name], [input model folder name], param )

where:
	-S = param.pfswitch				: PfastXML switch, setting this to 1 omits tail classifiers, thus leading to PfastXML algorithm. default=[value saved in trained model]
	-T = param.num_thread			: Number of threads to use. default=[value saved in trained model]
	-s = param.start_tree			: Starting tree index. default=[value saved in trained model]
	-t = param.num_tree				: Number of trees to be grown. default=[value saved in trained model]
	-n = param.actlbl				: Number of predicted scores per test instance. Lower value means quicker prediction. default=1000
	-q = param.quiet				: quiet option (0/1). default=[value saved in trained model]

	For C++, the feature and score files are expected to be in sparse matrix text format (refer to Miscellaneous section). For Matlab, the feature and score matrices are Matlab's sparse matrices.

Performance Evaluation
----------------------

Scripts for performance evaluation are only available in Matlab. To compile these scripts, execute "make" in the Tools folder from the Matlab terminal.
Following command is executed from Tools/metrics folder:
swiftXML_evaluate_predictions( [test score matrix], [revealed test label matrix], [held-out test label matrix], [inverse label propensity vector] );

Miscellaneous
-------------

* Scripts are provided in the 'Tools' folder for sparse matrix inter conversion between Matlab .mat format and text format.
    To read a text matrix into Matlab:

    	[matrix] = read_text_mat([text matrix name]); 

    To write a Matlab matrix into text format:

    	write_text_mat([Matlab sparse matrix], [text matrix name to be written to]);

* To generate inverse label propensity weights, run the following command inside 'Tools/metrics' folder on Matlab terminal:

    	[weights vector] = inv_propensity([training label matrix],A,B); 

    A,B are the parameters of the inverse propensity model. Following values are to be used over the benchmark datasets:

    	Wikipedia-LSHTC: A=0.5,  B=0.4
    	Amazon:          A=0.6,  B=2.6
    	Other:		 A=0.55, B=1.5

Toy Example
===========

The zip file containing the source code also includes the EUR-Lex dataset as a toy example.
To run SwiftXML on the EUR-Lex dataset, execute "bash sample_run.sh" (Linux) or "sample_run" (Windows) in the SwiftXML folder.
Read the comments provided in the above scripts for better understanding.


References
==========
[1] Y. Prabhu, A. Kag, S. Gopinath, K. Dahiya, S. Harsola, R. Agrawal and M. Varma. Extreme Multi-label Learning with Label Features for Warm-start Tagging, Ranking & Recommendation. In Proceedings of the ACM International Conference on Web Search and Data Mining, Los Angeles, United States, February 2018.
Bibtex source | Abstract | Download in pdf format 
