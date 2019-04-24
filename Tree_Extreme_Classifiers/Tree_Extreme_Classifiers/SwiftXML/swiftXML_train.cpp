#include <iostream>
#include <fstream>
#include <string>
#include <thread>

#include "timer.h"

#include "swiftXML.h"
#include "PfastreXML.h"

using namespace std;

void help()
{
	cerr<<"Sample Usage :"<<endl;
	cerr<<"./swiftXML_train [user feature file name] [item feature file name] [label file name] [inverse propensity file name] [model folder name] -S 0 -T 1 -s 0 -t 50 -b 1.0 -c 1.0 -m 10 -l 100 -g 30 -a 0.8 -q 1 -N 1000"<<endl<<endl;

	cerr<<"-S PfastXML switch, setting this to 1 omits tail classifiers, thus leading to PfastXML algorithm. default=0"<<endl;
	cerr<<"-T Number of threads to use. default=1"<<endl;
	cerr<<"-s Starting tree index. default=0"<<endl;
	cerr<<"-t Number of trees to be grown. default=50"<<endl;
	cerr<<"-b Feature bias value, extre feature value to be appended. default=1.0"<<endl;
	cerr<<"-c log-loss weight co-efficient for separator in user feature space. default=1.0"<<endl;
	cerr<<"-m Maximum allowed instances in a leaf node. Larger nodes are attempted to be split, and on failure converted to leaves. default=10"<<endl;
	cerr<<"-l Number of label-probability pairs to retain in a leaf. default=100"<<endl;
	cerr<<"-g gamma parameter appearing in tail label classifiers. default=30"<<endl;
	cerr<<"-a Trade-off parameter between PfastXML and tail label classifiers. default=0.8"<<endl;
	cerr<<"-q Quiet option (0/1). default=0"<<endl;
	cerr<<"-ic Log-loss weight co-efficient for separator in item feature space. default=1.0"<<endl;
	cerr<<"-f Relative importance of user and item feature separators during classification. default=0.5"<<endl;
	cerr<<"-N Number of pure training instances in dataset."<<endl;

	cerr<<"feature and label files are in sparse matrix format"<<endl;
	exit(1);
}

SParam parse_param(_int argc, char* argv[])
{
	SParam param;

	string opt;
	string sval;
	_float val;

	for(_int i=0; i<argc; i+=2)
	{
		opt = string(argv[i]);
		sval = string(argv[i+1]);
		val = stof(sval);
		
		if(opt=="-m")
			param.max_leaf = (_int)val;
		else if(opt=="-l")
			param.lbl_per_leaf = (_int)val;
		else if(opt=="-b")
			param.bias = (_float)val;
		else if(opt=="-c")
			param.log_loss_coeff = (_float)val;
		else if(opt=="-T")
			param.num_thread = (_int)val;
		else if(opt=="-s")
			param.start_tree = (_int)val;
		else if(opt=="-t")
			param.num_tree = (_int)val;
		else if(opt=="-S")
			param.pfswitch = (_bool)val;
		else if(opt=="-g")
			param.gamma = (_float)val;
		else if(opt=="-a")
			param.alpha = (_float)val;
		else if(opt=="-q")
			param.quiet = (_bool)val;
		else if(opt=="-ic")
			param.item_log_loss_coeff = (_float)val;
		else if(opt=="-f")
			param.feat_imp = (_float)val;
		else if(opt=="-N")
			param.num_trn_X = (_int)val;
	}

	return param;
}

int main(int argc, char* argv[])
{
	if(argc < 6)
		help();

	string ft_file = string(argv[1]);
	check_valid_filename(ft_file, true);
	SMatF* trn_X_Xf = new SMatF(ft_file);
	
	string item_ft_file = string(argv[2]);
	check_valid_filename(item_ft_file, true);
	SMatF* trn_item_X_Xf = new SMatF(item_ft_file);

	string lbl_file = string(argv[3]);
	check_valid_filename(lbl_file, true);
	SMatF* trn_X_Y = new SMatF(lbl_file);

	string prop_file = string(argv[4]);
	check_valid_filename(prop_file, true);
	ifstream fin;
	fin.open(prop_file);
	VecF inv_props;
	for(_int i=0; i<trn_X_Y->nr; i++)
	{
		_float f;
		fin>>f;
		inv_props.push_back(f);
	}
	fin.close();

	string model_folder = string(argv[5]);
	check_valid_foldername(model_folder);

	SParam param = parse_param(argc-6,argv+6);

	param.num_Xf = trn_X_Xf->nr;
	param.num_Y = trn_X_Y->nr;
	param.write(model_folder+"/param");

	if( param.quiet )
		loglvl = LOGLVL::QUIET;

	USE_IDCG = false;

	_float train_time = 0;
	Timer timer;

	timer.tic();
	/* Weighting label matrix with inverse propensity weights */
	for(_int i=0; i<trn_X_Y->nc; i++)
		for(_int j=0; j<trn_X_Y->size[i]; j++)
			trn_X_Y->data[i][j].second *= inv_props[trn_X_Y->data[i][j].first];
	train_time += timer.toc();

	/* training PfastXML trees */
	_float tmptime;
	swiftXML_train_trees(trn_X_Xf, trn_item_X_Xf, trn_X_Y, param, model_folder, tmptime);
	train_time += tmptime;

	/* if pfswitch is true, terminate here immediately after PfastXML */
	if(param.pfswitch)
	{
		cout << "training time: " << train_time/3600.0 << " hr" << endl;

		delete trn_X_Xf;
        delete trn_item_X_Xf;
		delete trn_X_Y;
		return 0;
	}

	timer.tic();

	/* normalize feature vectors to unit norm */
	trn_X_Xf->unit_normalize_columns();

	/*--- calculating model parameters saved in w ---*/

	SMatF* tmat = trn_X_Y->transpose();

	for(int i=0; i<tmat->nc; i++)
	{
		_float a = 1.0/(tmat->size[i]);
		for(int j=0; j<tmat->size[i]; j++)
			tmat->data[i][j].second = a;
	}

	SMatF* w = trn_X_Xf->prod(tmat);

	train_time += timer.toc();

	cout << "training time: " << train_time/3600.0 << " hr" << endl;


	w->write(model_folder+"/w");

	/* free allocated resources */
	delete tmat;
	delete w;
	delete trn_X_Xf;
	delete trn_item_X_Xf;
	delete trn_X_Y;
}
