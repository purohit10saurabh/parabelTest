#include <iostream>
#include <fstream>
#include <string>

#include "timer.h"

#include "swiftXML.h"
#include "PfastreXML.h"

using namespace std;

void help()
{
	cerr<<"Sample Usage :"<<endl;
	cerr<<"./swiftXML_predict [user feature file name] [item feature file name] [score file name] [model folder name] -S 0 -T 1 -s 0 -t 50 -n 1000 -q 1"<<endl<<endl;

	cerr<<"-S PfastXML switch, setting this to 1 omits tail classifiers, thus leading to PfastXML algorithm. default=[value saved in trained model]"<<endl;
	cerr<<"-T Number of threads to use. default=[value saved in trained model]"<<endl;
	cerr<<"-s Starting tree index. default=[value saved in trained model]"<<endl;
	cerr<<"-t Number of trees to be grown. default=[value saved in trained model]"<<endl;
	cerr<<"-n Number of predicted scores per test instance. Lower value means quicker prediction. default=1000"<<endl;
	cerr<<"-q quiet option (0/1). default=[value saved in trained model]"<<endl;
	cerr<<"feature and score files are in sparse matrix format"<<endl;
	exit(1);
}

SParam parse_param(int argc, char* argv[], string model_folder, _int& actlbl)
{
	SParam param(model_folder+"/param");
	actlbl = 1000;

	string opt;
	string sval;
	_float val;

	for(_int i=0; i<argc; i+=2)
	{
		opt = string(argv[i]);
		sval = string(argv[i+1]);
		val = stof(sval);

		if(opt=="-S")
			param.pfswitch = (_bool)val;
		else if(opt=="-T")
			param.num_thread = (_int)val;
		else if(opt=="-s")
			param.start_tree = (_int)val;
		else if(opt=="-t")
			param.num_tree = (_int)val;
		else if(opt=="-n")
			actlbl = (_int)val;
		else if(opt=="-q")
			param.quiet = (_bool)val;
	}

	return param;
}

int main(int argc, char* argv[])
{
	if(argc < 5)
		help();

	string ft_file = string(argv[1]);
	check_valid_filename(ft_file,true);
	SMatF* tst_X_Xf = new SMatF(ft_file);

	string item_ft_file = string(argv[2]);
	check_valid_filename(item_ft_file,true);
	SMatF* tst_item_X_Xf = new SMatF(item_ft_file);

	string score_file = string(argv[3]);
	check_valid_filename(score_file,false);

	string model_folder = string(argv[4]);
	check_valid_foldername(model_folder);

	_int actlbl;
	SParam param = parse_param(argc-5, argv+5, model_folder, actlbl);

	if( param.quiet )
		loglvl = LOGLVL::QUIET;

	_float model_size;
	_float predict_time;

	/* run PfastXML prediction */
	SMatF* pf_score_mat = swiftXML_predict_trees(tst_X_Xf, tst_item_X_Xf, param, model_folder, predict_time, model_size);

	/* if pfswitch is true, write PfastXML scores as final scores and terminate */
	if(param.pfswitch)
	{
		cout << "prediction time: " << ((predict_time/tst_X_Xf->nc)*1000.0) << " ms/point" << endl;
		cout << "model size: " << (model_size/1e+9) << " GB" << endl;

		pf_score_mat->write(score_file);
		delete pf_score_mat;
		delete tst_X_Xf;
		delete tst_item_X_Xf;
		return 0;
	}

	string w_file = model_folder + "/w";
	check_valid_filename(w_file,true);
	SMatF* w_mat = new SMatF(w_file);
	
	model_size += w_mat->get_ram();

	/* normalize feature vectors */
	Timer timer;
	timer.tic();
	tst_X_Xf->unit_normalize_columns();
	VecF w_sq = w_mat->column_norms();
	for(_int i=0; i<w_sq.size(); i++)
		w_sq[i] = SQ(w_sq[i]);

	_int num_Xf = tst_X_Xf->nr;
	_int num_item_Xf = tst_item_X_Xf->nr;
	_int num_X = tst_X_Xf->nc;
	_int num_Y = w_mat->nc;

	/* select "actlbl" active labels for each instance to facilitate faster tail classifier score calculation */
	for(_int i=0; i<pf_score_mat->nc; i++)
	{
		_int siz = pf_score_mat->size[i];
		_int newsiz = min(siz,actlbl);

		pairIF* vec = pf_score_mat->data[i];
		sort(vec, vec+siz, comp_pair_by_second_desc<_int,_float>);
		Realloc(siz,newsiz,pf_score_mat->data[i]);
		vec = pf_score_mat->data[i];
		sort(vec, vec+newsiz, comp_pair_by_first<_int,_float>);
		pf_score_mat->size[i] = newsiz;
	}

	/* --- calculate tail classifier scores --- */
	SMatF* tmat = pf_score_mat->transpose();
	VecF mask(num_Xf,0);

	for(_int i=0; i<num_Y; i++)
	{
		for(_int j=0; j<w_mat->size[i]; j++)
			mask[w_mat->data[i][j].first] = w_mat->data[i][j].second;

		for(_int j=0; j<tmat->size[i]; j++)
		{
			_int inst = tmat->data[i][j].first;
			_float prod = 0;
			for(_int k=0; k<tst_X_Xf->size[inst]; k++)
			{
				_int id = tst_X_Xf->data[inst][k].first;	
				_float val = tst_X_Xf->data[inst][k].second;
				prod += mask[id]*val;
			}
			_float tmpval = w_sq[i] + 1 - 2*prod;
			tmat->data[i][j].second = 1/(1+exp(param.gamma*tmpval));
		}

		for(_int j=0; j<w_mat->size[i]; j++)
			mask[w_mat->data[i][j].first] = 0;
	}

	SMatF* tail_score_mat = tmat->transpose();

	/* combine PfastXML scores and tail classifier scores to arrive at final scores */ 
	for(_int i=0; i<num_X; i++)
	{
		for(_int j=0; j<pf_score_mat->size[i]; j++)
		{
			_float pf_score = pf_score_mat->data[i][j].second;
			_float tail_score = tail_score_mat->data[i][j].second;
			tail_score_mat->data[i][j].second = pow(pf_score,param.alpha)*pow(tail_score,1-param.alpha);
		}
	}

	predict_time += timer.toc();
	cout << "prediction time: " << ((predict_time/tst_X_Xf->nc)*1000.0) << " ms/point" << endl;
	cout << "model size: " << (model_size/1e+9) << " GB" << endl;

	tail_score_mat->write(score_file);

	delete tst_X_Xf;
	delete tst_item_X_Xf;
	delete w_mat;
	delete pf_score_mat;
	delete tmat;
	delete tail_score_mat;

	return 0;
}
