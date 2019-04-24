#include "swiftXML.h"
#include "fastXML.h"

using namespace std;

pairII get_pos_neg_count( VecI& pos_or_neg, _int num_trn )
{
	pairII counts = make_pair(0,0);
	for(_int i=0; i<num_trn; i++)
	{
		if(pos_or_neg[i]==+1)
			counts.first++;
		else
			counts.second++;
	}
	return counts;
}

void test_svms( VecI& X, SMatF* X_Xf, SMatF* item_X_Xf, VecIF& w, VecIF& item_w, VecI& pos_or_neg, SParam& param )
{
	_float feat_imp = param.feat_imp;
    _int num_Xf = X_Xf->nr;
	pos_or_neg.resize( X.size() );
	VecF values( X.size() );
	VecF item_values( X.size() );

	test_svm( X, X_Xf, w, values );
	test_svm( X, item_X_Xf, item_w, item_values );

	for( _int i=0; i<X.size(); i++ )
	{
		if( feat_imp*values[i] + (1-feat_imp)*item_values[i] >= 0 )
			pos_or_neg[i] = 1;
		else
			pos_or_neg[i] = -1;
	}

	pairII num_pos_neg = get_pos_neg_count( pos_or_neg, pos_or_neg.size() );
}

void swiftXML_setup_thread_locals( _int num_X, _int num_Xf, _int num_item_Xf, _int num_Y )
{
    discounts.resize( num_Y );
    csum_discounts.resize( num_Y+1 );
    
    csum_discounts[0] = 1.0;
    _float sumd = 0;
    for( _int i=0; i<num_Y; i++ )
    {
        discounts[i] = 1.0/log2((_float)(i+2));
        sumd += discounts[i];
        
        if(USE_IDCG)
            csum_discounts[i+1] = sumd;
		else
			csum_discounts[i+1] = 1.0;
    }
    dense_w.resize( max( num_Xf, num_item_Xf ) );
    for( _int i=0; i<max( num_Xf, num_item_Xf ); i++ )
        dense_w[i] = 0;

	countmap.resize( max( max( num_Xf, num_item_Xf ), num_Y ), 0 );
}

void shrink_data_matrices( SMatF* trn_X_Xf, SMatF* trn_item_X_Xf, SMatF* trn_X_Y, _int num_trn_X, VecI& n_X, SMatF*& n_trn_Xf_X, SMatF*& n_trn_item_Xf_X, SMatF*& n_trn_X_Y, VecI& n_Xf, VecI& n_item_Xf, VecI& n_Y, _int& n_num_trn_X )
{
    trn_X_Xf->shrink_mat( n_X, n_trn_Xf_X, n_Xf, countmap, true ); // countmap is a thread_local variable
    trn_item_X_Xf->shrink_mat( n_X, n_trn_item_Xf_X, n_item_Xf, countmap, true ); // countmap is a thread_local variable
    trn_X_Y->shrink_mat( n_X, n_trn_X_Y, n_Y, countmap, false );
	
	n_num_trn_X = 0;
	for( _int i=0; i<n_X.size(); i++ )
		if( n_X[i] < num_trn_X )
			n_num_trn_X++;
}

_bool split_node( SNode* node, SMatF* Xf_X, SMatF* item_Xf_X, SMatF* X_Y, _int num_trn, VecI& pos_or_neg, SParam& param )
{
    _int num_X = Xf_X->nr;
	pos_or_neg.resize( num_X );
 
	for( _int i=0; i<num_X; i++ )
	{
		_llint r = reng();

		if(r%2)
			pos_or_neg[i] = 1;
		else
			pos_or_neg[i] = -1;
	}

	// one run of ndcg optimization
	_bool success;

	success = optimize_ndcg( X_Y, pos_or_neg );
	if(!success)
		return false;

	VecF C( num_X );
	pairII num_pos_neg = get_pos_neg_count( pos_or_neg, num_trn );
	_float frac_pos = (_float)num_pos_neg.first/(num_pos_neg.first+num_pos_neg.second);
	_float frac_neg = (_float)num_pos_neg.second/(num_pos_neg.first+num_pos_neg.second);
	_double Cp = param.item_log_loss_coeff/frac_pos;
	_double Cn = param.item_log_loss_coeff/frac_neg;  // unequal Cp,Cn improves the balancing in some data sets

	for( _int i=0; i<num_X; i++ )
		C[i] = pos_or_neg[i]==+1 ? Cp : Cn;

	// one run of log-loss optimization on item features
	success = optimize_log_loss( item_Xf_X, pos_or_neg, C, node->item_w, param );

	num_pos_neg = get_pos_neg_count( pos_or_neg, num_trn );
	frac_pos = (_float)num_pos_neg.first/(num_pos_neg.first+num_pos_neg.second);
	frac_neg = (_float)num_pos_neg.second/(num_pos_neg.first+num_pos_neg.second);
	Cp = param.log_loss_coeff/frac_pos;
	Cn = param.log_loss_coeff/frac_neg;  // unequal Cp,Cn improves the balancing in some data sets
	
	for( _int i=0; i<num_X; i++ )
		C[i] = pos_or_neg[i]==+1 ? Cp : Cn;

	for( _int i=0; i<num_X; i++ )
		C[i] = i<num_trn ? C[i] : 0;

	// one run of log-loss optimization on user features
	success = optimize_log_loss( Xf_X, pos_or_neg, C, node->w, param );

	if(!success)
		return false;
	return true;
}

void postprocess_node( SNode* node, SMatF* trn_X_Xf, SMatF* trn_item_X_Xf, SMatF* trn_X_Y, VecI& n_X, VecI& n_Xf, VecI& n_item_Xf, VecI& n_Y, VecI& pos_or_neg, SParam& param )
{
    if( node->is_leaf )
        reindex_VecIF( node->leaf_dist, n_Y );
    else
	{
        reindex_VecIF( node->w, n_Xf );
		reindex_VecIF( node->item_w, n_item_Xf );
		test_svms( node->X, trn_X_Xf, trn_item_X_Xf, node->w, node->item_w, pos_or_neg, param );
	}
}

STree* swiftXML_train_tree( SMatF* trn_X_Xf, SMatF* trn_item_X_Xf, SMatF* trn_X_Y, SParam& param, _int tree_no )
{
	reng.seed(tree_no);

	_int num_X = trn_X_Xf->nc;
	_int num_Xf = trn_X_Xf->nr;
	_int num_Y = trn_X_Y->nr;

	STree* tree = new STree;
	vector<SNode*>& nodes = tree->nodes;

	VecI X;
	for(_int i=0; i<num_X; i++)
		X.push_back(i);
	SNode* root = new SNode( X, 0, param.max_leaf );
	nodes.push_back(root);

	VecI pos_or_neg;

	for(_int i=0; i<nodes.size(); i++)
	{
		if(loglvl == LOGLVL::PROGRESS)
		{
			if(i%1000==0)
				cout<<"\tnode "<<i<<endl;
		}		

		SNode* node = nodes[i];
		VecI& n_X = node->X;	
		SMatF* n_trn_Xf_X;
		SMatF* n_trn_item_Xf_X;
		SMatF* n_trn_X_Y;
		VecI n_Xf;
		VecI n_item_Xf;
        VecI n_Y;
		_int n_num_trn_X;

		shrink_data_matrices( trn_X_Xf, trn_item_X_Xf, trn_X_Y, param.num_trn_X, n_X, n_trn_Xf_X, n_trn_item_Xf_X, n_trn_X_Y, n_Xf, n_item_Xf, n_Y, n_num_trn_X );

		if(node->is_leaf)
		{
			calc_leaf_prob( node, n_trn_X_Y, param );
			postprocess_node( node, trn_X_Xf, trn_item_X_Xf, trn_X_Y, n_X, n_Xf, n_item_Xf, n_Y, pos_or_neg, param );
        }
		else
		{
			VecI pos_or_neg;
			bool success = split_node( node, n_trn_Xf_X, n_trn_item_Xf_X, n_trn_X_Y, n_num_trn_X, pos_or_neg, param );

			if(success)
			{
				postprocess_node( node, trn_X_Xf, trn_item_X_Xf, trn_X_Y, n_X, n_Xf, n_item_Xf, n_Y, pos_or_neg, param );

				VecI pos_X, neg_X;
				for(_int j=0; j<n_X.size(); j++)
				{
					_int inst = n_X[j];
					if( pos_or_neg[j]==+1 )
						pos_X.push_back(inst);
					else
						neg_X.push_back(inst);
				}
	
				SNode* pos_node = new SNode( pos_X, node->depth+1, param.max_leaf );
				nodes.push_back(pos_node);
				node->pos_child = nodes.size()-1;

				SNode* neg_node = new SNode( neg_X, node->depth+1, param.max_leaf );
				nodes.push_back(neg_node);
				node->neg_child = nodes.size()-1;
			}
			else
			{
				node->is_leaf = true;
				i--;
			}
		}

		delete n_trn_Xf_X;
		delete n_trn_item_Xf_X;
		delete n_trn_X_Y;
	}
	tree->num_Xf = num_Xf;
	tree->num_Y = num_Y;

	return tree;
}

void swiftXML_train_trees_thread( SMatF* trn_X_Xf, SMatF* trn_item_X_Xf, SMatF* trn_X_Y, SParam param, _int s, _int t, string model_dir, _float* train_time )
{
	Timer timer;
	timer.tic();
    _int num_X = trn_X_Xf->nc;
    _int num_Xf = trn_X_Xf->nr;
    _int num_item_Xf = trn_item_X_Xf->nr;
    _int num_Y = trn_X_Y->nr;
    swiftXML_setup_thread_locals( num_X, num_Xf, num_item_Xf, num_Y );
    {
		lock_guard<mutex> lock(mtx);
		*train_time += timer.toc();
    }
    
	for(_int i=s; i<s+t; i++)
	{
		timer.tic();
		cout<<"tree "<<i<<" training started"<<endl;

		STree* tree = swiftXML_train_tree( trn_X_Xf, trn_item_X_Xf, trn_X_Y, param, i );
		{
			lock_guard<mutex> lock(mtx);
			*train_time += timer.toc();
		}

		tree->write( model_dir, i );

		timer.tic();
		delete tree;

		cout<<"tree "<<i<<" training completed"<<endl;
		
		{
			lock_guard<mutex> lock(mtx);
			*train_time += timer.toc();
		}
	}
}

void swiftXML_train_trees( SMatF* trn_X_Xf, SMatF* trn_item_X_Xf, SMatF* trn_X_Y, SParam& param, string model_dir, _float& train_time )
{
	_float* t_time = new _float;
	*t_time = 0;
	Timer timer;
	
	timer.tic();
	trn_X_Xf->append_bias_feat( param.bias );
	trn_item_X_Xf->append_bias_feat( param.bias );
    
	_int tree_per_thread = (_int)ceil((_float)param.num_tree/param.num_thread);
	vector<thread> threads;
	_int s = param.start_tree;
	for( _int i=0; i<param.num_thread; i++ )
	{
		if( s < param.start_tree+param.num_tree )
		{
			_int t = min( tree_per_thread, param.start_tree+param.num_tree-s );
			threads.push_back( thread( swiftXML_train_trees_thread, trn_X_Xf, trn_item_X_Xf, trn_X_Y, param, s, t, model_dir, ref(t_time) ));
			s += t;
		}
	}
	*t_time += timer.toc();	

	for(_int i=0; i<threads.size(); i++)
		threads[i].join();

	train_time = *t_time;
	delete t_time;
}


SMatF* swiftXML_predict_tree( SMatF* tst_X_Xf, SMatF* tst_item_X_Xf, STree* tree, SParam& param )
{
	_int num_X = tst_X_Xf->nc;
	_int num_Xf = param.num_Xf;
	_int num_item_Xf = param.num_item_Xf;
	_int num_Y = param.num_Y;

	vector<SNode*>& nodes = tree->nodes;
	SNode* node = nodes[0];
	node->X.clear();

	for(_int i=0; i<num_X; i++)
		node->X.push_back(i);

	SMatF* tst_score_mat = new SMatF(num_Y,num_X);
	VecI pos_or_neg;

	for(_int i=0; i<nodes.size(); i++)
	{
		if(loglvl == LOGLVL::PROGRESS)
		{
			if(i%1000==0)
				cout<<"\tnode "<<i<<endl;
		}		

		SNode* node = nodes[i];
	
		if(!node->is_leaf)
		{
			VecI& X = node->X;
			test_svms(X, tst_X_Xf, tst_item_X_Xf, node->w, node->item_w, pos_or_neg, param);
			SNode* pos_node = nodes[node->pos_child];
			pos_node->X.clear();
			SNode* neg_node = nodes[node->neg_child];
			neg_node->X.clear();

			for(_int j=0; j<X.size(); j++)
			{
				if(pos_or_neg[j]==+1)
					pos_node->X.push_back(X[j]);
				else
					neg_node->X.push_back(X[j]);
			}
		}
		else
		{
			VecI& X = node->X;
			VecIF& leaf_dist = node->leaf_dist;
			_int* size = tst_score_mat->size;
			pairIF** data = tst_score_mat->data;

			for(_int j=0; j<X.size(); j++)
			{
				_int inst = X[j];
				size[inst] = leaf_dist.size();
				data[inst] = new pairIF[leaf_dist.size()];

				for(_int k=0; k<leaf_dist.size(); k++)
					data[inst][k] = leaf_dist[k];
			}
		}
	}

	return tst_score_mat;
}

void swiftXML_predict_trees_thread( SMatF* tst_X_Xf, SMatF* tst_item_X_Xf, SMatF* score_mat, SParam param, _int s, _int t, string model_dir, _float* prediction_time, _float* model_size )
{
    Timer timer;
    
    timer.tic();
    _int num_Xf = tst_X_Xf->nr;
	_int num_item_Xf = tst_item_X_Xf->nr;
    dense_w.resize( max(num_Xf, num_item_Xf) );
    for( _int i=0; i<max(num_Xf, num_item_Xf); i++ )
        dense_w[i] = 0;
	{
		lock_guard<mutex> lock(mtx);
		*prediction_time += timer.toc();
	}

	for(_int i=s; i<s+t; i++)
	{
		if(loglvl == LOGLVL::PROGRESS)
			cout<<"tree "<<i<<" testing started"<<endl;

		STree* tree = new STree( model_dir, i );
        timer.tic();
		SMatF* tree_score_mat = swiftXML_predict_tree( tst_X_Xf, tst_item_X_Xf, tree, param );

		{
			lock_guard<mutex> lock(mtx);
			score_mat->add(tree_score_mat);
            *model_size += tree->get_ram();
		}

		delete tree;
		delete tree_score_mat;

		if(loglvl == LOGLVL::PROGRESS)
			cout<<"tree "<<i<<" testing completed"<<endl;
        {
			lock_guard<mutex> lock(mtx);
			*prediction_time += timer.toc();
		}
	}
}

SMatF* swiftXML_predict_trees( SMatF* tst_X_Xf, SMatF* tst_item_X_Xf, SParam& param, string model_dir, _float& prediction_time, _float& model_size )
{
    _float* p_time = new _float;
	*p_time = 0;

	_float* m_size = new _float;
	*m_size = 0;

	Timer timer;

	timer.tic();
    tst_X_Xf->append_bias_feat( param.bias );
    tst_item_X_Xf->append_bias_feat( param.bias );
    
    _int num_X = tst_X_Xf->nc;
    SMatF* score_mat = new SMatF( param.num_Y, num_X );

	_int tree_per_thread = (_int)ceil((_float)param.num_tree/param.num_thread);
	vector<thread> threads;

	_int s = param.start_tree;
	for(_int i=0; i<param.num_thread; i++)
	{
		if(s < param.start_tree+param.num_tree)
		{
			_int t = min(tree_per_thread, param.start_tree+param.num_tree-s);
            threads.push_back( thread( swiftXML_predict_trees_thread, tst_X_Xf, tst_item_X_Xf, ref(score_mat), param, s, t, model_dir, ref( p_time ), ref( m_size ) ));
			s += t;
		}
	}
    *p_time += timer.toc();
	
	for(_int i=0; i<threads.size(); i++)
		threads[i].join();

    timer.tic();

	for(_int i=0; i<score_mat->nc; i++)
		for(_int j=0; j<score_mat->size[i]; j++)
			score_mat->data[i][j].second /= param.num_tree;
    
    model_size = *m_size;
	delete m_size;
    
    *p_time += timer.toc();
	prediction_time = *p_time;
	delete p_time;

	return score_mat;
}
