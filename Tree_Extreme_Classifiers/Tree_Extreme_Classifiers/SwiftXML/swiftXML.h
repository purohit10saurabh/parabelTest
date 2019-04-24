#pragma once

#include "PfastreXML.h"
#include "fastXML.h"

class SParam : public PfParam 
{
public:
	_int num_trn_X;
	_int num_item_Xf;
	_float item_log_loss_coeff;
	_float feat_imp;

	SParam(): PfParam()
	{
		num_trn_X = 0;
		num_item_Xf = 0;
		item_log_loss_coeff = 1.0;
		feat_imp = 0.5;
	}

	SParam( string fname )
	{
		check_valid_filename( fname, true );
		ifstream fin;
		fin.open(fname);
		fin >> (*this);
		fin.close();
	}

	void write( string fname )
	{
		check_valid_filename( fname, false );
		ofstream fout;
		fout.open(fname);
		fout << (*this);
		fout.close();
	}

	friend istream& operator>>( istream& fin, SParam& sparam )
	{
		fin >> static_cast<PfParam&>( sparam );
		fin >> sparam.num_trn_X;
		fin >> sparam.num_item_Xf;
		fin >> sparam.item_log_loss_coeff;
		fin >> sparam.feat_imp;
		return fin;
	}

	friend ostream& operator<<( ostream& fout, const SParam& sparam )
	{
		fout << static_cast<const PfParam&>( sparam );
		fout << sparam.num_trn_X << "\n";
		fout << sparam.num_item_Xf << "\n";
		fout << sparam.item_log_loss_coeff << "\n";
		fout << sparam.feat_imp << endl;
		return fout;
	}
};

class SNode : public Node
{
public:
	VecIF item_w;

	SNode() : Node()
	{
		return;
	}

	SNode( VecI X, _int depth, _int max_leaf ) : Node( X, depth, max_leaf )
	{
		return;
	}

	~SNode()
	{
		return;
	}

	_float get_ram()
	{
		_float ram = sizeof( SNode );
		ram += X.size() * sizeof( _int );
		ram += w.size() * sizeof( pairIF );
		ram += leaf_dist.size() * sizeof( pairIF );
		ram += item_w.size() * sizeof( pairIF );
		return ram;
	}

	friend ostream& operator<<(ostream& fout, const SNode& snode)
	{
		fout << static_cast<const Node&> ( snode );
		if( ! snode.is_leaf )
		{
			fout << snode.item_w.size();
			for( _int i=0; i<snode.item_w.size(); i++ )
			{
				fout << " " << snode.item_w[i].first << ":" << snode.item_w[i].second;
			}
			fout << "\n";
		}
		return fout;
	}

	friend istream& operator>>( istream& fin, SNode& snode )
	{
		fin >> static_cast<Node&> ( snode );
        if( ! snode.is_leaf )
        {
            _int siz;
            _int ind;
            _float val;
            char c;
            snode.item_w.clear();
            fin >> siz;
            for( _int i=0; i<siz; i++ )
            {
                fin >> ind >> c >> val;
                snode.item_w.push_back( make_pair( ind, val ) );
            }
        }
		return fin;
	}
};

typedef GTree<SNode> STree;

void swiftXML_train_trees( SMatF* trn_X_Xf, SMatF* trn_item_X_Xf, SMatF* trn_X_Y, SParam& param, string model_dir, _float& train_time );
SMatF* swiftXML_predict_trees( SMatF* tst_X_Xf, SMatF* tst_item_X_Xf, SParam& param, string model_dir, _float& prediction_time, _float& model_size );

