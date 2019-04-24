function swiftXML_preprocess_data( trn_file, tst_file, Y_Yf_file, trn_X_Xf_file, trn_item_X_Xf_file, trn_X_Y_file, tst_X_Xf_file, tst_item_X_Xf_file, inc_tst_X_Y_file, exc_tst_X_Y_file, inv_prop_file, frac, A, B )

    system( sprintf( 'perl ../Tools/convert_format.pl %s %s %s', trn_file, trn_X_Xf_file, trn_X_Y_file ) );
    tst_X_Y_file = tempname;
    system( sprintf( 'perl ../Tools/convert_format.pl %s %s %s', tst_file, tst_X_Xf_file, tst_X_Y_file ) );

    trn_X_Xf = read_text_mat( trn_X_Xf_file );
	tst_X_Xf = read_text_mat( tst_X_Xf_file );
	trn_X_Y = read_text_mat( trn_X_Y_file  );			
	tst_X_Y = read_text_mat( tst_X_Y_file  );
	Y_Yf = read_text_mat( Y_Yf_file  );

   	inc_tst_X_Y = get_inc_mat( tst_X_Y, frac );
    exc_tst_X_Y = tst_X_Y - inc_tst_X_Y;
    inv_prop = inv_propensity( trn_X_Y, A, B );
	
	prm.type = 'NORM_1';
	trn_item_X_Xf = Y_Yf*trn_X_Y;
	[~,trn_item_X_Xf] = norm_features.fit_transform(trn_item_X_Xf,prm);
	tst_item_X_Xf = Y_Yf*inc_tst_X_Y;
	[~,tst_item_X_Xf] = norm_features.fit_transform(tst_item_X_Xf,prm);

	trn_X_Xf = [ trn_X_Xf tst_X_Xf ];
	trn_item_X_Xf = [ trn_item_X_Xf tst_item_X_Xf ];	
	trn_X_Y = [ trn_X_Y inc_tst_X_Y ];
	
    write_text_mat( trn_X_Xf, trn_X_Xf_file );
    write_text_mat( trn_X_Y, trn_X_Y_file );
	write_text_mat( trn_item_X_Xf, trn_item_X_Xf_file );
	write_text_mat( tst_item_X_Xf, tst_item_X_Xf_file );
	write_text_mat( inc_tst_X_Y, inc_tst_X_Y_file );
	write_text_mat( exc_tst_X_Y, exc_tst_X_Y_file );
	csvwrite( inv_prop_file, inv_prop );
end

function inc_tst_X_Y = get_inc_mat( tst_X_Y, frac )
	% expects unweighted label matrix
	% outputs subsampled unweighted label matrix

	seed = 1;
    rng( seed );
 
    num_tst = size(tst_X_Y,2);
    num_lbl = size(tst_X_Y,1);
    [X,Y,V] = find(tst_X_Y);
   
    V = rand(numel(V),1);
    rd_mat = sparse(X,Y,V,num_lbl,num_tst);
    rank_mat = sort_sparse_mat(rd_mat); % This function is available from "Tools/matlab" folder
    
    ths = floor(frac*sum(tst_X_Y,1));
    th_mat = bsxfun(@times, tst_X_Y, ths);
    
    rank_mat(rank_mat>th_mat) = 0;
    inc_tst_X_Y = spones(rank_mat);
end
