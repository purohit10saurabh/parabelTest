function swiftXML_evaluate_predictions( score_mat, inc_tst_X_Y, exc_tst_X_Y, inv_prop )

	score_mat( inc_tst_X_Y>0 ) = 0;
	metrics = get_all_metrics( score_mat, exc_tst_X_Y, inv_prop );
end
