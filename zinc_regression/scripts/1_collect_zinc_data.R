# -*- tab-width:2;indent-tabs-mode:t;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=2 noet:


library(plyr)
library(dplyr)
library(readr)
library(Zr)
library(ggplot2)
library(tidyr)

catalog_short_names <- c(
	"hmdbendo",
	"aggregators")

substances <- plyr::ldply(catalog_short_names, function(catalog_short_name){
	substances <- Zr::catalog_items(
		catalog_short_name=catalog_short_name,
		result_batch_size=1000,
	output_fields=c(
		"zinc_id",
		"supplier_code",
		"substance.preferred_name",
		"substance.smiles",
		"substance.purchasable",
		"substance.purchasability",
		"substance.rb",
		"substance.reactive",
		"substance.features"),
		verbose=T) %>%
		dplyr::mutate(catalog=catalog_short_name)
	substances %>%
		readr::write_tsv(paste0("data/", catalog_short_name,"_substances_", Zr::date_code(), ".tsv"))
	substances
})


protomers <- plyr::ldply(catalog_short_names, function(catalog_short_name){
	protomers <- Zr::catalog_protomers(
		catalog_short_name=catalog_short_name,
		result_batch_size=1000,
		output_fields=c(
			"zinc_id",
			"prot_id",
			"smiles",
			"net_charge",
			"desolv_apol",
			"desolv_pol",
			"ph_mod_fk",
			"true_logp",
			"true_mwt",
			"hba",
			"hbd",
			"num_aliphatic_rings",
			"num_aromatic_rings",
			"num_heavy_atoms",
			"num_rotatable_bonds",
			"chiral_centers",
			"reactive",
			"reactivity",
			"tpsa",
			"tranche_name",
			"tranche_prefix"),
		verbose=T) %>%
		dplyr::mutate(catalog=catalog_short_name)
	protomers %>%
		readr::write_tsv(paste0("data/", catalog_short_name,"_protomers_", Zr::date_code(), ".tsv"))
	protomers
})

system("
/mnt/nfs/work/momeara/tools/anaconda2/bin/python scripts/train_fingerprint_function.py \\
  --input_data_fname data/hmdbendo_protomers_160606.csv \\
  --output_fp_function_fname data/hmdbendo_protomers_160606_net_charge.fp_func \\
  --output_training_curve_fname data/hmdbendo_protomers_160606_net_charge.fp_func.curve \\
  --verbose \\
  --smiles_column smiles \\
  --target_column num_rotatable_bonds \\
  --N_train 800 \\
  --N_validate 800 \\
  --N_test 800 \\
  --seed 0 \\
  --fp_length 512 \\
  --fp_depth 4 \\
  --fp_width 20 \\
  --h1_size 100 \\
  --log_l2_penalty -2 \\
  --fp_normalize \\
  --nll_func_name neuralfingerprint.util.rmse \\
  --prediction_layer_sizes 512 100 \\
  --num_iters 1000 \\
  --batch_size 100 \\
  --log_init_scale -4 \\
  --log_stepsize -6 \\
  --log_b1 -3 \\
  --log_b2 -2
")

loss_curve <- readr::read_csv("data/hmdbendo_protomers_160606_net_charge.fp_func.curve") %>%
	dplyr::mutate(iter=1:n())
ggplot2::ggplot(loss_curve ) + ggplot2::theme_bw() +
	ggplot2::geom_line(aes(x=iter, y=loss))
ggplot2::ggsave("data/hmdbendo_protomers_160606_net_charge.fp_func.curve.pdf")

# weights parser
#OrderedDict([
#	 (('layer output weights', 0), (slice(0, 31744, None), (62, 512))),      # (num_atom_features, fp_length)
#  (('layer output bias', 0), (slice(31744, 32256, None), (1, 512))),      # (1, fp_length)
#  (('layer output weights', 1), (slice(32256, 42496, None), (20, 512))),  # (num_hidden_features[0], fp_length)
#  (('layer output bias', 1), (slice(42496, 43008, None), (1, 512))),      # 
#  (('layer output weights', 2), (slice(43008, 53248, None), (20, 512))),
#  (('layer output bias', 2), (slice(53248, 53760, None), (1, 512))),
#  (('layer output weights', 3), (slice(53760, 64000, None), (20, 512))),
#  (('layer output bias', 3), (slice(64000, 64512, None), (1, 512))),
#  (('layer output weights', 4), (slice(64512, 74752, None), (20, 512))),
#  (('layer output bias', 4), (slice(74752, 75264, None), (1, 512))),
#  (('layer', 0, 'biases'), (slice(75264, 75284, None), (1, 20))),
#  (('layer', 0, 'self filter'), (slice(75284, 76524, None), (62, 20))),
#  ('layer 0 degree 0 filter', (slice(76524, 77884, None), (68, 20))),
#  ('layer 0 degree 1 filter', (slice(77884, 79244, None), (68, 20))),
#  ('layer 0 degree 2 filter', (slice(79244, 80604, None), (68, 20))),
#  ('layer 0 degree 3 filter', (slice(80604, 81964, None), (68, 20))),
#  ('layer 0 degree 4 filter', (slice(81964, 83324, None), (68, 20))),
#  ('layer 0 degree 5 filter', (slice(83324, 84684, None), (68, 20))),
#  (('layer', 1, 'biases'), (slice(84684, 84704, None), (1, 20))),
#  (('layer', 1, 'self filter'), (slice(84704, 85104, None), (20, 20))),
#  ('layer 1 degree 0 filter', (slice(85104, 85624, None), (26, 20))),
#  ('layer 1 degree 1 filter', (slice(85624, 86144, None), (26, 20))),
#  ('layer 1 degree 2 filter', (slice(86144, 86664, None), (26, 20))),
#  ('layer 1 degree 3 filter', (slice(86664, 87184, None), (26, 20))),
#  ('layer 1 degree 4 filter', (slice(87184, 87704, None), (26, 20))),
#  ('layer 1 degree 5 filter', (slice(87704, 88224, None), (26, 20))),
#  (('layer', 2, 'biases'), (slice(88224, 88244, None), (1, 20))),
#  (('layer', 2, 'self filter'), (slice(88244, 88644, None), (20, 20))),
#  ('layer 2 degree 0 filter', (slice(88644, 89164, None), (26, 20))),
#  ('layer 2 degree 1 filter', (slice(89164, 89684, None), (26, 20))),
#  ('layer 2 degree 2 filter', (slice(89684, 90204, None), (26, 20))),
#  ('layer 2 degree 3 filter', (slice(90204, 90724, None), (26, 20))),
#  ('layer 2 degree 4 filter', (slice(90724, 91244, None), (26, 20))),
#  ('layer 2 degree 5 filter', (slice(91244, 91764, None), (26, 20))),
#  (('layer', 3, 'biases'), (slice(91764, 91784, None), (1, 20))),
#  (('layer', 3, 'self filter'), (slice(91784, 92184, None), (20, 20))),
#  ('layer 3 degree 0 filter', (slice(92184, 92704, None), (26, 20))),
#  ('layer 3 degree 1 filter', (slice(92704, 93224, None), (26, 20))),
#  ('layer 3 degree 2 filter', (slice(93224, 93744, None), (26, 20))),
#  ('layer 3 degree 3 filter', (slice(93744, 94264, None), (26, 20))),
#  ('layer 3 degree 4 filter', (slice(94264, 94784, None), (26, 20))),
#  ('layer 3 degree 5 filter', (slice(94784, 95304, None), (26, 20)))])
#


system("
/mnt/nfs/work/momeara/tools/anaconda2/bin/python scripts/train_fingerprint_function-tensorflow.py \\
  --input_data_fname data/hmdbendo_protomers_160606.csv \\
  --output_fp_function_fname data/hmdbendo_protomers_160606_net_charge.fp_func \\
  --output_training_curve_fname data/hmdbendo_protomers_160630_net_charge.fp_func.curve \\
  --verbose \\
  --smiles_column smiles \\
  --target_column true_mwt \\
  --N_train 6000 \\
  --N_validate 1000 \\
  --N_test 1000 \\
  --fp_length 512 \\
  --fp_depth 6 \\
  --fp_width 20 \\
  --h1_size 100 \\
  --l2_penalty .01 \\
  --l1_penalty 0.0 \\
  --fp_normalize \\
  --prediction_layer_sizes 512 100 \\
  --epochs 15 \\
  --batch_size 10 \\
  --eval_frequency 100 \\
  --log_init_scale -4 \\
  --log_learning_rate -4 \\
  --log_stepsize -6 \\
  --log_b1 -3 \\
  --log_b2 -2
")


loss_curve <- readr::read_csv("data/hmdbendo_protomers_160630_net_charge.fp_func.curve") %>%
	dplyr::mutate(batch=1:n()) %>%
	tidyr::gather("curve", "rmse", -batch) %>%
	dplyr::mutate(rmse = as.numeric(ifelse(rmse == "None", NA, rmse)))

ggplot2::ggplot(loss_curve) + ggplot2::theme_bw() +
#	ggplot2::geom_line(aes(x=batch, y=rmse, color=curve)) +
	ggplot2::geom_smooth(aes(x=batch, y=rmse, color=curve), span = 0.00001) +
	ggplot2::scale_y_continuous("RMSE")
ggplot2::ggsave("data/hmdbendo_protomers_160630_net_charge.fp_func.curve.pdf", height=4, width=10)
ggplot2::ggsave("data/hmdbendo_protomers_160630_net_charge.fp_func.curve.png", height=4, width=10)
