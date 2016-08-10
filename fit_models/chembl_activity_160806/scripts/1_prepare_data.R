# -*- tab-width:2;indent-tabs-mode:t;show-trailing-whitespace:t;rm-trailing-spaces:t -*-
# vi: set ts=2 noet:


library(plyr)
library(dplyr)
library(readr)
library(Zr)
library(ggplot2)
library(tidyr)
source("~/work/sea/scripts/data_repo.R")
data_repo <- get_data_repo("sea_chembl21")

ACTIVE_THRESHOLD <- 1000
INACTIVE_THRESHOLD <- 10000
MAX_TRIPLES_PER_TARGET <- 1000
MAX_TRIPLES_PER_COMPOUND_TARGET <- 100

activities <- data_repo %>%
	dplyr::tbl("activities") %>%
	dplyr::select(
		uniprot_entry,
		zinc_id,
		chembl_smiles,
		activity) %>%
	dplyr::collect(n=Inf)

compounds <- data_repo %>%
	dplyr::tbl("compounds") %>%
	dplyr::select(
		zinc_id,
		chembl_smiles) %>%
	dplyr::collect(n=Inf)


# co-annotated sustances
samples <- activities %>% plyr::ddply(
	.variables="uniprot_entry",
	.fun=function(target_activities){
		actives <- target_activities %>%
			dplyr::filter(activity < ACTIVE_THRESHOLD) %>%
			dplyr::select(zinc_id, chembl_smiles)
		n_samples <- min(
			MAX_TRIPLES_PER_TARGET,
			(nrow(actives) * MAX_TRIPLES_PER_COMPOUND_TARGET))
		samples <- bind_cols(
			actives %>%
				dplyr::slice(1:nrow(actives) %>% sample(n_samples, replace=T)) %>%
				dplyr::select(substance_id = zinc_id, smiles = chembl_smiles),
			actives %>%
				dplyr::slice(1:nrow(actives) %>% sample(n_samples, replace=T)) %>%
				dplyr::select(substance_plus_id = zinc_id, smiles_plus = chembl_smiles)) %>%
			dplyr::filter(substance_id != substance_plus_id)
})

# add decoys
samples <- samples %>%
	bind_cols(
		compounds %>%
			dplyr::slice(1:nrow(compounds) %>% sample(nrow(samples), replace=T)) %>%
			dplyr::select(substance_minus_id = zinc_id, smiles_minus = chembl_smiles))

samples %>% write_tsv("data/activity_triples_160808.tsv")

system("
filter_valid_substances \\
 --input_fname data/activity_triples_160808.tsv \\
 --input_substance_id_colname substance_id \\
 --input_smiles_colname smiles \\
 --ouput_fname data/activity_triples_160808_1.tsv
filter_valid_substances \\
 --input_fname data/activity_triples_160808_1.tsv \\
 --input_substance_id_colname substance_plus_id \\
 --input_smiles_colname smiles_plus \\
 --ouput_fname data/activity_triples_160808_2.tsv
filter_valid_substances \\
 --input_fname data/activity_triples_160808_2.tsv \\
 --input_substance_id_colname substance_minus_id \\
 --input_smiles_colname minus_minus \\
 --ouput_fname data/activity_triples_160808_3.tsv
")

samples <- readr::read_tsv("data/activity_triples_160808_1.tsv")

splits <- 1:nrow(samples) %>% quantile(c(.6, .2, .2)) %>% floor %>% cumsum
samples %>%
	slice(1:splits[1]) %>%
	readr::write_tsv("data/activity_triples_160808_train.tsv")
samples %>%
	slice(splits[1]:splits[2]) %>%
	readr::write_tsv("data/activity_triples_160808_validate.tsv")
samples %>%
	slice(splits[2]:splits[3]) %>%
	readr::write_tsv("data/activity_triples_160808_test.tsv")


system("
python /mnt/nfs/work/momeara/sea/DeepSEA/tensorflow/tensorflow/tensorflow/tensorboard/tensorboard.py  --logdir=log/train_20160809_08-09-02
")
