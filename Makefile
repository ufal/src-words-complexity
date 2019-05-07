SHELL=/bin/bash

LRC_FLAG=
ifeq ($(LRC),1)
LRC_FLAG=-p --jobs 100 --priority=0 --mem=10g --workdir=tmp/treex_runs/{NNN}-run-{XXXX}
endif
TREEX=treex $(LRC_FLAG)

ALIGNER=/home/mnovak/projects/align_coref/AnaphBus/align4treex/scripts/run_extra_only.sh

include makefile.split_token
include makefile.for_giza
include makefile.align
include makefile.labels
include makefile.train_dev_test

data/de-de_mt.giza.gz :
	LRC=$(LRC) TMPDIR=$(PWD)/tmp EXTRA_NAME=all_extra EXTRA_SAMPLE_PERC="100" $(ALIGNER) $@ de-de_mt
