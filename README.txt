==============================================================================
Joint Drug Interaction Information Extraction 
with Attention-Gated Graph Convolutions

Code Author: Tung Tran (tung.tran@uky.edu)

==============================================================================

Requirements:
-- tensorflow version 1.12
-- sklearn
-- gensim
-- nltk
-- pandas
-- scipy

==============================================================================

To run the experiments, first download pretrain embeddings from: 
http://evexdb.org/pmresources/vec-space-models/

You can use PMC-w2v.bin, or --no-embeddings to use randomly initialized 
embeddings instead.

The program runs in either (1) testing and annotation mode or (2) n-fold cross 
validation mode. Use (1) to generate final results on the test sets and (2) to 
obtain validation results without touching the test sets. By default, it will 
run in mode (1) unless you specify --xval-dir to indicate a directory to send 
cross-validation results. Use --annotate with mode (1) to print out results to 
a directory. The --strategy argument can be set to `A` to just train on TAC22 
(default), `B` to train on TAC22 and NLM180, and `C` to train on TAC22, 
NLM180, and DDI2013. Run --help for a full list of program parameters.

The *dataset* directory contains all relevant dataset files in both XML and a 
simpler custom text-based format. The scripts `xml_convert.py` and 
`xml_revert.py` in the *dataset* folder can be used to convert back and forth 
between TAC DDI XML format and our unified text format. We only provide data in our unified format. For the original XML source, please see the original task page:
https://bionlp.nlm.nih.gov/tac2018druginteractions/

==============================================================================

Other important files:

* tacEval_relaxed.py --- official script used for test set evaluation
* run_eval.gcn.sh --- example bash file to perform an end-to-end run of an experiment
* bootstrap.py --- used to perform voting-based ensembling

==============================================================================

The full program settings are as follows:

usage: main.py [-h] [--annotate] [--xval-dir XVAL_DIR]
               [--heldout-count HELDOUT_COUNT] [--strategy STRATEGY] [--coord]
               [--annotation-prefix ANNOTATION_PREFIX] [--no-embeddings]
               [--stemming] [--embeddings EMBEDDINGS]
               [--hidden-size HIDDEN_SIZE]
               [--char-embedding-size CHAR_EMBEDDING_SIZE]
               [--gcn-hidden-size GCN_HIDDEN_SIZE]
               [--gcn-attn-size GCN_ATTN_SIZE] [--gcn-depth GCN_DEPTH]
               [--bucket-min BUCKET_MIN] [--bucket-max BUCKET_MAX]
               [--num-epoch NUM_EPOCH] [--early-stopping EARLY_STOPPING]
               [--devseed DEVSEED] [--load-checkpoint LOAD_CHECKPOINT]


optional arguments:
  -h, --help            show this help message and exit
  --annotate            annotate test files instead of xval
  --xval-dir XVAL_DIR   path to dump annotation for xval
  --heldout-count HELDOUT_COUNT
                        number of drugs to hold out for xval
  --strategy STRATEGY   strategy to use for training: [A, B, C]
  --coord               combine coordinated mentions
  --annotation-prefix ANNOTATION_PREFIX
                        prefix for prediction output directory
  --no-embeddings       use word embeddings
  --stemming            enable stemming of tokens
  --embeddings EMBEDDINGS
                        specify pretrained word embeddings
  --hidden-size HIDDEN_SIZE
                        hidden size for RNN reps
  --char-embedding-size CHAR_EMBEDDING_SIZE
                        size of character embeddings
  --gcn-hidden-size GCN_HIDDEN_SIZE
                        size of attention rep for GCNs
  --gcn-attn-size GCN_ATTN_SIZE
                        size of attention rep for GCNs
  --gcn-depth GCN_DEPTH
                        number of applications for GCNs
  --bucket-min BUCKET_MIN
                        skipping experiments below this bucket index
  --bucket-max BUCKET_MAX
                        skipping experiments above this bucket index
  --num-epoch NUM_EPOCH
                        number of epochs to run
  --early-stopping EARLY_STOPPING
                        number of epochs without improvements before halting
                        training
  --devseed DEVSEED     seed value for picking dev examples
  --load-checkpoint LOAD_CHECKPOINT
                        load the following checkpoint instead of training from
                        scratch