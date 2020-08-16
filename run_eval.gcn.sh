run_experiment () {
  CMD=$1
  PREFIX=$3/SESS$RANDOM
  mkdir -p $PREFIX
  python -u ${CMD}.py $2 --annotate --annotation-prefix=${PREFIX}/ &> ${PREFIX}/${CMD}.txt
  cat ${PREFIX}/test1-raw/* | python -u dataset/xml_revert.py --textnodes=dataset/test1 --output-dir=${PREFIX}/test1-pred
  cat ${PREFIX}/test2-raw/* | python -u dataset/xml_revert.py --textnodes=dataset/test2 --output-dir=${PREFIX}/test2-pred
  echo "Results for $CMD ($2)" >> ${PREFIX}/${CMD}.txt
  python -u tacEval_relaxed.py -1 -2 dataset/test1 ${PREFIX}/test1-pred >> ${PREFIX}/${CMD}.txt
  python -u tacEval_relaxed.py -1 -2 dataset/test2 ${PREFIX}/test2-pred >> ${PREFIX}/${CMD}.txt
}

# Strategy A = TR22
CUDA_VISIBLE_DEVICES=0   run_experiment main.gcn --strategy=A main.gcn.A

# Strategy B = TR22 + NLM180
CUDA_VISIBLE_DEVICES=1   run_experiment main.gcn --strategy=B main.gcn.B

# Strategy C = TR22 + NLM180 + DDI2013
CUDA_VISIBLE_DEVICES=2   run_experiment main.gcn --strategy=C main.gcn.C