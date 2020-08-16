import os, sys, time, random, re, json, collections, argparse, sklearn.metrics
import pandas as pd
import numpy as np
import tensorflow as tf
import scipy.stats.mstats as stats
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.stem import PorterStemmer

parser = argparse.ArgumentParser(description='Process drug labels')          
parser.add_argument('--annotate', default=False, action='store_true',
                    help='annotate test files instead of xval')
parser.add_argument('--xval-dir', type=str, default=None,
                    help='path to dump annotation for xval')
parser.add_argument('--heldout-count', default=2, type=int, action='store',
                    help='number of drugs to hold out for xval')
parser.add_argument('--strategy', default='A', type=str,
                    help='strategy to use for training: [A, B, C]')
parser.add_argument('--coord', default=False, action='store_true',
                    help='combine coordinated mentions')
#parser.add_argument('--nlm180', default=False, action='store_true',
#                    help='additionally train on full nlm180 data')
parser.add_argument('--annotation-prefix', default='', type=str,
                    help='prefix for prediction output directory')
parser.add_argument('--no-embeddings', default=False, action='store_true',
                    help='use word embeddings')
parser.add_argument('--stemming', default=False, action='store_true',
                    help='enable stemming of tokens')
parser.add_argument('--embeddings', default='PMC-w2v.bin', type=str, action='store',
                    help='specify pretrained word embeddings')
parser.add_argument('--hidden-size', default=50, type=int, action='store',
                    help='hidden size for RNN reps')
parser.add_argument('--char-embedding-size', default=25, type=int, action='store',
                    help='size of character embeddings')
parser.add_argument('--gcn-hidden-size', default=100, type=int, action='store',
                    help='size of attention rep for GCNs')
parser.add_argument('--gcn-attn-size', default=25, type=int, action='store',
                    help='size of attention rep for GCNs')
parser.add_argument('--gcn-depth', default=1, type=int, action='store',
                    help='number of applications for GCNs')
parser.add_argument('--bucket-min', default=0, type=int, action='store',
                    help='skipping experiments below this bucket index')
parser.add_argument('--bucket-max', default=10, type=int, action='store',
                    help='skipping experiments above this bucket index')
parser.add_argument('--num-epoch', default=100, type=int, action='store',
                    help='number of epochs to run')
parser.add_argument('--early-stopping', default=50, type=int, action='store',
                    help='number of epochs without improvements before halting training')
parser.add_argument('--devseed', default=0, type=int, action='store',
                    help='seed value for picking dev examples')
parser.add_argument('--load-checkpoint', default='', type=str,
                    help='load the following checkpoint instead of training from scratch')

NeuralModel = collections.namedtuple('NeuralModel',  ['word_vocab', 'seqlen', 'charlen', 'sess', 
        'saver', 'dropout_keep', 'target_ner', 'target_pd', 'target_pk', 'y_loss', 'train_step'])

Sentence = collections.namedtuple('Sentence',['drug_name', 'sentence_id','sect_id',
                                            'set_id','orig_sent','tokens','entity_labels', 'pd_relations', 'pk_relations'])
PD = collections.namedtuple('PD',['sentence', 'precipitant', 'effect', 'code'])
PK = collections.namedtuple('PK',['sentence', 'precipitant', 'code'])

Token = collections.namedtuple('Token',['id','form','lemma','upos','xpos','feats','head',
                                        'deprel','start','end','form2'])

Target = collections.namedtuple('Target',  ['labels','y_true','y_out','y_loss','train_step', 'symbols'])
Eval = collections.namedtuple('Eval',  ['acc'])
EvalNER = collections.namedtuple('Eval',  ['f','p','r','f_partial','p_partial','r_partial'])

NER_BILOU = [ 'O', 'U-DYN', 'B-DYN', 'I-DYN', 'L-DYN',
                    'U-KIN', 'B-KIN', 'I-KIN', 'L-KIN', 
                    'U-UNK', 'B-UNK', 'I-UNK', 'L-UNK',
                    'U-TRI', 'B-TRI', 'I-TRI', 'L-TRI',
                    'U-EFF', 'B-EFF', 'I-EFF', 'L-EFF' ]

def main(args):
    print(args)
    tac22_fname = 'dataset/training22'
    with open(tac22_fname + '.txt','r') as f:
        with open(tac22_fname + '.conllu.txt','r') as fconllu:
            dataset = load_data(f,fconllu,stemming=args.stemming)

    nlm180 = []
    if args.strategy in ['B','C']:
        nlm180_fname = 'dataset/nlm180'
        with open(nlm180_fname + '.txt','r') as f:
            with open(nlm180_fname + '.conllu.txt','r') as fconllu:
                nlm180 = load_data(f,fconllu,stemming=args.stemming,maxlen=101)
    
    ddi2013 = []
    if args.strategy in ['C']:
        ddi2013b_fname = 'dataset/drugbank2013'
        with open(ddi2013b_fname + '.txt','r') as f:
            with open(ddi2013b_fname + '.conllu.txt','r') as fconllu:
                ddi2013 = load_data(f,fconllu,stemming=args.stemming)
        
        ddi2013m_fname = 'dataset/medline2013'
        with open(ddi2013m_fname + '.txt','r') as f:
            with open(ddi2013m_fname + '.conllu.txt','r') as fconllu:
                ddi2013 += load_data(f,fconllu,stemming=args.stemming)
        
        random.shuffle(ddi2013)

    global_data = dataset + nlm180 + ddi2013
    train_buckets = bucketize(dataset)

    if not args.xval_dir:
        print("TESTING AND ANNOTATION MODE")
        test_and_annotate(args, dataset, nlm180, ddi2013, global_data, train_buckets)
    else:
        print("N-FOLD CROSS VALIDATION MODE")
        nfold_xvalidation(args, dataset, nlm180, ddi2013, global_data, train_buckets)

def nfold_xvalidation(args, dataset, nlm180, ddi2013, global_data, train_buckets):
    label_dist, seqlen_mean, seqlen_max = data_stats(global_data)
    print("Dataset size> {}".format(len(global_data)))
    print("Token length> Mean={} Max={}".format(seqlen_mean,seqlen_max))
    print("Label distribution> ")
    for k, v in label_dist.items():
        print('\t',k,v)

    word_vocab = build_vocab(global_data)
    print("Word vocab> Length={} : .., {},..".format(len(word_vocab),', '.join(word_vocab[-100:-50])))
    
    drugnames = sorted(list(train_buckets.keys()))
    n_folds = int(len(drugnames)/args.heldout_count)
    if len(drugnames) % args.heldout_count > 0:
        n_folds += 1

    for k in range(n_folds):
        if len(drugnames) == 0:
            break
        
        print('\n######### WORKING ON FOLD {}/{}\n'.format(k+1,n_folds))

        heldout_drugs = []
        for kk in range(args.heldout_count):
            if len(drugnames) == 0:
                continue
            
            heldout_drugs.append(drugnames.pop())

        exists = 0
        for hd in heldout_drugs:
            dpath = '{}/{}.txt'.format(args.xval_dir,hd.replace(' ','_'))
            if os.path.exists(dpath):
                exists += 1

        if (exists == args.heldout_count or k < args.bucket_min or k > args.bucket_max):
            print('Skipping {}'.format(heldout_drugs))
            continue

        devdrugs = [ k for k, b in sorted(train_buckets.items()) if k not in heldout_drugs ][-2:]
        testset = flatten([ train_buckets[k] for k in heldout_drugs ])

        trainset = [ b for k, b in train_buckets.items() if k not in heldout_drugs + devdrugs ]
        devset = [ b for k, b in train_buckets.items() if k  in devdrugs ]

        devset = flatten(devset)
        trainset = flatten(trainset) 
        
        print('@ Training-22 set',len(trainset))
        print('@ NLM-180 set',len(nlm180))
        print('@ DDI 2013 set',len(ddi2013))    
        print('@ Dev set {} > {}'.format(len(devset),devdrugs))
        print('@ Test set {} > {}'.format(len(testset),heldout_drugs))
        
        # Build the model
        if args.no_embeddings:
            word_embeddings = None
        else:
            word_embeddings = load_embeddings(args.embeddings,word_vocab)
        
        model = new_model(word_vocab, seqlen_max, 
                            word_embeddings=word_embeddings,
                            char_embedding_size=args.char_embedding_size,
                            hidden_size=args.hidden_size,
                            gcn_hidden_size=args.gcn_hidden_size, 
                            gcn_attn_size=args.gcn_attn_size, 
                            gcn_depth=args.gcn_depth)
        
        if args.load_checkpoint:
            print("LOADING CHECKPOINT AT {}".format(args.load_checkpoint))
            model.saver.restore(model.sess, args.load_checkpoint)
        else:
            random.seed(args.devseed)
            train_strategy(args, model, trainset, devset, nlm180, ddi2013)
        
        #testset = [ b for k, b in train_buckets.items() if k in heldout_drugs ]
        eval_test(model,bucketize(testset))
        
        testset_annotated = annotate_test(model, testset)
        save_data(testset_annotated, args.xval_dir)

flatten = lambda l: [item for sublist in l for item in sublist]
def test_and_annotate(args, dataset, nlm180, ddi2013, global_data, train_buckets):
    trainset = [ b for k, b in random.sample(train_buckets.items(),len(train_buckets)) ]

    # core dataset
    devset = flatten(trainset[:6])
    trainset = flatten(trainset[6:])
    
    label_dist, seqlen_mean, seqlen_max = data_stats(global_data)
    print("Trainingset size> {}".format(len(global_data)))
    print("Token length> Mean={} Max={}".format(seqlen_mean,seqlen_max))
    print("Label distribution> ")
    for k, v in label_dist.items():
        print('\t',k,v)

    with open('dataset/test1.txt','r') as f:
        with open('dataset/test1.conllu.txt','r') as fconllu:
            testset1 = load_data(f,fconllu,seqlen_max,stemming=args.stemming)
    
    with open('dataset/test2.txt','r') as f:
        with open('dataset/test2.conllu.txt','r') as fconllu:
            testset2 = load_data(f,fconllu,seqlen_max,stemming=args.stemming)

    global_data += testset1 + testset2

    dataset_pd = pd_examples(global_data)
    dataset_pk = pk_examples(global_data)

    word_vocab = build_vocab(global_data)
    print("Word vocab> Length={} : .., {},.."
        .format(len(word_vocab),', '.join(word_vocab[-100:-50])))

    print('@ Training-22 set',len(trainset))
    print('@ NLM-180 set',len(nlm180))
    print('@ DDI 2013 set',len(ddi2013))    
    print('@ Dev set', len(devset))
    print('@ Test set1 > {} '.format(len(testset1)))
    print('@ Test set2 > {} '.format(len(testset2)))

    # Build the model
    if args.no_embeddings:
        word_embeddings = None
    else:
        word_embeddings = load_embeddings(args.embeddings,word_vocab)
    
    model = new_model(word_vocab, seqlen_max, 
                        word_embeddings=word_embeddings,
                        char_embedding_size=args.char_embedding_size,
                        hidden_size=args.hidden_size,
                        gcn_hidden_size=args.gcn_hidden_size, 
                        gcn_attn_size=args.gcn_attn_size, 
                        gcn_depth=args.gcn_depth)

    if args.load_checkpoint:
        print("LOADING CHECKPOINT AT {}".format(args.load_checkpoint))
        model.saver.restore(model.sess, args.load_checkpoint)
    else:
        random.seed(args.devseed)
        train_strategy(args, model, trainset, devset, nlm180, ddi2013)

    eval_test(model, bucketize(testset1 + testset2))

    if args.annotate:
        for testset, out_dir in [(testset1,'test1-raw'), (testset2, 'test2-raw')]:
            final_dir = args.annotation_prefix + out_dir
            print("Working on {}".format(final_dir))
            test_buckets = bucketize(testset)
            for i, (k, ld_test) in enumerate(test_buckets.items()):
                print('Annotating {} ({}/{})'.format(k,i+1,len(test_buckets)))
                annotated_ld = annotate_test(model, ld_test)
                save_data(annotated_ld, final_dir)

def eval_test(model,test_buckets):
    test_eval = []
    test_eval_pd = []
    test_eval_pk = []
    random.seed(0)
    for i, (k, ld_test) in enumerate(test_buckets.items()):
        testset_feed = compile_examples_ner(model, ld_test)
        if testset_feed:
            test_eval.append(eval_model(model,testset_feed))

            testset_feed_pd = compile_examples_pd(model, ld_test)
            if testset_feed_pd:
                test_eval_pd.append(eval_model_outcome(model,testset_feed_pd,target='pd'))

            testset_feed_pk = compile_examples_pk(model, ld_test)
            if testset_feed_pk:
                test_eval_pk.append(eval_model_outcome(model,testset_feed_pk,target='pk'))
    
    try: 
        print("\nTest Set Evaluation (estimate)> ")
        test_eval = np.mean(np.array(test_eval),axis=0)
        print("    Core:      f1={:.2%}  p={:.2%}  r={:.2%} ".format(*test_eval))
        test_eval_pd = np.mean(np.array(test_eval_pd),axis=0)
        print("    PD:        accuracy={:.2%}".format(*test_eval_pd))
        test_eval_pk = np.mean(np.array(test_eval_pk),axis=0)
        print("    PK:        accuracy={:.2%}".format(*test_eval_pk))
        score = stats.gmean([test_eval[0],test_eval_pd[0],test_eval_pk[0]])
        print("    Overall:   mean={:.2%}".format(score))
    except:
        print("Warning: Unable to fully eval")

def train_strategy(args, model, trainset, devset, nlm180, ddi2013):
    print('STRATEGY: {}'.format(args.strategy))
    
    random.seed(0)
    random.shuffle(ddi2013)
    random.shuffle(nlm180)

    if args.strategy == 'A': # Training22
        model = train_model(model, trainset, devset, args.num_epoch,  early_stopping=args.early_stopping)
    elif args.strategy == 'B': # NLM180 > Training22
        print("Plan B - Phase 1/2: Training on NLM180")
        model = train_model(model, nlm180[300:], nlm180[:300], 100)
        print("Plan B - Phase 2/2: Training on Training22")
        model = train_model(model, trainset, devset, args.num_epoch, early_stopping=args.early_stopping)
    elif args.strategy == 'C': # DDI > NLM180 > Training22
        print("Plan C - Phase 1/3: Training on DDI2013")
        model = train_model(model, ddi2013[300:], ddi2013[:300], 30, tune_only_ner=True)
        print("Plan C - Phase 2/3: Training on NLM180")
        model = train_model(model, nlm180[300:], nlm180[:300], 100)
        print("Plan C - Phase 3/3: Training on Training22")
        model = train_model(model, trainset, devset, args.num_epoch,  early_stopping=args.early_stopping)
    else:
        raise Exception('Unknown strategy: {}'.format(args.strategy))

def bucketize(dataset):
    buckets = {}
    for sent in dataset:
        if sent.drug_name not in buckets:
            buckets[sent.drug_name] = []

        buckets[sent.drug_name].append(sent)
    
    return buckets

def annotate_test(model, testset):
    testset_feed = compile_examples_ner(model, testset)
    y_true, y_pred = model_predict(model,testset_feed)

    testset_annotated = []
    for sent in testset:
        yp = []
        for i in range(len(sent.entity_labels)):
            yp.append(y_pred.pop(0))

        pd_relations = []
        pk_relations = []
        for i in range(len(yp)):
            for j in range(len(yp)):
                if yp[i] in ['B-DYN','U-DYN'] and yp[j] in ['B-EFF','U-EFF']:
                    # NEG is just dummy value
                    pd_relations.append((i,j,'NEG')) 
        
            if yp[i] in ['B-KIN','U-KIN']:
                # C54355 is just dummy value
                pk_relations.append((i,'C54355'))
                
        outsent = sent._replace(entity_labels=yp,pd_relations=pd_relations,pk_relations=pk_relations)
        annotated_example = [ outsent ]

        if len(pd_relations) > 0:
            pd_feed = compile_examples_pd(model,annotated_example)
            _, y_pred_pd = model_predict_outcome(model,pd_feed,target='pd')
            for i, (prep_pos, eff_pos, code) in enumerate(pd_relations):
                if y_pred_pd[i] == 'POS':
                    y_pred_pd[i] = '1'
                else:
                    y_pred_pd[i] = '0'
                pd_relations[i] = (prep_pos+1,eff_pos+1,y_pred_pd[i])

        if len(pk_relations) > 0:
            pk_feed = compile_examples_pk(model,annotated_example)
            _, y_pred_pk = model_predict_outcome(model,pk_feed,target='pk')
            for i, (prep_pos, code) in enumerate(pk_relations):
                pk_relations[i] = (prep_pos+1,y_pred_pk[i])
        
        outsent = sent._replace(entity_labels=yp,pd_relations=pd_relations,pk_relations=pk_relations)
        testset_annotated.append(outsent)
        
    assert(len(y_pred) == 0)
    return testset_annotated

def train_model(model, trainset, devset, num_checkpoints, tune_only_ner=False, early_stopping=1e9):
    # Train the model
    dropout_keep = 0.5
    print("Training pool size: {}".format(len(trainset)))
    #batch_size = int(len(trainset)/300) + 1
    batch_size = 10
    checkpoint_iter = 100
    tmproot = 'tmp'

    devfeed = compile_examples_ner(model, devset)
    trainsample = random.sample(trainset, 200)
    fitfeed = compile_examples_ner(model, trainsample)

    pk_devfeed = compile_examples_pk(model, devset + trainsample)
    pd_devfeed = compile_examples_pd(model, devset + trainsample)

    sess_id = int(time.time())
    best_score = -1
    best_model = None 
    print('Num. Epochs: {}, Batch Size: {}'.format(num_checkpoints,batch_size))
    t_start = time.time()
    fatigue = 0
    for ep in range(1,num_checkpoints+1):
        chpt_loss = []
        
        for i in range(checkpoint_iter):
            minibatch = random.sample(trainset,batch_size)
            batch = compile_examples_ner(model, minibatch, keep_prob=dropout_keep)
            batch_pd = compile_examples_pd(model, minibatch, keep_prob=dropout_keep)
            batch_pk = compile_examples_pk(model, minibatch, keep_prob=dropout_keep)

            if batch_pk and batch_pd: 
                batch.update(batch_pk)
                batch.update(batch_pd)
                _, loss = model.sess.run([model.train_step['core_pd_pk'], model.y_loss['core_pd_pk']], batch)
            elif batch_pd: 
                batch.update(batch_pd)
                _, loss = model.sess.run([model.train_step['core_pd'], model.y_loss['core_pd']], batch)
            elif batch_pk:
                batch.update(batch_pk)
                _, loss = model.sess.run([model.train_step['core_pk'], model.y_loss['core_pk']], batch)
            else:
                _, loss = model.sess.run([model.train_step['core'], model.y_loss['core']], batch)
            
            chpt_loss.append(loss)
            avg_loss = np.mean(chpt_loss)

            print("checkpoint {}> iter {}> {}/{} loss> {:.4f}       "
                .format(ep, i, i*batch_size, checkpoint_iter*batch_size, avg_loss),end='\r')

        b_eval = eval_model(model,fitfeed)
        d_eval = eval_model(model,devfeed)
        if tune_only_ner:
            dp_eval = Eval(0)
            dk_eval = Eval(0)
            score = d_eval.f
        else:
            dp_eval = eval_model_outcome(model,pd_devfeed, target='pd')
            dk_eval = eval_model_outcome(model,pk_devfeed, target='pk')
            score = stats.gmean([d_eval.f,dp_eval.acc,dk_eval.acc])

        if score > best_score:
            best_score = score
            best_model = "{}/model-{}-ep{}.ckpt".format(tmproot,sess_id,ep)
            
            if not os.path.exists(tmproot):
                os.mkdir(tmproot)
            
            model.saver.save(model.sess, best_model)
            marker = '*'
            fatigue = 0
        else:
            marker = ' '
            fatigue += 1
        
        minutes, secs = elapsed(time.time()-t_start)
        print(("checkpoint {}> loss {:.3f} fit> f1={:.2%} dev> " + 
                "f1={:.2%} pd={:.2%} pk={:.2%} mean={:.2%} [{}m{}s] {}")
                    .format(ep, avg_loss, b_eval.f, d_eval.f, dp_eval.acc, dk_eval.acc, 
                        score, minutes, secs, marker))
        
        if fatigue >= early_stopping:
            print("No improvements after {} checkpoints. Stopping training.".format(early_stopping))
            break

    print("    Elapsed training time: {}m{}s".format(*elapsed(time.time()-t_start)))
    print("Restoring best model: {}".format(best_model))
    model.saver.restore(model.sess, best_model)
    
    print("Dev. Set Evaluation> ")
    dev_eval = eval_model(model,devfeed)
    print("    Core:      f1={:.2%}  p={:.2%}  r={:.2%}".format(*dev_eval))

    if not tune_only_ner:
        dev_eval_pd = eval_model_outcome(model,pd_devfeed, target='pd')
        print("    PD:        accuracy={:.2%}".format(*dev_eval_pd))
        dev_eval_pk = eval_model_outcome(model,pk_devfeed, target='pk')
        print("    PK:        accuracy={:.2%}".format(*dev_eval_pk))
        score = stats.gmean([dev_eval.f,dev_eval_pd.acc,dev_eval_pk.acc])
        print("    Overall:   mean={:.2%}".format(score))
    else:
        print('Re-initializing output NER layer')
        model.sess.run(model.target_ner.symbols['W_out'].initializer)
        model.sess.run(model.target_ner.symbols['b_out'].initializer)
        model.sess.run(model.target_ner.symbols['transition_matrix'].initializer)
        
    return model

def elapsed(s):
    s = int(s)
    minutes = int(s / 60)
    seconds = s % 60
    return minutes, seconds

def build_vocab(dataset):
    word_freq = {}
    for sent in dataset:
        for token in sent.tokens:
            if token.form2 not in word_freq:
                word_freq[token.form2] = 1
            else:
                word_freq[token.form2] += 1
    
    word_vocab = ['ZERO','UNK', 'XXXXXXXX', 'GGGGGGGG', '<PRECIPITANT>', '<EFFECT>']
    word_vocab +=  sorted([ w for w, freq in word_freq.items() 
                        if freq >= 1 and w not in word_vocab ])

    return word_vocab

def data_stats(dataset):
    label_freq = {}
    token_lengths = []

    for sent in dataset:
        for label in sent.entity_labels:
            if label in label_freq:
                label_freq[label] += 1
            else:
                label_freq[label] = 1
    
        token_lengths.append(len(sent.tokens))
    
    return label_freq, np.mean(token_lengths), np.max(token_lengths)

def model_predict(model, feed_dict):
    y_true, y_out, transmat, x_length = model.sess.run([model.target_ner.y_true, 
                                model.target_ner.y_out, 
                                model.target_ner.symbols['transition_matrix'],
                                model.target_ner.symbols['x_length']], feed_dict)

    assert(y_true.shape[0] == x_length.shape[0])
    y_true_masked = []
    y_out_masked = []
    for i in range(y_true.shape[0]):
        seqlen = x_length[i]
        viterbi, viterbi_score = tf.contrib.crf.viterbi_decode(y_out[i,:seqlen,:],transmat)
        for j in range(seqlen):
            y_true_masked.append(np.argmax(y_true[i,j,:]))
            y_out_masked.append(viterbi[j])

    labelize = lambda i: model.target_ner.labels[i]
    y_pred = [ labelize(idx) for idx in y_out_masked ]
    y_true = [ labelize(idx) for idx in y_true_masked ]
    return y_true, y_pred

def extract_rpr(seq, typed=False):
    mentions = []
    spans = []
    pos = len(seq)
    tagify = lambda x: x.split('-')[-1]
    
    for tag in reversed(seq):
        pos -= 1
        if tag != 'O':
            spans.append((pos, tagify(tag)))
            if tag.startswith('B') or tag.startswith('U'):
                final_spans = []
                for p, t in sorted(spans):
                    if tagify(t) != tagify(tag):
                        break
                    
                    final_spans.append((p,t))

                mentions.append(final_spans)
                spans = []
        else:
            spans = []
    
    reps = []
    for ment in mentions:
        label = ''.join(sorted(set([ tagify(tag) for _, tag in ment ])))
        pos_start = str(min([ pos for pos, _ in ment ]))
        pos_end = str(max([ pos for pos, _ in ment ]))
        if typed:
            reps.append(':'.join([pos_start, pos_end, label]))
        else:
            reps.append(':'.join([pos_start, pos_end]))

    return reps

def eval_rpr(y_true,y_pred,typed):
    y_true = [extract_rpr(y_true,typed)]
    y_pred = [extract_rpr(y_pred,typed)]
    if y_true == [[]]:
        return 1, 1, 1
    
    m = MultiLabelBinarizer().fit(y_true + y_pred)
    y_true = m.transform(y_true)
    y_pred = m.transform(y_pred)

    f = sklearn.metrics.f1_score(y_true,y_pred,average='micro')
    p = sklearn.metrics.precision_score(y_true, y_pred,average='micro')
    r = sklearn.metrics.recall_score(y_true, y_pred,average='micro')
    return f, p, r

def eval_model(model, feed_dict):
    y_true, y_pred = model_predict(model, feed_dict)
    f, p, r = eval_rpr(y_true,y_pred,typed=True)
    f_partial, p_partial, r_partial = eval_rpr(y_true,y_pred,typed=False)
    return EvalNER(f, p, r, f_partial, p_partial, r_partial)

def eval_model_outcome(model, feed_dict, target = 'pk'):
    y_true, y_pred = model_predict_outcome(model, feed_dict, 
                                    target=target)

    if target == 'pd':
        labels = model.target_pd.labels
    elif target == 'pk':
        labels = model.target_pk.labels
    else:
        raise Exception('what')

    return Eval(sklearn.metrics.accuracy_score(y_true, y_pred))

def model_predict_outcome(model, feed_dict, target = 'pk'):
    if target == 'pd':
        output_node = model.target_pd
    elif target == 'pk':
        output_node = model.target_pk
    else:
        raise Exception('what')
        
    y_true, y_out = model.sess.run([output_node.y_true, 
                            output_node.y_out], feed_dict)

    labelize = lambda i: output_node.labels[i]
    y_pred = [ labelize(idx) for idx in np.argmax(y_out,axis=-1) ]
    y_true = [ labelize(idx) for idx in np.argmax(y_true,axis=-1) ]
    return y_true, y_pred

def pk_examples(dataset):
    examples = []
    for sent in dataset:
        for prep_pos, code in sent.pk_relations:
            examples.append(PK(sent, prep_pos, code))
        
    return examples

def pd_examples(dataset):
    pos_examples = []
    neg_examples = []
    for sent in dataset:
        for prep_pos, eff_pos, code in sent.pd_relations:
            pos_examples.append(PD(sent, prep_pos, eff_pos, code))
        
        preps = []
        effs = []
        for i, x in enumerate(sent.entity_labels):
            if x == 'B-DYN': preps.append(i)
            if x == 'B-EFF': effs.append(i)
        
        for p in preps:
            for e in effs:
                if (p,e,1) not in sent.pd_relations:
                    neg_examples.append(PD(sent,p,e,0))

    return neg_examples + pos_examples

def bind_token(tokens,entity_labels,index,marker):
    bound_tokens = []

    #inside = False
    for i in range(len(entity_labels)):
        if i == index:
            entok = marker
            bound_tokens.append(tokens[i]._replace(form2=marker))
        else:
            bound_tokens.append(tokens[i])

        '''if not inside and i == index:
            inside = True
        elif (entity_labels[i][0] in ['B','U']  
            or entity_labels[i] == 'O'):
            inside = False
        
        #print(i,tokens[i].form2,entity_labels[i],inside)
        if inside:
            if entity_labels[i][-3:] in ['KIN','DYN','EFF']:
                entok = marker
            else:
                inside = False
                entok = tokens[i].form2
                print('------------------------------------')
                for i in range(len(entity_labels)):
                    print(i,tokens[i].form2,entity_labels[i])

                raise Exception('what? {} @ position {}'.format(entity_labels[i],i))

            bound_tokens.append(tokens[i]._replace(form2=entok))
        else:
            bound_tokens.append(tokens[i])'''
    
    return bound_tokens

def encode_graph_matrix(sent,seqlen,dep_path=None):
    n = len(sent.tokens)
    A = np.zeros((seqlen,seqlen))
    for t in sent.tokens:
        if dep_path and (t.id not in dep_path or t.head not in dep_path):
            continue
        
        if t.id < seqlen and t.head < seqlen: 
            A[t.id,t.head] = 1.0
            A[t.head,t.id] = 1.0
    
    A += np.eye(A.shape[0],dtype=np.float32)
    return A 

def dependency_path(tokens,entity_a,entity_b):
    path_a = path_to_root(tokens,entity_a)
    path_b = path_to_root(tokens,entity_b)
    if not path_a or not path_b:
        return None
    
    while len(path_a) > 2 and len(path_b) > 2:
        if path_a[-1] == path_b[-1]:
            path_a.pop()
            path_b.pop()
        else:
            break

    path = path_a + [ x for x in reversed(path_b) if x not in path_a ]
    return path

def path_to_root(tokens,leaf_id):
    try:
        for t in tokens:
            if t.id == leaf_id:
                path = [ t.id ]
                while t.head > 0:
                    t = tokens[t.head]
                    path.append(t.id)

                return path
    except:
        return None

onehot = lambda i, n : np.eye(n)[i,:]

def compile_examples_ner(model, examples, keep_prob=1):
    core_x_length = []
    core_c_input = []
    core_x_input = []
    core_A_matrix = []
    core_y_true = []

    for sent in examples:
        lenx = len(sent.tokens)
        assert(lenx == len(sent.entity_labels))
        
        core_c_input.append(encode_char(sent.tokens, model.seqlen, model.charlen))
        core_x_input.append(encode_words(sent.tokens,model.word_vocab,model.seqlen))
        core_x_length.append(lenx)
        core_A_matrix.append(encode_graph_matrix(sent, model.seqlen))

        lhot = []
        for label in sent.entity_labels:
            lhot.append(onehot(model.target_ner.labels.index(label), 
                            len(model.target_ner.labels)))

        core_y_true.append(lhot + [ np.zeros(len(model.target_ner.labels)) 
                            for i in range(model.seqlen - lenx) ])
    
    feed_dict = {   model.target_ner.symbols['x_input'] : np.array(core_x_input), 
                    model.target_ner.symbols['c_input'] : np.array(core_c_input),
                    model.target_ner.symbols['x_length'] : np.array(core_x_length), 
                    model.target_ner.symbols['A_matrix'] : np.array(core_A_matrix),
                    model.target_ner.y_true : np.array(core_y_true),
                    model.dropout_keep : keep_prob }

    return feed_dict

def compile_examples_pd(model, examples, keep_prob=1):
    pd_x_length = []
    pd_c_input = []
    pd_x_input = []
    pd_A_matrix = []
    pd_x_entity = []
    pd_y_true = []

    for sent in examples:
        lenx = len(sent.tokens)
        assert(lenx == len(sent.entity_labels))
        
        for pd_ex in pd_examples([sent]):
            xencoded = encode_words(pd_ex.sentence.tokens,model.word_vocab,model.seqlen)
            xencoded_char = encode_char(pd_ex.sentence.tokens, model.seqlen, model.charlen)
            depath = dependency_path(pd_ex.sentence.tokens,pd_ex.precipitant,pd_ex.effect)
            A_matrix = encode_graph_matrix(pd_ex.sentence, model.seqlen, dep_path=depath)
            
            bound_tokens = bind_token(pd_ex.sentence.tokens,
                                    pd_ex.sentence.entity_labels,
                                    pd_ex.precipitant,'<PRECIPITANT>')

            bound_tokens = bind_token(bound_tokens,
                                    pd_ex.sentence.entity_labels,
                                    pd_ex.effect,'<EFFECT>')

            xencoded_bound = encode_words(bound_tokens,model.word_vocab,model.seqlen)

            pd_c_input.append(xencoded_char)
            pd_x_input.append(xencoded_bound)
            pd_x_length.append(lenx)
            pd_A_matrix.append(A_matrix)
            pd_x_entity.append([pd_ex.precipitant,pd_ex.effect])

            if pd_ex.code == 1:
                code = 'POS'
            else:
                code = 'NEG'
            
            pd_y_true.append(onehot(model.target_pd.labels.index(code), 
                                    len(model.target_pd.labels)))
    
    if len(pd_y_true) == 0:
        return None

    feed_dict = {   model.target_pd.symbols['x_input'] : np.array(pd_x_input), 
                    model.target_pd.symbols['c_input'] : np.array(pd_c_input),
                    model.target_pd.symbols['x_length'] : np.array(pd_x_length), 
                    model.target_pd.symbols['A_matrix'] : np.array(pd_A_matrix),
                    model.target_pd.symbols['x_entity'] : np.array(pd_x_entity),
                    model.target_pd.y_true : np.array(pd_y_true),
                    model.dropout_keep : keep_prob }

    return feed_dict

def compile_examples_pk(model, examples, keep_prob=1):
    pk_x_length = []
    pk_c_input = []
    pk_x_input = []
    pk_A_matrix = []
    pk_x_entity = []
    pk_y_true = []

    for sent in examples:
        lenx = len(sent.tokens)
        assert(lenx == len(sent.entity_labels))

        for pk_ex in pk_examples([sent]):
            xencoded_char = encode_char(sent.tokens, model.seqlen, model.charlen)

            candidates = []
            for t in pk_ex.sentence.tokens:
                if t.form2 == 'XXXXXXXX':
                    candidates.append(('A',abs(pk_ex.precipitant-t.id),t.id))
                elif t.form2 == 'GGGGGGGG':
                    candidates.append(('B',abs(pk_ex.precipitant-t.id),t.id))
            
            candidates = sorted(candidates)

            if len(candidates) > 0:
                depath = dependency_path(pk_ex.sentence.tokens,pk_ex.precipitant,candidates[0][-1])
            else:
                depath = None
            
            A_matrix = encode_graph_matrix(pk_ex.sentence, model.seqlen, dep_path=depath)

            bound_tokens = bind_token(pk_ex.sentence.tokens,
                                    pk_ex.sentence.entity_labels,
                                    pk_ex.precipitant,'<PRECIPITANT>')
            
            xencoded_bound = encode_words(bound_tokens,model.word_vocab,model.seqlen)

            pk_x_input.append(xencoded_bound)
            pk_c_input.append(xencoded_char)
            pk_x_length.append(lenx)
            pk_A_matrix.append(A_matrix)
            pk_x_entity.append([pk_ex.precipitant,pk_ex.precipitant])

            pk_y_true.append(onehot(model.target_pk.labels.index(pk_ex.code), 
                                    len(model.target_pk.labels)))

    if len(pk_y_true) == 0:
        return None

    feed_dict = {   model.target_pk.symbols['x_input'] : np.array(pk_x_input), 
                    model.target_pk.symbols['c_input'] : np.array(pk_c_input),
                    model.target_pk.symbols['x_length'] : np.array(pk_x_length), 
                    model.target_pk.symbols['A_matrix'] : np.array(pk_A_matrix),
                    model.target_pk.symbols['x_entity'] : np.array(pk_x_entity),
                    model.target_pk.y_true : np.array(pk_y_true),
                    model.dropout_keep : keep_prob }

    return feed_dict

def encode_words(tokens,word_vocab,seqlen):
    tidx = []
    for token in tokens:
        if token.form2 in word_vocab:
            tidx.append(word_vocab.index(token.form2))
        else:
            raise Exception('word not in vocab {}'.format(token.form2))
    
    return tidx + [0] * (seqlen - len(tokens))

def encode_char(tokens, seqlen, charlen):
    tidx = []
    for token in tokens:
        cidx = []
        for c in token.form[:charlen]:
            z = ord(c)
            if z in range(128):
                cidx.append(z)
            else:
                cidx.append(1)

        tidx.append(cidx + [0] * (charlen - len(cidx)))
    
    while len(tidx) < seqlen:
        tidx.append([0] * charlen)

    return tidx

randinit = lambda x: tf.truncated_normal(x, stddev=0.1)
def new_model(word_vocab, seqlen, word_embeddings=None, embedding_size=200, 
                char_embedding_size = 24, hidden_size=50, gcn_hidden_size=100, 
                gcn_attn_size=50, gcn_depth=1):
    
    max_charlen = 64
    
    tf.reset_default_graph()
    
    W_em = tf.Variable(randinit([len(word_vocab), embedding_size]))
    w_input = tf.placeholder(tf.float32, [len(word_vocab), embedding_size],name='embeddings_input')

    dropout_keep = tf.placeholder(tf.float32, None,name='dropout_keep')

    build_rep = tf.make_template("rep", rep_network, word_vocab=word_vocab, 
            W_em=W_em, charlen=max_charlen, char_embedding_size=char_embedding_size, 
            seqlen=seqlen, embedding_size=embedding_size, hidden_size=hidden_size, 
            gcn_attn_size=gcn_attn_size, gcn_hidden_size=gcn_hidden_size, 
            gcn_depth=gcn_depth, dropout_keep=dropout_keep)

    input_width = 2*hidden_size + gcn_hidden_size

    build_cnn = tf.make_template("cnn", cnn_network, input_width=input_width)
    
    ner_target = branch_ner(build_rep, seqlen, max_charlen, 
                            dropout_keep, embedding_size, hidden_size)
    
    pd_target = branch_pd(build_rep, build_cnn, seqlen, max_charlen,
                            dropout_keep, embedding_size, hidden_size)
    
    pk_target = branch_pk(build_rep, build_cnn, seqlen, max_charlen, 
                            dropout_keep, embedding_size, hidden_size)
    
    y_loss = { 'core': ner_target.y_loss, 
                'core_pd': ner_target.y_loss + pd_target.y_loss,
                'core_pk': ner_target.y_loss + pk_target.y_loss,
                'core_pd_pk': ner_target.y_loss + pd_target.y_loss + pk_target.y_loss}

    opt = tf.train.RMSPropOptimizer(learning_rate=0.001)
    train_step = dict([ (k,opt.minimize(v)) for k,v in y_loss.items() ])

    embedding_init = W_em.assign(w_input)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    if word_embeddings is not None:
        sess.run(embedding_init, { w_input: word_embeddings })

    param_count = 0
    for v in tf.trainable_variables():
        param_count += np.prod([ int(dimsize) for dimsize in v.get_shape() ])

    print("Compiled model with {} variables and {} parameters".format(
            len(tf.trainable_variables()),param_count))
    
    saver = tf.train.Saver(max_to_keep=100)
    return NeuralModel(word_vocab, seqlen, max_charlen, sess, saver, dropout_keep, 
                        ner_target, pd_target, pk_target, y_loss, train_step)

def branch_ner(build_representation, seqlen, charlen, dropout_keep, 
                embedding_size, hidden_size, label_embedding_size=8):
    
    hidden_size = 2 * hidden_size
    labels = NER_BILOU
    
    c_input = tf.placeholder(tf.int32, [None, seqlen, charlen], name='c_input_core')
    x_input = tf.placeholder(tf.int32, [None, seqlen], name='x_input_core')
    A_matrix = tf.placeholder(tf.float32, [None, seqlen,seqlen], name='A_matrix_core')
    x_length = tf.placeholder(tf.int32, [None], name='x_length_core')
    y_true = tf.placeholder(tf.float32, [None, seqlen, len(labels)], name='y_true_core')

    xw_rep, bilstm_rep, gcn_rep = build_representation(c_input,x_input,x_length,A_matrix)

    state_size = 2*hidden_size + 2*hidden_size + embedding_size
    output_states = tf.concat([gcn_rep, bilstm_rep, xw_rep], axis=-1)

    rnn_fw2 = tf.nn.rnn_cell.LSTMCell(hidden_size,name='lstm_fw_ner')
    rnn_bw2 = tf.nn.rnn_cell.LSTMCell(hidden_size,name='lstm_bw_ner')
    (fw_out2, bw_out2), _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw2, rnn_bw2, 
                                        output_states, sequence_length=x_length, 
                                        dtype=tf.float32,scope='bilstm2')

    output_states = tf.concat([fw_out2, bw_out2], axis=-1)
    output_states = tf.nn.dropout(output_states, keep_prob=dropout_keep)
    outputs_flat = tf.reshape(output_states,shape=[-1, 2*hidden_size])

    W_out = tf.Variable(randinit([2*hidden_size, len(labels)]))
    b_out = tf.Variable(randinit([len(labels)]))
    y_logits = tf.reshape(outputs_flat @ W_out + b_out, shape=[-1, seqlen, len(labels)])
    y_out = tf.nn.softmax(y_logits)

    # crf loss
    log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(
                                y_logits,tf.argmax(y_true,axis=-1),x_length)
    crf_loss = -tf.reduce_sum(log_likelihood,axis=-1)

    # cross entropy match loss
    y_true_flat = tf.reshape(y_true,shape=[-1,len(labels)])
    y_logit_flat = tf.reshape(y_logits,shape=[-1,len(labels)])
    losses_flat = tf.losses.softmax_cross_entropy(y_true_flat, y_logit_flat,
                                    reduction=tf.losses.Reduction.NONE)

    seq_mask = tf.sequence_mask(x_length, seqlen, dtype=tf.float32) # bsize, seqlen
    masked_losses = tf.reshape(losses_flat,shape=[-1,seqlen]) * seq_mask

    cross_entropy_loss = tf.reduce_sum(masked_losses,axis=-1) 

    y_loss = tf.reduce_sum((crf_loss + cross_entropy_loss) / tf.cast(x_length,dtype=tf.float32))

    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(y_loss)

    symbols = { 'c_input': c_input, 'x_input': x_input, 
                    'A_matrix': A_matrix, 'x_length': x_length, 
                    'transition_matrix': transition_matrix,
                    'W_out': W_out, 'b_out': b_out }

    return Target(labels, y_true, y_out, y_loss, train_step, symbols)

def branch_pd(build_representation, build_cnn, seqlen, charlen, 
                dropout_keep, embedding_size, hidden_size):

    c_input = tf.placeholder(tf.int32, [None, seqlen, charlen], name='c_input_pd')
    x_input = tf.placeholder(tf.int32, [None, seqlen], name='x_input_pd')
    A_matrix = tf.placeholder(tf.float32, [None, seqlen,seqlen], name='A_matrix_pd')
    x_length = tf.placeholder(tf.int32, [None], name='x_length_pd')
    x_entity = tf.placeholder(tf.int32, [None,2], name='x_entity_pd')

    xw_rep, bilstm_rep, gcn_rep = build_representation(c_input,x_input,x_length,A_matrix)
    pd_labels = ['NEG', 'POS']
    
    y_true_pd = tf.placeholder(tf.float32, [None, len(pd_labels)], name='y_true_pd')

    context_rep = tf.concat([bilstm_rep, gcn_rep],axis=2)
    h_sent = build_cnn(context_rep)
    h_entity = tf.batch_gather(context_rep,x_entity)
    h_final = tf.concat([h_sent,h_entity[:,0,:],h_entity[:,1,:]],axis=1)
    
    y_out_pd = tf.layers.dense(tf.nn.dropout(h_final, dropout_keep),len(pd_labels), name='pd_out')
    
    y_loss_pd = tf.losses.softmax_cross_entropy(y_true_pd,y_out_pd)
    train_step_pd = tf.train.AdamOptimizer(learning_rate=0.001).minimize(y_loss_pd)

    symbols = { 'c_input': c_input, 'x_input': x_input, 
                    'A_matrix': A_matrix, 'x_length': x_length,
                    'x_entity': x_entity }
    
    return Target(pd_labels, y_true_pd, y_out_pd, y_loss_pd, train_step_pd, symbols)

def branch_pk(build_representation, build_cnn, seqlen, charlen, 
                dropout_keep, embedding_size, hidden_size):

    c_input = tf.placeholder(tf.int32, [None, seqlen, charlen], name='c_input_pk')
    x_input = tf.placeholder(tf.int32, [None, seqlen], name='x_input_pk')
    A_matrix = tf.placeholder(tf.float32, [None, seqlen,seqlen], name='A_matrix_pk')
    x_length = tf.placeholder(tf.int32, [None], name='x_length_pd')
    x_entity = tf.placeholder(tf.int32, [None,2], name='x_entity_pd')

    xw_rep, bilstm_rep, gcn_rep = build_representation(c_input,x_input,x_length,A_matrix)
    pk_labels = list(pkcode_map.keys())
    
    y_true_pk = tf.placeholder(tf.float32, [None, len(pk_labels)], name='y_true_pk')

    context_rep = tf.concat([bilstm_rep, gcn_rep],axis=2)
    h_sent = build_cnn(context_rep)
    h_entity = tf.batch_gather(context_rep,x_entity)
    h_final = tf.concat([h_sent,h_entity[:,0,:],h_entity[:,1,:]],axis=1)
    
    y_out_pk = tf.layers.dense(tf.nn.dropout(h_final, dropout_keep),len(pk_labels), name='pk_out')

    y_loss_pk = tf.losses.softmax_cross_entropy(y_true_pk,y_out_pk)
    train_step_pk = tf.train.AdamOptimizer(learning_rate=0.001).minimize(y_loss_pk)

    symbols = { 'c_input': c_input, 'x_input': x_input, 
                    'A_matrix': A_matrix, 'x_length': x_length,
                    'x_entity': x_entity }
    
    return Target(pk_labels, y_true_pk, y_out_pk, y_loss_pk, train_step_pk, symbols)

def rep_network(c_input, x_input, x_length, A_matrix, word_vocab, W_em, charlen, 
            char_embedding_size, seqlen, embedding_size, hidden_size, gcn_attn_size, 
            gcn_hidden_size, gcn_depth, dropout_keep):

    C_em = tf.get_variable('C_em',initializer=randinit([128, char_embedding_size]))
    char_input = tf.nn.embedding_lookup(C_em, c_input)
    xc_input = char_cnn(char_input, seqlen, charlen, char_embedding_size)

    xw_input = tf.nn.embedding_lookup(W_em, x_input)

    xwc_input = tf.concat([xw_input,xc_input],axis=-1)
    
    rnn_fw = tf.nn.rnn_cell.LSTMCell(hidden_size,name='lstm_fw_gcn')
    rnn_bw = tf.nn.rnn_cell.LSTMCell(hidden_size,name='lstm_bw_gcn')

    (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(rnn_fw, rnn_bw, 
                                        xwc_input, sequence_length=x_length, 
                                        dtype=tf.float32)
    
    bilstm_out = tf.concat([fw_out,bw_out],axis=-1)
    bilstm_out = tf.nn.dropout(bilstm_out, keep_prob=dropout_keep)

    gcn_out = graph_convolution(bilstm_out,A_matrix,seqlen, gcn_attn_size, 
                                gcn_hidden_size, gcn_depth)
    
    gcn_out = tf.nn.dropout(gcn_out, keep_prob=dropout_keep)
    return xwc_input, bilstm_out, gcn_out

def cnn_network(input_rep, input_width, num_filters=50, filter_sizes=[3,4,5]):
    input_rep = tf.expand_dims(input_rep, axis=-1)

    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        filter_shape = [filter_size, input_width, 1, num_filters]
        W_conv = tf.get_variable("W_{}".format(i), initializer=randinit(filter_shape))
        b_conv = tf.get_variable("b_{}".format(i), initializer=randinit([num_filters]))
        conv = tf.nn.conv2d(input_rep, W_conv, strides=[1, 1, 1, 1], padding="VALID")
        h = tf.nn.relu(tf.nn.bias_add(conv, b_conv))
        if h.shape[2] != 1:
            raise Exception('Input width is wrong')
        
        pooled_outputs.append(tf.reduce_max(h,axis=1))

    num_filters_total = num_filters * len(filter_sizes)
    cnn_pool = tf.reshape(tf.concat(pooled_outputs,axis=-1),shape=[-1,num_filters_total])
    return cnn_pool

def char_cnn(input_rep, seqlen, charlen, char_embedding_size): 
    # input is [bsize, seqlen, charlen, char_embedding_size]
    filter_size = 3
    num_filters = 50
    flat_input = tf.reshape(input_rep,shape=[-1,charlen,char_embedding_size,1])
    filter_shape = [filter_size, char_embedding_size, 1, num_filters]
    W_conv = tf.get_variable('W_char_conv',initializer=randinit(filter_shape))
    b_conv = tf.get_variable('b_char_conv',initializer=randinit([num_filters]))
    conv = tf.nn.conv2d(flat_input, W_conv, strides=[1, 1, 1, 1], padding="VALID")
    h = tf.nn.relu(tf.nn.bias_add(conv, b_conv))
    return tf.reshape(tf.reduce_max(h,axis=1),shape=[-1,seqlen,num_filters])

def graph_convolution(context_emb, A_matrix, seqlen, attn_size, hidden_size, depth):
    gcn_layer = { 0: context_emb }
    
    for l in range(1,depth+1):
        attn_left = tf.make_template('attn_left_L{}'.format(l),tf.layers.dense, units=attn_size, use_bias=False)
        attn_right = tf.make_template('attn_right_L{}'.format(l),tf.layers.dense, units=attn_size, use_bias=False)
        h_bias = tf.get_variable("h_bias-L{}".format(l),initializer=randinit([hidden_size]))
        attn_bias = tf.get_variable("h_dense-L{}".format(l),initializer=randinit([attn_size]))

        rep_in = tf.layers.dense(gcn_layer[l-1],hidden_size,name="dense-L{}".format(l),use_bias=False)
        attn_src = attn_left(rep_in)

        h_rep = [None] * seqlen
        for i in range(seqlen):
            A_mask = tf.expand_dims(A_matrix[:,i,:],axis=-1)

            attn_target = tf.expand_dims(attn_right(rep_in[:,i,:]),axis=1)
            attn_rep = tf.nn.tanh(tf.nn.bias_add(attn_src + attn_target,attn_bias))
            attn_gates = tf.nn.sigmoid(tf.layers.dense(attn_rep,1,use_bias=False))

            rep_out = tf.reduce_sum(A_mask * attn_gates * rep_in,axis=1)

            h_rep[i] = tf.nn.relu(tf.nn.bias_add(rep_out,h_bias))
        
        gcn_layer[l] = tf.stack(h_rep,axis=1)
    
    return gcn_layer[depth]

def load_data(f,fconllu, maxlen=None, stemming=True):
    cast_int = lambda x: tuple([ int(z) if not z.startswith('C') else z for z in x ])

    conllu_map = {}
    porter = PorterStemmer()
    while True:
        header = fconllu.readline()
        if header.strip() == '':
            break

        _, dname, sent_id = header.strip().split('\t')
        tokens = []
        while True:
            line = fconllu.readline()
            if line.strip() == '':
                break
            
            row = line.strip().split('\t')
            row[0] = int(row[0])-1
            row[6] = int(row[6])-1
            if row[1] == 'DRUG' or row[1] == 'DRUGS':
                row[1] = dname
            
            row.append('NULL')
            
            tokens.append(Token(*row))
        
        conllu_map[sent_id] = tokens

    examples = []
    while True:
        header = f.readline()
        if header == "":
            break

        orig_sent = f.readline().strip()
        if orig_sent == "":
            break
        
        dname, sent_id, sect_id, set_id, num_tokens = header.split('\t')

        y = []
        for i in range(int(num_tokens)):
            line = f.readline()
            _, st, ed, label, tok = line.rstrip().split('\t')

            tmp = conllu_map[sent_id][i]._replace(start=st,end=ed)

            if tok not in ['XXXXXXXX','GGGGGGGG'] and stemming:
                tmp = tmp._replace(form2=porter.stem(tok))
            else:
                tmp = tmp._replace(form2=tok)
            
            conllu_map[sent_id][i] = tmp

            y.append(label)
        
        x = conllu_map[sent_id]
        if maxlen is not None:
            x = x[:maxlen]
            y = y[:maxlen]
        
        assert(len(x) == len(y))

        linkages = f.readline().strip()
        
        pd_relations = []
        for z in re.findall(r'D\/([0-9]+):([0-9]+):([01])',linkages):
            x1, x2, x3 = cast_int(z)
            pd_relations.append((x1-1, x2-1, x3))

        pk_relations = [  ]
        for z in re.findall(r'K\/([0-9]+):(C[0-9]+)',linkages):
            x1, x2 = cast_int(z)
            pk_relations.append((x1-1, x2))

        examples.append(Sentence(dname, sent_id, sect_id, set_id, 
                        orig_sent, x, y, pd_relations, pk_relations,))
        f.readline()

    return examples

def save_data(dataset, outdir='outputs'):
    dnames = []
    for sent in dataset:
        dname = sent.drug_name.replace(' ','_') + '_' + sent.set_id
        if dname not in dnames:
            dnames.append(dname)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    filehandler = {}
    for dname in dnames:
        filehandler[dname] = open('{}/{}.txt'.format(outdir,dname),'w')

    for sent in dataset:
        #dname, sent_id, sect_id, set_id, orig_sent, (pd_relations,pk_relations) = attr
        f = filehandler[sent.drug_name.replace(' ','_') + '_' + sent.set_id]
        print(sent.drug_name, sent.sentence_id, sent.sect_id, sent.set_id, len(sent.tokens), sep='\t', file=f)
        print(sent.orig_sent, file=f)
        for i, (token, label) in enumerate(zip(sent.tokens,sent.entity_labels),start=1):
            print(i, token.start, token.end, label, token.form, sep='\t', file=f)
        
        linkpd = ' '.join(['D/' + ':'.join([ str(z) for z in v]) for v in sent.pd_relations])
        linkpk = ' '.join(['K/' + ':'.join([ str(z) for z in v]) for v in sent.pk_relations])
        link = linkpd + linkpk
        if link == '':
            link = 'NULL'
        
        print(link, file=f)
        print('', file=f)
    
    for fh in filehandler.values():
        fh.close()

def load_embeddings(fname,vocab,dim=200):
    from gensim.models import KeyedVectors
    shape = (len(vocab), dim)
    weight_matrix = np.random.uniform(-0.10, 0.10, shape).astype(np.float32)
    w2v = KeyedVectors.load_word2vec_format(fname, binary=True)

    c = 0 
    for i in range(len(vocab)):
        if vocab[i] in w2v.vocab:
            weight_matrix[i,:] = w2v[vocab[i]]
        else:
            c += 1
    
    print("{}/{} pretrained word embeddings loaded".format(len(vocab)-c,len(vocab)))
    return weight_matrix

'''def fix_iob(y):
    y_new = [x for x in y ]
    for i in range(len(y)):
        if y[i] != 'O':
            if i > 0:
                if y[i-1] == 'O':
                    y_new[i] = 'B-' + y[i][-1]
                elif y[i-1][-1] != y[i][-1]:
                    y_new[i] = 'B-' + y[i][-1]
            else:
                if not y[i].startswith('B'):
                    y_new[i] = 'B-' + y[i][-1]

        #print(y[i],'\t',y_new[i])

    return y_new'''

pkcode_map = {'C54355': 'INCREASED DRUG LEVEL',
            'C54602': 'INCREASED DRUG CMAX',
            'C54603': 'INCREASED DRUG HALF LIFE',
            'C54604': 'INCREASED DRUG TMAX',
            'C54605': 'INCREASED DRUG AUC',
            'C54357': 'INCREASED CONCOMITANT DRUG LEVEL',
            'C54610': 'INCREASED CONCOMITANT DRUG CMAX',
            'C54611': 'INCREASED CONCOMITANT DRUG HALF LIFE',
            'C54612': 'INCREASED CONCOMITANT DRUG TMAX',
            'C54613': 'INCREASED CONCOMITANT DRUG AUC',
            'C54356': 'DECREASED DRUG LEVEL',
            'C54606': 'DECREASED DRUG CMAX',
            'C54607': 'DECREASED DRUG HALF LIFE',
            'C54608': 'DECREASED DRUG TMAX',
            'C54609': 'DECREASED DRUG AUC',
            'C54358': 'DECREASED CONCOMITANT DRUG LEVEL',
            'C54615': 'DECREASED CONCOMITANT DRUG CMAX',
            'C54616': 'DECREASED CONCOMITANT DRUG HALF LIFE',
            'C54617': 'DECREASED CONCOMITANT DRUG TMAX',
            'C54614': 'DECREASED CONCOMITANT DRUG AUC' }

if __name__ == '__main__':
    main(parser.parse_args())
    