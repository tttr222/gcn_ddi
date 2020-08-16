import os, sys, time, random, re, json, collections, argparse, string, unicodedata
import xml.etree.ElementTree as ET
from xml.dom import minidom
from multiprocessing.dummy import Pool as ThreadPool 

parser = argparse.ArgumentParser(description='Process drug labels')
parser.add_argument('directory', type=str,
                    help='directory to build ensembles')
parser.add_argument('--second', type=str, default=None,
                    help='second directory to build ensembles')
parser.add_argument('--count', type=int, default=4,
                    help='ensemble count')
parser.add_argument('--size', type=int, default=4,
                    help='ensemble size')

def main(args):
    sessions = find_sessions(args.directory)
    
    if args.second:
        sessions_second = find_sessions(args.second)

    ensemble_size = args.size
    ensemble_count = args.count

    print('Found {} sessions'.format(len(sessions)))
    print('Ensemble> size: {}, count: {}'.format(ensemble_size,ensemble_count))
    random.seed(0)

    todo = []
    for i in range(ensemble_count):
        if args.second:
            session_sample = (random.sample(sessions,int(ensemble_size/2)) 
                     + random.sample(sessions_second,int(ensemble_size/2)))
            outdir_path = '{}.{}'.format(args.directory,'.'.join(args.second.split('.')[1:]))
            output_dir = os.path.join(outdir_path,'ENS{:02d}'.format(i))
        else:
            session_sample = random.sample(sessions,ensemble_size)
            output_dir = os.path.join(args.directory,'ENS{:02d}'.format(i))

        os.makedirs(output_dir,exist_ok=True)
        todo.append((session_sample,output_dir))
    
    for x in todo:
        work_thread(x, args)

def work_thread(in_object, args):
    session_sample, output_dir = in_object
    create_ensemble(session_sample, output_dir, 'test1-pred')
    create_ensemble(session_sample, output_dir, 'test2-pred')
    
    if 'bilstm' in args.directory:
        dataset_dir = 'dataset_old'
    else:
        dataset_dir = 'dataset'
    
    os.system('python -u tacEval_relaxed.py -1 -2 {0}/test1 {1}/test1-pred > {1}/result.txt'.format(dataset_dir,output_dir))
    os.system('python -u tacEval_relaxed.py -1 -2 {0}/test2 {1}/test2-pred >> {1}/result.txt'.format(dataset_dir,output_dir))

def find_sessions(dirpath):
    sessions = []
    for x in os.listdir(dirpath):
        if os.path.exists(os.path.join(dirpath,x,'test1-pred')) and x.startswith('SESS'):
            sessions.append(os.path.join(dirpath,x))
    
    return sessions

def create_ensemble(sessions, output_dir, target_set):
    mentions = {}
    interactions = {}
    print(output_dir.split('\\')[-1],target_set,end=': ')
    print('Counting votes', end='.. ')
    for sess in sessions:
        predictions = os.path.join(sess,target_set)
        for fname in os.listdir(predictions):
            if not fname.endswith('.xml'):
                continue

            fpath = os.path.join(predictions,fname)
            mentions_dl, interactions_dl = load_annotations(fpath)
            for k in mentions_dl.keys():
                mention_dict = dict([ sign_mention(x) 
                            for x in mentions_dl[k] ])
                
                if k in mentions:
                    mentions[k] += list(mention_dict.values())
                else:
                    mentions[k] = list(mention_dict.values())

                local_intrs = []
                for x in interactions_dl[k]:
                    local_intrs += sign_intr(x,mention_dict)
                
                if k in interactions:
                    interactions[k] += list(local_intrs)
                else:
                    interactions[k] = list(local_intrs)
    
    print('Ensembling',end='.. ')
    mentions, interactions = voting_ensemble(mentions,interactions)

    print('Saving results',end='.. ')
    for sess in sessions:
        predictions = os.path.join(sess,target_set)
        for fname in os.listdir(predictions):
            if not fname.endswith('.xml'):
                continue

            fpath = os.path.join(predictions,fname)
            outdir = os.path.join(output_dir,target_set)
            save_annotations(fpath, outdir, mentions, interactions)
    print('Done.')

def voting_ensemble(mentions, interactions):
    midc = 0
    kidc = 0
    prev = None

    for k in mentions:
        if k.split('#')[0] != prev:
            midc = 0
            kidc = 0
        
        prev = k.split('#')[0]

        #print(k)
        mentions_candidate = best_mentions(mentions[k])
        mentions[k] = []
        trigger_list = []
        mention_ref = {}

        for mtype, span, offset in mentions_candidate:
            assert(mtype is not None)
            assert(span is not None)
            assert(offset is not None)
            midc += 1
            node = ET.Element('Mention', 
                            attrib={'id': 'M{}'.format(midc),
                                    'str': span, 
                                    'span': offset,
                                    'type': mtype,
                                    'code': 'NO MAP'})
            mentions[k].append(node)
            mention_ref[(mtype, span, offset)] = 'M{}'.format(midc)
            if mtype == 'Trigger':
                trigger_list.append('M{}'.format(midc))
            
        intr_candidates = best_interactions(interactions[k],mentions_candidate)
        
        pd_group = {}
        for itype, precipitant, effect in intr_candidates:
            if itype == 'Pharmacodynamic interaction':
                if precipitant in pd_group:
                    pd_group[precipitant].append(effect)
                else:
                    pd_group[precipitant] = [effect]

        pd_found = []
        interactions[k] = []
        for itype, precipitant, effect in intr_candidates:
            trigger_mid = mention_ref[precipitant]
            if len(trigger_list) > 1:
                trigger_mid = trigger_list.pop(0)
            elif len(trigger_list) == 1:
                trigger_mid = trigger_list[0]
            
            kidc += 1

            intr_attrib= {'id': 'I{}'.format(kidc),
                            'type': itype,
                            'trigger': trigger_mid, 
                            'precipitant': mention_ref[precipitant]}
            if itype == 'Pharmacodynamic interaction':
                if precipitant in pd_found:
                    continue

                eref = [ mention_ref[e] for e in pd_group[precipitant] ]
                intr_attrib['effect'] = ';'.join(eref)

                pd_found.append(precipitant)
            elif itype == 'Pharmacokinetic interaction':
                intr_attrib['effect'] = effect

            node = ET.Element('Interaction', intr_attrib)
            interactions[k].append(node)
        
    return mentions, interactions

def best_interactions(interactions,best_mentions):
    intr_freq = {}
    for sig in interactions:
        if sig in intr_freq:
            intr_freq[sig] += 1
        else:
            intr_freq[sig] = 1
    
    occupied = {}
    candidates = []
    for k, v in sorted(intr_freq.items(),
            key=lambda x: x[1],reverse=True):
        
        linked = True
        for i in range(1,3):
            if k[i] not in best_mentions and isinstance(k[i],tuple):
                #print(k[i],'not found')
                linked = False

        noconflict = False
        if k[1] not in occupied or occupied[k[1]] == k[0]:
            noconflict = True

        if v > 2 and linked and noconflict:
            #print('INTR >> ',k[1],k[2],v, noconflict, linked)
            candidates.append(k)
            occupied[k[1]] = k[0]
        else:
            pass
            #print(k[1],k[2],v, noconflict, linked)
    
    return candidates

def best_mentions(mentions):
    mention_freq = {}
    for sig in mentions:
        if sig in mention_freq:
            mention_freq[sig] += 1
        else:
            mention_freq[sig] = 1
    
    candidates = []
    for k, v in sorted(mention_freq.items(),
            key=lambda x: (x[1] + len(x[0][1])/100),reverse=True):

        add = True
        for m in candidates:
            if overlap(m[-1],k[-1]):
                add = False

        if add and v > 2:
            candidates.append(k)
            #print('PRCP >>', k,v + len(k[1])/100,)
        else:
            pass
            #print(k,v + len(k[1])/100)
    
    return candidates

def overlap(offsets_a,offsets_b):
    range_a = []
    for a, b in parse_offsets(offsets_a):
        range_a += range(a,b)
    
    range_b = []
    for a, b in parse_offsets(offsets_b):
        range_b += range(a,b)
    
    return bool(set(range_a) & set(range_b))

def parse_offsets(raw_offsets):
    offsets = []
    for segment in raw_offsets.split(';'):
        x, y = segment.split(' ')
        offsets.append((int(x),int(x)+int(y)))
        
    return offsets

def sign_mention(node):
    return node.get('id'), (node.get('type'), node.get('str'), node.get('span'))

def sign_intr(node, mentions):
    itype = node.get('type')
    precipitant = mentions[node.get('precipitant')]
    
    effect = [ node.get('effect') ]
    if itype == 'Pharmacodynamic interaction':
        effect = [ mentions[x] for x in node.get('effect').split(';') if x ]

    return [ (itype, precipitant, e) for e in effect ]

def load_annotations(fpath):
    annotations = []
    base_tree = ET.parse(fpath)
    base_root = base_tree.getroot()
    sent_seen = []

    labeldrug = base_root.get('drug')
    setid = base_root.get('setid')

    mentions = {}
    interactions = {}
    for s in base_root.findall('Sentences/Sentence'):
        sent_id = "{}#{}".format(labeldrug.replace(' ','_'), s.get('id'))
        if sent_id not in sent_seen:
            sent_seen.append(sent_id)
        else:
            continue
        
        mentions[sent_id] = s.findall('Mention')
        interactions[sent_id] = s.findall('Interaction')

    return mentions, interactions

def save_annotations(base_fpath, output_dir, mentions, interactions):
    annotations = []
    base_tree = ET.parse(base_fpath)
    base_root = base_tree.getroot()
    sent_seen = []

    labeldrug = base_root.get('drug')
    setid = base_root.get('setid')

    for s in base_root.findall('Sentences/Sentence'):
        sent_id = "{}#{}".format(labeldrug.replace(' ','_'), s.get('id'))
        if sent_id not in sent_seen:
            sent_seen.append(sent_id)
        else:
            continue
        
        for child in s[::-1]:
            if child.tag != 'SentenceText':
                s.remove(child)

        for node in mentions[sent_id]:
            s.append(node)
        
        for node in interactions[sent_id]:
            s.append(node)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir,exist_ok=True)
    
    fname = "{}_{}.xml".format(labeldrug.replace(' ','_'), setid)
    outfile = os.path.join(output_dir,fname) 
    with open(outfile,'w',encoding='utf8') as f:
        print(prettify(base_root), file=f)

def prettify(elem):
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

if __name__ == '__main__':
    main(parser.parse_args())