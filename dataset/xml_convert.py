import os, sys, time, random, re, json, collections, argparse, string, unicodedata
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

parser = argparse.ArgumentParser(description='Process drug labels')
parser.add_argument('--input-dir', type=str, default='tac_training',
                    help='path to directory with drug label XML files')
parser.add_argument('--coord', default=False, action='store_true',
                    help='combine coordinated mentions')

def main(args):
    output_file = open("{}.txt".format(args.input_dir),'w')
    conllu_file = open("{}.conllu.txt".format(args.input_dir),'w')

    drug_generics = {}
    with open('drug_alias.txt','r',encoding='utf-8') as f:
        for l in f:
            drug_key, drug_aliases, drug_classes = l.strip().split('|')
            drug_aliases = [ x for x in drug_aliases.split(',') 
                             if x and x != drug_key ]
            drug_classes = [ x for x in drug_classes.split(',') 
                             if x and x not in drug_aliases ]
            drug_generics[drug_key] = (drug_aliases, drug_classes)

    if os.path.exists(args.input_dir):
        basenames = []
        for fname in sorted(os.listdir(args.input_dir)):
            if not fname.endswith('.xml'):
                continue

            fpath = os.path.join(args.input_dir,fname)
            if args.input_dir == 'nlm180_old':
                parse_nlm180(fpath, output_file, conllu_file, 
                             args.coord, drug_generics)
            else:
                parse_xmldata(fpath, output_file, conllu_file, 
                              args.coord, drug_generics)
    else:
        print('what')
        exit()

def mask_druglabel(sent, drug_tokens, drug_class=False):
    drug_tokens = drug_tokens.split(' ')
    if drug_class:
        ck = 'G'
    else:
        ck = 'X'
    
    if len(drug_tokens) == 1:
        match = re.findall(drug_tokens[0], sent, re.IGNORECASE)
        if match:
            for x in match:
                sent = sent.replace(x,ck * len(x))
    else:
        pattern = '.{1,3}'.join(drug_tokens)
        base_pattern = r'{}'.format(pattern)

        match = re.findall(base_pattern, sent, re.IGNORECASE)
        if match:
            for x in match:
                sent = sent.replace(x,ck * len(x))
    
    return sent

def parse_xmldata(fpath, fout, fconllu, coordination, drug_generics={}):
    base_tree = ET.parse(fpath)
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
        
        #if not (labeldrug == 'Aldactone' and s.get('id') == '14'):
        #    continue
        #if not (labeldrug == 'ADCIRCA' and s.get('id') == '584'):
        #    continue
        
        orig_sent = fixchar(s.find('SentenceText').text)

        sent = mask_druglabel(orig_sent, labeldrug)

        d_aliases, d_classes = drug_generics[labeldrug.lower()]
        for alias in d_aliases:
            sent = mask_druglabel(sent, alias)
        
        for dc in d_classes:
            sent = mask_druglabel(sent, dc, drug_class=True)

        mentions = dict()
        for m in s.findall('Mention'):
            segments = m.get('span').split(';')
            num_segs = len(segments)
            
            seglist = []
            for seg in segments:
                mstart, mlen = seg.split()
                mend = int(mstart) + int(mlen) - 1
                seglist.append((int(mstart), mend))
            
            mentions[m.get('id')] = parse_mention(seglist,m.get('type'),sent)

        if coordination:
            groupings = {}
            for k, v in mentions.items():
                key = str((v[1],v[2]))
                if key not in groupings:
                    groupings[key] = []
                
                groupings[key].append(v)

            for k, v in mentions.items():
                if v[2] != 'Precipitant':
                    continue
                
                key = str((v[1],v[2]))
                reach = v[1]
                for w in groupings[key]:
                    if w[0] < reach:
                        reach = w[0]

                coord_seg = orig_sent[reach:v[1]+1]
                headword = match_headwords(coord_seg)
                if headword:
                    mentions[k] = (reach,v[1],v[2],1)
                    print('>>>', orig_sent[reach:v[1]+1], '[',headword,']')

        linkage = []
        interactions = []
        spans = []
        for r in s.findall('Interaction'):
            pid = r.get('precipitant')
            tid = r.get('trigger')
            eid = r.get('effect')
            if pid not in mentions:
                continue

            if r.get('type') == 'Pharmacodynamic interaction':
                eid_list = eid.split(';')

                if mentions[pid][3] == 1: # num_seg == 1?
                    interactions.append(('DYN', pid, mentions[pid]))
                    for eid in eid_list:
                        linkage.append((pid,eid))

                for eid in eid_list:
                    _, _, _, span_segs = mentions[eid]
                    if span_segs == 1:
                        spans.append(('EFF', eid, mentions[eid]))
                        
            elif r.get('type')  == 'Pharmacokinetic interaction':
                if mentions[pid][3] == 1: # num_seg == 1?
                    interactions.append(('KIN', pid, mentions[pid]))
                    linkage.append((pid,eid))

                tid = tid.split(';')[0] # only one
                if mentions[tid][3] == 1: # num_seg == 1?
                    spans.append(('TRI', tid, mentions[tid]))
                
            elif r.get('type')  == 'Unspecified interaction':
                if mentions[pid][3] == 1: # num_seg == 1?
                    interactions.append(('UNK', pid, mentions[pid]))

                tid = tid.split(';')[0]
                _, _, _, span_segs = mentions[tid]
                if span_segs == 1:
                    spans.append(('TRI', tid, mentions[tid]))
            else:
                raise Exception('Undefined interaction type')

        tokens = using_split2(sent)
        if len(tokens) == 0:
            continue

        align_ctype, align_cid = align_mentions(interactions + spans, tokens)
        joint_tags = apply_bilou(align_ctype,align_cid)
        pd_outcomes, pk_outcomes = align_linkage(linkage,align_ctype,align_cid)
        
        print(labeldrug, s.get('id'), s.get('section'), 
                setid, len(tokens), sep='\t', file=fout)
        print(orig_sent, file=fout)
        if fconllu is not None:
            print('#', labeldrug, s.get('id'), sep='\t', file=fconllu)
        
        for i, (x, xstart, xend) in enumerate(tokens, start=1):
            nerlabs = []

            if len(x.lstrip('*')) > 0:
                x = x.lstrip('*')

            if set(x) == set('X'):
                x = 'XXXXXXXX'

            if set(x) == set('G'):
                x = 'GGGGGGGG'

            print(i, xstart, xend, joint_tags[i-1], x, sep='\t', file=fout)

            if fconllu is not None:
                if x == 'XXXXXXXX':
                    x = 'DRUG'
                elif x == 'GGGGGGGG':
                    x = 'DRUGS'

                print(i, x, *('_' * 8), sep='\t', file=fconllu)
        
        linkpd = ' '.join(['D/' + ':'.join(v) for v in pd_outcomes])
        linkpk = ' '.join(['K/' + ':'.join(v) for v in pk_outcomes])
        link = linkpd + linkpk
        if link == '':
            link = 'NULL'
        print(link, file=fout)
        print('', file=fout)
        if fconllu is not None:
            print('', file=fconllu)

    print("Processed {} sentences from {}".format(len(sent_seen), fpath), file=sys.stderr)

def match_headwords(coord_seg):
    headwords = ['agents','drugs','inducers','inhibitors',
                 'substances','antibiotics','substrate']
    
    for c in string.punctuation:
        if c != '-' and c in coord_seg:
            return None

    if (coord_seg.split(' ').count('and') + coord_seg.split(' ').count('or')) != 1:
        return None

    for h in headwords:
        if coord_seg.endswith(h):
            return h
    
    return None

def parse_mention(segs,mtype,sent):
    if mtype in ['Trigger','SPAN'] and len(segs) == 2:
        (xs1, xe1), (xs2, xe2) = segs
        left = sent[xs1:xe1+1]
        middle = sent[xe1+2:xs2-1]
        right = sent[xs2:xe2+1]
        join = True
        for t in middle.split(' '):
            if t not in stoplist and 'XXX' not in t:
                join = False
                
        if join:
            segs = [(xs1,xe2)]

    return (segs[0][0], segs[-1][1], mtype, len(segs))

def align_mentions(interactions,tokens):
    tags = []
    for i, (x, xstart, xend) in enumerate(tokens, start=1):
        match = []
        for ptype, pid, concept in interactions:
            (ystart, yend, mtype, _) = concept
            if xstart >= ystart and xend <= yend and ptype not in match:
                match.append((ptype,pid))

        match = list(set(match))
        if len(match) > 0:
            if len(tags) > 0 and tags[-1] in match:  
                # prioritize completing the span
                tags.append(tags[-1])
            else:
                # otherwise pick prioritize PD interactions
                # arbitrarily prioritize lower IDs
                ranked = sorted(match)
                tags.append(ranked[0])
        else:
            if i > 1 and tokens[i-1][0] == 'of' and x == 'XXXXXXXX' and tags[-1] == None:
                tags.pop(-1)
                tags.append(tags[-1])
                tags.append(tags[-1])
            else:
                tags.append((None,None))
    
    return zip(*tags)

def align_linkage(linkage, ctype, cid):
    #print(linkage)

    pd_drugs = []
    pd_effects = []
    pd_outcomes = []
    pk_outcomes = []

    mid2pos = {}
    for i in range(len(ctype)):
        if ctype[i] is None:
            continue
        
        if cid[i] not in mid2pos:
            mid2pos[cid[i]] = str(i+1)
        
        if ctype[i] == 'DYN':
            pd_drugs.append(cid[i])
        elif ctype[i] == 'EFF':
            pd_effects.append(cid[i])
        elif ctype[i] == 'KIN':
            lookup = dict(linkage)
            if cid[i] in lookup:
                edge = (mid2pos[cid[i]],lookup[cid[i]])
                pk_outcomes.append(edge)
    
    for pd_drug in pd_drugs:
        for pd_effect in pd_effects:
            edge = (mid2pos[pd_drug],mid2pos[pd_effect])
            
            if (pd_drug,pd_effect) in linkage:
                pd_outcomes.append((*edge,'1'))
            else:
                pd_outcomes.append((*edge,'0'))
    
    return sorted(set(pd_outcomes)), sorted(set(pk_outcomes))

def parse_nlm180(fpath, fout, fconllu, coordination, drug_generics={}):
    base_tree = ET.parse(fpath)
    base_root = base_tree.getroot()
    mtypes = ['Precipitant','Trigger','SpecificInteraction']
    sent_seen = []

    labeldrug = base_root.find('labelDrug').get('name')
    if labeldrug == 'Missing':
        print("Skipping missing drug label @ {}".format(fpath))
        return

    #generic_str = base_root.find('generic').get('name')
    #aliases = re.findall(r'[a-zA-z][a-zA-z\-\s]+[a-zA-z]', 
    #                        generic_str.replace('and', ','))
    #print('\t'.join([labeldrug.lower()] + [ a.lower() for a in aliases ]))
    
    pk_labels = pd.read_csv('pk_bootstrap/pk.nlm180.tsv',sep='\t',header=None)
    pk_lookup = {}
    for i, d in pk_labels.iterrows():
        pk_lookup[(d[0],d[1])] = d[3]

    precipitant_types = ['Biomedical_Entity','Substance','Drug','Drug_Class']
    #print(fpath)
    for s in base_root.findall('sentence'):
        sent_id = "{}#{}".format(labeldrug.replace(' ','_'), s.get('id'))
        if sent_id not in sent_seen:
            sent_seen.append(sent_id)
        else:
            continue

        orig_sent = fixchar(s.get('text'))
        sent = mask_druglabel(orig_sent, labeldrug)

        aliases, d_classes = drug_generics[labeldrug.lower()]
        for alias in aliases:
            sent = mask_druglabel(sent, alias)
        
        for dc in d_classes:
            sent = mask_druglabel(sent, dc, drug_class=True)
        
        sent = sent.replace('<main>','@main ').replace('<item>','@item ')

        mentions = dict()
        for m in s.findall('entity'):
            segments = m.get('charOffset').split('|')
            
            seglist = []
            for seg in segments:
                mstart, mend = seg.split(':')
                seglist.append((int(mstart), int(mend)-1))
                
            mentions[m.get('id')] = parse_mention(seglist,m.get('type'),sent)
        
        if coordination:
            groupings = {}
            for k, v in mentions.items():
                key = str((v[1],v[2]))
                if key not in groupings:
                    groupings[key] = []
                
                groupings[key].append(v)

            for k, v in mentions.items():
                if v[2] not in precipitant_types:
                    continue
                
                key = str((v[1],v[2]))
                reach = v[1]
                for w in groupings[key]:
                    if w[0] < reach:
                        reach = w[0]

                coord_seg = orig_sent[reach:v[1]+1]
                headword = match_headwords(coord_seg)
                if headword:
                    mentions[k] = (reach,v[1],v[2],1)
                    print('>>>', orig_sent[reach:v[1]+1], '[',headword,']')

        linkage = []
        interactions = []
        spans = []
        for r in s.findall('drugInteraction'):
            int_node = r.find('interaction')
            #print(r.get('id'))

            ents = {}
            for obj in int_node.find('relations').findall('relation'):
                if obj.find('entity') == None:
                    continue
                
                if obj.get('type') not in ents:
                    ents[obj.get('type')] = []
                
                ents[obj.get('type')].append(obj.find('entity').get('id'))
            
            participants = []
            if 'hasPrecipitant' in ents:
                participants += ents['hasPrecipitant']
            
            if 'hasObject' in ents:
                participants += ents['hasObject']

            for pid in participants:
                if pid not in mentions:
                    continue
                
                startp, endp, mtype, ent_segs = mentions[pid]

                # Ignore cases where GGG serves as a stand in for XXX
                if ('XXX' not in sent and 'GGG' in sent[startp:endp] and mtype in precipitant_types):
                    continue

                # Ignore cases where label drug itself is a precipitant
                if ('XXX' in sent[startp:endp] and mtype in precipitant_types):
                    continue

                tid = int_node.get('trigger')
                _, _, int_type, int_segs = mentions[tid]
                
                if int_type == 'Specific_Interaction':
                    if ent_segs == 1:
                        interactions.append(('DYN', pid, mentions[pid]))
                        linkage.append((pid,tid))

                    if int_segs == 1:
                        spans.append(('EFF', tid, mentions[tid]))

                elif int_type in ['Decrease_Interaction','Increase_Interaction']:
                    if ent_segs == 1:
                        interactions.append(('KIN', pid, mentions[pid]))
                        key = (s.get('id'), pid)
                        if key in pk_lookup:
                            linkage.append((pid,pk_lookup[key]))
                    
                    if int_segs == 1:
                        spans.append(('TRI', tid, mentions[tid]))

                elif int_type == 'Caution_Interaction':
                    if ent_segs == 1:
                        interactions.append(('UNK', pid, mentions[pid]))
                    
                    if int_segs == 1:
                        spans.append(('TRI', tid, mentions[tid]))

                else:
                    print('Unknown type', m.get('type'))
                    continue

        tokens = using_split2(sent)

        align_ctype, align_cid = align_mentions(interactions + spans, tokens)
        joint_tags = apply_bilou(align_ctype,align_cid)
        pd_outcomes, pk_outcomes = align_linkage(linkage,align_ctype, align_cid)

        #if len(pd_outcomes + pk_outcomes) > 0 and len([w for w, s, e in tokens if 'XX' in w ]) == 0:
        #    print(labeldrug, [w for w, s, e in tokens ])
            
        print(labeldrug, s.get('id'), 'NO_SECTION', 
                'NO_SETID', len(tokens), sep='\t', file=fout)
        print(orig_sent, file=fout)
        if fconllu is not None:
            print('#', labeldrug, s.get('id'), sep='\t', file=fconllu)
        
        for i, (x, xstart, xend) in enumerate(tokens, start=1):
            nerlabs = []

            if len(x.lstrip('*')) > 0:
                x = x.lstrip('*')

            if set(x) == set('X'):
                x = 'XXXXXXXX'
            
            if set(x) == set('G'):
                x = 'GGGGGGGG'

            print(i, xstart, xend, joint_tags[i-1],x, sep='\t', file=fout)

            if fconllu is not None:
                if x == 'XXXXXXXX':
                    x = 'DRUG'
                elif x == 'GGGGGGGG':
                    x = 'DRUGS'

                print(i, x, *('_' * 8), sep='\t', file=fconllu)

        linkpd = ' '.join(['D/' + ':'.join(v) for v in pd_outcomes])
        linkpk = ' '.join(['K/' + ':'.join(v) for v in pk_outcomes])
        link = linkpd + linkpk
        if link == '':
            link = 'NULL'
        print(link, file=fout)
        print("", file=fout)
        if fconllu is not None:
            print('', file=fconllu)

    print("Processed {} sentences from {}".format(len(sent_seen), fpath), file=sys.stderr)

def apply_bilou(ctype,cid):
    assert(len(ctype) == len(cid))
    tags = ['O'] * len(cid)

    for i in range(len(tags)):
        if cid[i] == None:
            continue
        
        tags[i] = '{}-{}'.format('I',ctype[i])
    
    for i in range(1,len(tags)):
        if ctype[i] is not None and cid[i] != cid[i-1]:
            tags[i] = '{}-{}'.format('B',ctype[i])
    
    for i in range(0,len(tags)-1):
        if ctype[i] is not None and cid[i] != cid[i+1]:
            tags[i] = '{}-{}'.format('L',ctype[i])
    
    for i in range(0,len(tags)-1):
        if ctype[i] is not None and cid[i] != cid[i+1] and cid[i] != cid[i-1]:
            tags[i] = '{}-{}'.format('U',ctype[i])
    
    if ctype[0] is not None:
        if cid[0] != cid[1]:
            tags[0] = '{}-{}'.format('U',ctype[0])
        else:
            tags[0] = '{}-{}'.format('B',ctype[0])
    
    if ctype[-1] is not None:
        if cid[-1] != cid[-2]:
            tags[-1] = '{}-{}'.format('U',ctype[-1])
        else:
            tags[-1] = '{}-{}'.format('L',ctype[-1])
            
    return tags

# from stackexchange; user: aquavitae
def using_split2(line, _len=len):
    words = tokenize(line)
    index = line.index
    offsets = []
    append = offsets.append
    running_offset = 0
    for word in words:
        word_offset = index(word, running_offset)
        word_len = _len(word)
        running_offset = word_offset + word_len
        append((word, word_offset, running_offset - 1))
    return offsets

def fixchar(x):
    #for c in x:
    #    if c not in string.printable:
    #        if c == c.translate(literation):
    #            print(c,ord(c),c.translate(literation))
    new_x = ''.join([ z.translate(literation) for z in x ])
    assert(len(x) == len(new_x))
    return new_x

def tokenize(x):
    tokens_raw1 = x.split(' ')

    tokens_raw2 = []
    for i, t in enumerate(tokens_raw1):
        if '/' in t:
            for e in t.split('/'):
                tokens_raw2 += [e,'/']
            tokens_raw2.pop(-1)
        else:
            tokens_raw2.append(t)

    # split hyphens
    tokens_raw3 = []
    for i, t in enumerate(tokens_raw2):
        tokens_raw3 += [ e for e in t.split('-') if e ]
    
    tokens = []
    for t in tokens_raw3:
        match = re.search('([^X]*)(XXX+)([^X]*)',t)
        if match:
            left, middle, right = match.groups()

            if len(left) > 0:
                tokens.append(left)
            tokens.append(middle)
            if len(right) > 0:
                tokens.append(right)
        else:
            tokens.append(t)

    final = []
    for t in tokens: # fix the broken tokenization  
        fixed = False 
        if len(t) > 1:
            for symb in [',',':']:
                if t.endswith(symb):
                    t_fin = t[:-1].strip('.;][()')
                    if len(t_fin) == 0:
                        continue

                    final.append(t_fin)
                    final.append(symb)
                    fixed = True
                    break
            
        if not fixed:
            t_fin = t.strip('.;][()')
            if len(t_fin) == 0:
                continue
            
            final.append(t_fin)
                
    return final

stoplist = list(string.punctuation)

literation = { 174: ord('*'), # registered
            175: ord('*'), # macron
            176: ord('*'),  # circle dot
            177: ord('*'), # plus-minus
            178: ord('*'), # ^2
            180: ord('*'), # acute accent
            181: ord('u'), # micro
            183: ord('*'), # dot
            186: ord('*'), # ord indicator
            189: ord('%'), # half
            195: ord('A'), # accent A
            215: ord('*'), # times
            223: ord('*'), # tilde
            233: ord('e'), # accent e
            239: ord('i'), # accent i
            244: ord('o'), # accent o
            916: ord('D'), # delta
            945: ord('A'), # alpha
            946: ord('B'), # beta
            947: ord('G'), # gamma
            956: ord('M'),  # mu
            964: ord('T'), # tau
            8195: ord(' '), # space
            8209: ord('-'),  # non-break hyphen
            8211: ord('-'),  # en dash
            8212: ord('-'), # long dash
            8216: ord('\''), # left single quote
            8217: ord('\''), # right single quote
            8220: ord('"'),  # left double quotes
            8221: ord('"'),  # right double quotes
            8224: ord('*'),  # dagger
            8225: ord('*'),  # double cross
            8226: ord('*'),  # bullet
            8242: ord('\''), # prime
            8482: ord('*'),  # ^tm
            8593: ord('*'),  # up arrow
            8595: ord('*'), # down arrow
            8596: ord('*'),  # leftright
            8597: ord('*'),  # upside
            8722: ord('-'), # minus sign
            8729: ord('*'), # bullet operator
            8734: ord('*'), #infinity
            8773: ord('='), # approx. eq
            8804: ord('<'), # leq
            8805: ord('>'), # geq
            9679: ord('*'), # bullet
}

if __name__ == '__main__':
    main(parser.parse_args())