import os, sys, time, random, re, json, collections, argparse, string
import xml.etree.ElementTree as ET
from xml.dom import minidom
from nltk.corpus import stopwords
from itertools import groupby

stoplist = stopwords.words('english')
parser = argparse.ArgumentParser(description='Process drug labels')
parser.add_argument('--input', type=str, default=None,
                    help='file input, defaults to STDIN')
parser.add_argument('--output-dir', type=str, default='tac_prediction',
                    help='path to dump data')
parser.add_argument('--coord', default=False, action='store_true',
                    help='combine coordinated mentions')
parser.add_argument('--textnodes', type=str, default=None,
                    help='indicate the directory to get text node')

def main(args):
    if args.input is None:
        dataset = load_data(sys.stdin)
    else:
        dataset = load_data(open(args.input,'r'))
    
    textnodes = None
    if args.textnodes is not None:
        textnodes = load_textnodes(args.textnodes)
    else:
        print('WARNING: TEXTNOTES NOT PROVIDED --- NEED THIS FOR FINAL SUBMISSION')
        print('CONTINUING IN 3 SECONDS...')
        time.sleep(3)

    drug_labels = {}
    for (attribs, x), y in dataset:
        dname = attribs[0]
        if dname not in drug_labels:
            drug_labels[dname] = []

        drug_labels[dname].append((attribs,x,y))

    for dname, sents in drug_labels.items():
        print(dname)
        midc = 0
        intc = 0
        label_node = ET.Element('Label')

        if textnodes is not None:
            label_node.append(textnodes[dname])
        else:
            ET.SubElement(label_node, 'Text')
            if args.textnodes:
                print('WARNING --- TEXTNODES FOR DRUGNAME NOT FOUND')

        sents_node = ET.SubElement(label_node, 'Sentences')
        ET.SubElement(label_node, 'LabelInteractions')
        
        global_interactions = []
        matched_sents = []
        for attrib, x, y in sorted(sents):
            dname, sent_id, sect_id, set_id, orig_sent, linkage = attrib
            linkage_pd, linkage_pk = linkage

            for pos, code in linkage_pk:
                if code == 'C54602':
                    linkage_pk.append((pos,'C54605'))
                elif code == 'C54606':
                    linkage_pk.append((pos,'C54609'))
                elif code == 'C54610':
                    linkage_pk.append((pos,'C54613'))
                elif code == 'C54615':
                    linkage_pk.append((pos,'C54614'))

            if textnodes is not None:
                sent_key = "{}#{}".format(dname, sent_id)
                matched_sents.append(sent_key)
                sent_node = textnodes[sent_key]
                orig_sent = sent_node.find('SentenceText').text
                sents_node.append(sent_node)
            else:
                sent_node = ET.SubElement(sents_node, 'Sentence', 
                        attrib={'id': sent_id,
                                'section': sect_id, 
                                'LabelDrug': dname})

                senttext = ET.SubElement(sent_node, 'SentenceText')
                senttext.text = orig_sent
            
            pos2mid = {}
            mentions_span = extract_mentions(x,y,orig_sent,args.coord)
            #print(mentions_span)
            trigger_ids = []

            for pos, start, end, label, span in sorted(mentions_span):
                if label.endswith('TRI'):
                    mtype  = 'Trigger'
                elif label.endswith('EFF'):
                    mtype = 'SpecificInteraction'
                else:
                    continue

                offsets = ';'.join([ '{} {}'.format(s, e-s+1)
                                    for s, e in zip(start,end) ])

                midc += 1
                if label.endswith('TRI'):
                    trigger_ids.append(midc)
                
                ment_node = ET.SubElement(sent_node, 'Mention', 
                            attrib={'id': 'M{}'.format(midc),
                                    'str': span, 
                                    'span': offsets,
                                    'type': mtype,
                                    'code': 'NO MAP'})
                
                if label.endswith('EFF'):
                    pos2mid[pos] = midc

            if args.coord:
                endpos_group = {}
                for pos, start, end, label, span in sorted(mentions_span):
                    if label in ['DYN','KIN','UNK']:
                        if not isinstance(end,list):
                            end = [end]

                        if end[-1] not in endpos_group:
                            endpos_group[end[-1]] = [pos]
                        else:
                            endpos_group[end[-1]].append(pos)
                
                for group in endpos_group.values():
                    for pos_a in group:
                        match = None
                        for (ppos,epos,pred) in linkage_pd:
                            if ppos == pos_a:
                                match = (epos,pred)
                        
                        if match is None:
                            continue

                        for pos_b in group:
                            if pos_b == pos_a:
                                continue
                            if (pos_b, *match) not in linkage_pd:
                                linkage_pd.append((pos_b, *match))

                for group in endpos_group.values():
                    for pos_a in group:
                        match = None
                        for ppos, pred in linkage_pk:
                            if ppos == pos_a:
                                match = pred
                        
                        if match is None:
                            continue

                        for pos_b in group:
                            if pos_b == pos_a:
                                continue
                            if (pos_b, match) not in linkage_pk:
                                linkage_pk.append((pos_b, match))

                    

            for pos, start, end, label, span in sorted(mentions_span):
                if label.endswith('DYN'):
                    int_type = 'Pharmacodynamic interaction'
                elif label.endswith('KIN'):
                    int_type = 'Pharmacokinetic interaction'
                elif label.endswith('UNK'):
                    int_type = 'Unspecified interaction'
                else:
                    continue

                offsets = ';'.join([ '{} {}'.format(s, e-s+1)
                                    for s, e in zip(start,end) ])

                midc += 1
                ment_node = ET.SubElement(sent_node, 'Mention', 
                            attrib={'id': 'M{}'.format(midc),
                                    'str': span, 
                                    'span': offsets,
                                    'type': 'Precipitant',
                                    'code': 'NO MAP'})
            
                trig_midc = midc
                if len(trigger_ids) > 1:
                    trig_midc = trigger_ids.pop(-1)
                elif len(trigger_ids) == 1:
                    trig_midc = trigger_ids[0]

                attr = {    'type': int_type,
                            'precipitant': 'M{}'.format(midc),
                            'trigger': 'M{}'.format(trig_midc) }

                if label.endswith('DYN'):
                    links = [ epos for mpos, epos, pred in linkage_pd 
                                if pos == mpos and pred == 1]
                    
                    if len(links) == 0:
                        pass # switch this on/off for submission
                        #intc += 1
                        #attr['id'] = 'I{}'.format(intc)
                        #int_node = ET.SubElement(sent_node, 
                        #            'Interaction', attrib=attr)
                    else:
                        mids = [ 'M{}'.format(pos2mid[eid]) for eid in links if eid in pos2mid ]
                        intc += 1
                        attr['id'] = 'I{}'.format(intc)
                        attr['effect'] = ';'.join(sorted(mids))
                        int_node = ET.SubElement(sent_node, 
                                    'Interaction', attrib=attr)

                elif label.endswith('KIN'):
                    links = [ eff for mpos, eff in linkage_pk 
                                if pos == mpos ]

                    for eff in links:
                        intc += 1
                        attr['id'] = 'I{}'.format(intc)
                        attr['effect'] = eff
                        int_node = ET.SubElement(sent_node, 
                                    'Interaction', attrib=attr)

                else:
                    intc += 1
                    attr['id'] = 'I{}'.format(intc)
                    int_node = ET.SubElement(sent_node, 'Interaction', 
                                                attrib=attr)
        
        if textnodes is not None:
            for k, v in textnodes.items():
                if k not in matched_sents and k.startswith("{}#".format(dname)):
                    sents_node.append(v)

        label_node.set('setid', set_id)
        label_node.set('drug', dname)
        
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        
        fname = "{}_{}.xml".format(dname.replace(' ','_'), set_id)
        outfile = os.path.join(args.output_dir,fname) 
        with open(outfile,'w',encoding='utf8') as f:
            print(prettify(label_node), file=f)

def load_textnodes(xml_dir):
    textnodes = {}
    if os.path.exists(xml_dir):
        for fname in os.listdir(xml_dir):
            if not fname.endswith('.xml'):
                continue

            fpath = os.path.join(xml_dir,fname)

            base_tree = ET.parse(fpath)
            base_root = base_tree.getroot()
            sent_seen = []

            setid = base_root.get('setid')
            dname = base_root.get('drug')
            textnodes[dname] = base_root.find('Text')

            for snode in base_root.findall('Sentences/Sentence'):
                sent_key = "{}#{}".format(dname, snode.get('id'))

                for child in snode[::-1]:
                    if child.tag != 'SentenceText':
                        snode.remove(child)
                
                textnodes[sent_key] = snode

            #print(dname, textnodes[dname])
    
    return textnodes

def prettify(elem):
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def extract_mentions(x,y,orig_sent, coordination):
    mentions = []
    spans = []

    #y = fix_iob(y)
    
    pos = len(x)+1
    for (start, end, w), tag in reversed(list(zip(x,y))):
        pos -= 1
        #print(tag, pos, start, end, w)
        if tag != 'O':
            spans.append((tag, pos, start, end, w))
            if tag.startswith('B') or tag.startswith('U'):
                if coordination:
                    mentions += postprocess_span_coord(spans, orig_sent)
                else:
                    mentions += postprocess_span(spans)

                spans = []

    mentions_out = []
    for ment in mentions:
        chunks = (list(g) for k,g in groupby(reversed(ment), 
                        key=lambda x: x != None) if k)

        starts = []
        ends = []
        spans = []
        for chunk in chunks:
            start = min([ s for _, _, s, _, _ in chunk ])
            end = max([ e for _, _, _, e, _ in chunk ])
            spans.append(orig_sent[start:end+1])
            starts.append(start)
            ends.append(end)

        label = [ tag[-3:] for tag, _, _, _, _ in [ z for z in ment if z is not None ] ]
        label = max(set(label), key=label.count)
        pos = min([ pos for _, pos, _, _, _ in [ z for z in ment if z is not None ] ])
        
        mentions_out.append((pos, starts, ends, 
                        label, ' | '.join(spans)))
 
    return mentions_out 

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

def postprocess_span_coord(span, sent):
    final_spans = []
    if len(span) > 2 and span[0][0][-3:] in ['DYN','KIN','UNK']:
        segment = ' '.join([ w for t, pos, start, end, w in sorted(span,key=lambda x: x[1]) ])
        headword = match_headwords(segment)
        anchor = segment.find(' and ')
        if anchor < 0:
            anchor = segment.find(' or ')
        
        if match_headwords(segment) and anchor >= 0:
            anchor += min([start for t, pos, start, end, w in span])
            
            new_span = []
            for z in span:
                if z[4] == headword:
                    new_span.append(z)
                elif (z[3] < anchor and z[4] not in 
                        [headword,'moderate','strong','potent']):
                    new_span.append(z)
                else:
                    new_span.append(None)
            
            final_spans.append(new_span)

            new_span = []
            for z in span:
                if z[4] == headword:
                    new_span.append(z)
                elif (z[2] > anchor+2 and z[4] not in 
                        [headword,'moderate','strong','potent']):
                    new_span.append(z)
                else:
                    new_span.append(None)
            
            final_spans.append(new_span)

            print(final_spans)

    if len(final_spans) == 0:
        return postprocess_span(span)
    else:
        return final_spans

def postprocess_span(span):
    if span[0][0][-3:] in ['EFF','DYN','KIN','UNK']:
        words = [ x[-1] for x in span if x is not None ]
        stopw = [ w for w in words if w in stoplist
            + ['XXXXXXXX','drug','drugs','agent','agents'] ]
        if len(stopw) == len(words) or 'not' in words:
            #print(words,'NUKED')
            return []

    # Split triggers on drug label -- only helps NER
    if len(span) > 2 and span[0][0].endswith('T'):
        for i, (tag, pos, start, end, w) in enumerate(span):
            if 'XXX' in w:
                span[i] = None
    
    # Remove adjectives from span
    elif len(span) > 1 and span[0][0][-3:] in ['DYN','KIN','UNK']:
        for i, (tag, pos, start, end, w) in enumerate(span):
            if i == len(span)-1 and w in ['moderate','strong','potent']:
                span[i] = None
            
    return [span]

def load_data(f):
    cast_int = lambda x: tuple([ int(z) if not z.startswith('C') else z for z in x ])

    examples = []
    while True:
        header = f.readline()
        if header == "":
            break

        orig_sent = f.readline().strip()
        if orig_sent == "":
            break
        
        dname, sent_id, sect_id, set_id, num_tokens = header.split('\t')

        x = []
        y = []
        for i in range(int(num_tokens)):
            _, st, ed, label, tok = f.readline().rstrip().split('\t')
            x.append((int(st), int(ed), tok))
            y.append(label)

        linkages = f.readline().strip()
        
        pds = [ cast_int(z) for z in 
                re.findall(r'D\/([0-9]+):([0-9]+):([01])',linkages) ]
        pks = [ cast_int(z) for z in 
                re.findall(r'K\/([0-9]+):(C[0-9]+)',linkages) ]
        
        examples.append((((dname, sent_id, sect_id, set_id, 
                            orig_sent, (pds,pks)), x), y))
        f.readline()

    return examples

if __name__ == '__main__':
    main(parser.parse_args())