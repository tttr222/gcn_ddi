import sys
import os
import re
import argparse
from xml.etree import ElementTree


VALID_MENTION_TYPES = set(['Trigger', 'Precipitant', 'SpecificInteraction', 'SpecialInteraction'])
VALID_INTERACTION_TYPES = set(['Pharmacodynamic interaction', 'Unspecified interaction', 'Pharmacokinetic interaction'])
VALID_MENTION_OFFSETS = re.compile('([0-9]+)\s+([0-9]+);?')


class Label:

    def __init__(self, drug):
        self.drug = drug
        self.sentences = {}
        self.mentions = []
        self.interactions = []
        self.globalinteractions = []

        
class Sentence:
    def __init__(self, sid, text):
        self.sid = sid
        self.text = text

class Mention:
    
    def __init__(self, sid, mid, stype, span, code, codestr, mstr):
        self.sid = sid
        self.mid = mid
        self.stype = stype
        self.span = span
        self.code = code
        self.codestr = codestr
        self.mstr = mstr
        
    def __str__(self):
        return 'Mention(isd={},stype={},span={},code={},mstr="{}")'.format(self.sid, self.stype, self.span, self.code, self.mstr)
    def __repr__(self):
        return str(self)


class Interaction:

    def __init__(self, sid, rid, rtype, trigger, precipitant, effect):
        self.sid = sid
        self.rid = rid
        self.rtype = rtype
        self.trigger = trigger
        self.precipitant = precipitant
        self.effect = effect
    def __str__(self):
        return 'Relation(type={},precipitant={},effect={})'.format(self.rtype, self.precipitant, self.effect)
    def __repr__(self):
        return str(self)

class Results:
    
    def __init__(self, task1, task2, task3, task4):
        self.task1 = Task1() if task1 else None
        self.task2 = Task2() if task2 else None
        self.task3 = Task3() if task3 else None
        self.task4 = Task4() if task4 else None

class Task1:
    def __init__(self):
        self.exact_type = Classification()
        self.exact_notype = Classification()

class Task2:
    def __init__(self):
        self.full_type = Classification()
        self.full_notype = Classification()
        self.binary_type = Classification()
        self.binary_notype = Classification()


class Task3:
    def __init__(self):
        self.classifications = []


class Task4:
    def __init__(self):
        self.classifications = []


class Classification:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        
    def precision(self):
        if self.tp == 0:
            return 0.0
        return 100. * self.tp / (self.tp + self.fp)
    
    def recall(self):
        if self.tp == 0:
            return 0.0
        return 100. * self.tp / (self.tp + self.fn)
    
    def f1(self):
        p, r = self.precision(), self.recall()
        if p + r == 0.0:
            return 0.0
        return 2 * p * r / (p + r)

def merge_classifications(classifications):
    classification = Classification()
    for c in classifications:
        classification.tp += c.tp
        classification.fp += c.fp
        classification.fn += c.fn
        classification.tn += c.tn
    return classification

# Returns all the XML files in a directory as a dict
def xml_files(adir):
    files = {}
    for file in os.listdir(adir):
        if file.endswith('.xml'):
            files[file.replace('.xml', '')] = os.path.join(adir, file)
    return files


# Compares the files in the two directories using compare_files
def compare_dirs(gold_dir, guess_dir, results):
    gold_files = xml_files(gold_dir)
    guess_files = xml_files(guess_dir)
    match1 = True
    match2 = True
    keys = sorted([x for x in gold_files if x in guess_files])
    for key in gold_files:
        if key not in keys:
            print('WARNING: gold label file not found in guess directory: ' + key)
            match1 = False
    for key in guess_files:
        if key not in keys:
            print('WARNING: guess label file not found in gold directory: ' + key)
            match2 = False
    if not match1:
        sys.exit('Not all test files submitted!')
    if not match2:
        sys.exit('Non-test files submitted!')
        
    for key in keys:
#        print('reading ' + key)
        compare_files(gold_files[key], guess_files[key], results)


def read(file):
    root = ElementTree.parse(file).getroot()
    assert root.tag == 'Label', 'Root is not Label: ' + root.tag
    label = Label(root.attrib['drug'])
    assert len(root) == 3, 'Expected 3 Children: ' + str(list(root))
    assert root[0].tag == 'Text', 'Expected \'Text\': ' + root[0].tag
    assert root[1].tag == 'Sentences', 'Expected \'Sentences\': ' + root[0].tag
    assert root[2].tag == 'LabelInteractions', 'Expected \'LabelInteractions\': ' + root[0].tag
    
    for elem in root[1]:
        assert elem.tag == 'Sentence', 'Expected \'Sentence\': ' + elem.tag
        #label.sentences.append(Sentence(elem.attrib['id'], elem.getchildren()[0].text) )
        label.sentences[elem.attrib['id']] = elem.getchildren()[0].text
        for mention in elem.findall('Mention'):
#            if mention.attrib['type'] =='Trigger':
                label.mentions.append(
                Mention( elem.attrib['id'], \
                mention.attrib['id'], \
                mention.attrib['type'], \
                mention.attrib['span'], \
                '', \
                '', \
                mention.attrib['str']))
#             else:
#                 matchCode = re.match( '^([0-9]+):', mention.attrib['code'])
#                   
#                 if matchCode:
#                     label.mentions.append(
#                     Mention( elem.attrib['id'], \
#                     mention.attrib['id'], \
#                     mention.attrib['type'], \
#                     mention.attrib['span'], \
#                     matchCode.group(1),\
#                     mention.attrib['code'], \
#                    mention.attrib['str']))
#                       
#                 else:
#                     label.mentions.append(
#                     Mention( elem.attrib['id'], \
#                     mention.attrib['id'], \
#                     mention.attrib['type'], \
#                     mention.attrib['span'], \
#                     mention.attrib['code'], \
#                     ' ', \
#                    mention.attrib['str']))
#               
        for interaction in elem.findall('Interaction'): #rid, rtype, trigger, precipitant, effect
            if len(interaction.attrib) == 4:
                label.interactions.append(
                Interaction(elem.attrib['id'], \
                interaction.attrib['id'], \
                interaction.attrib['type'], \
                interaction.attrib['trigger'], \
                interaction.attrib['precipitant'], \
                ''))
                 
            else:
                label.interactions.append(
                Interaction(elem.attrib['id'], \
                interaction.attrib['id'], \
                interaction.attrib['type'], \
                interaction.attrib['trigger'], \
                interaction.attrib['precipitant'], \
                interaction.attrib['effect']))
    
    for interaction in root[2]:
        assert interaction.tag == 'LabelInteraction', 'Expected \'LabelInteraction\': ' + interaction.tag
        if len(interaction.attrib) == 3:
            label.globalinteractions .append(
                 Interaction('','', \
                interaction.attrib['type'], \
                '', \
                interaction.attrib['precipitantCode'], \
                ''))
        else:
            label.globalinteractions .append(
                Interaction('','', \
                interaction.attrib['type'], \
                '', \
                interaction.attrib['precipitantCode'], \
                interaction.attrib['effect']))
      
    return label


# Compares the two files
def compare_files(gold_file, guess_file, results):
    # print('Evaluating: ' + os.path.basename(gold_file).replace('.xml', ''))
    gold_label = read(gold_file)
    guess_label = read(guess_file)
    validate_ind(guess_label)
    validate_ind(gold_label)
    validate_both(gold_label, guess_label)
    
    if results.task1 is not None:
        eval_task1(gold_label, guess_label, results)
    
    if results.task2:
        eval_task2(gold_label, guess_label, results)
        
    if results.task3:
        eval_task3(gold_label, guess_label, results)
        
    if results.task4:
        eval_task4(gold_label, guess_label, results)


# Validates an individual Label
def validate_ind(label):
    mentions = {}
    interactions = {}
   
    for mention in label.mentions:
        assert mention.mid.startswith('M'), 'Mention ID does not start with M: ' + mention.mid
        assert mention.mid not in mentions, 'Duplicate Mention ID: ' + mention.mid
#        assert VALID_MENTION_OFFSETS.match(mention.span), 'Invalid span attribute: ' + mention.span
        assert mention.stype in VALID_MENTION_TYPES, 'Invalid Mention type: ' + mention.stype
        
        if mention.mstr is not None:
            mentions[mention.mid] = mention
            text = ''
            for (sstart, slen) in re.findall(VALID_MENTION_OFFSETS, mention.span):
                start = int(sstart)
                end =  start + int(slen)
                if len(text) > 0:
                    text += ' '
                span = label.sentences.get(mention.sid)[start:end]
                span = re.sub('\s+', ' ', span)
                text += span
                str2 = re.sub('\\|', ' ', mention.mstr)
#                assert text ==  re.sub('\s+', ' ', str2.strip() ), 'Mention has wrong string value.' + \
 #               '  From \'str\': \'' + re.sub('[\s+\\|]', ' ', mention.mstr) + '\'' + \
  #              '  From offsets: \'' + text + '\'' + label.drug
          
   
    for interaction in label.interactions:
        assert interaction.rid.startswith('I'), 'Relation ID does not start with I: ' + interaction.rid
        assert interaction.rid not in interactions, 'Duplicate Relation ID: ' + interaction.rid 
        assert interaction.rtype in VALID_INTERACTION_TYPES, 'Invalid Relation type: ' + interaction.rtype 
#        for anInt in interaction.trigger.split(';'):
#            assert anInt in mentions, 'Interaction ' + interaction.rid  + ' trigger not in mentions: ' + anInt + label.drug
        assert interaction.precipitant  in mentions, 'Interaction ' +interaction.rid + ' not in mentions: ' + interaction.precipitnt 
        if interaction.effect!='' and interaction.effect is not None and not interaction.effect .startswith('C'):
            for anInt in interaction.effect.split(';'):
                assert anInt in mentions, 'Interaction ' + interaction.rid + ' not in mentions:' + anInt +"~"
      
        interactions[interaction.rid] = interaction
    
    for gi in label.globalinteractions:
        assert gi.rtype in VALID_INTERACTION_TYPES, 'Invalid Relation type: ' + gi.rtype 
      
# Validates that the two Labels are similar enough to merit comparing
# performance metrics, mainly just comparing the sections/text to make sure
# they're identical
def validate_both(l1, l2):
    assert len(l1.sentences) == len(l2.sentences), \
      'Different number of sentences: ' + str(len(l1.sentences)) + \
      ' vs. ' + str(len(l2.sentences))
    for key, value in l2.sentences.items():
        assert key in l1.sentences.keys(), \
        'Different sentence ID in GUESS ' + key 
        assert value == l1.sentences.get(key), 'Different sentence text: ' + value +\
        ' vs. ' + l1.sentences.get(key)
   
# Evaluates Task 1 (Mentions)
def eval_task1(gold_label, guess_label, results):
    # EXACT + TYPE
    eval_f(set([exact_mention_repr(m, type=True) for m in gold_label.mentions]), \
           set([exact_mention_repr(m, type=True) for m in guess_label.mentions]), \
           results.task1.exact_type)
    # EXACT - TYPE
    eval_f(set([exact_mention_repr(m, type=False) for m in gold_label.mentions]), \
           set([exact_mention_repr(m, type=False) for m in guess_label.mentions]), \
           results.task1.exact_notype)

# Representation for exact matching of mentions
def exact_mention_repr(m, type):
    repr = m.sid + ':' + m.span
    if type:
        repr += ':' + m.stype 
    return repr

# Evaluates Task 2 (Relations)
def eval_task2(gold_label, guess_label, results):
    gold_mentions = {m.mid:m for m in gold_label.mentions}
    guess_mentions = {m.mid:m for m in guess_label.mentions}
    # Full + TYPE
    eval_f(set([full_relation_repr(r, gold_mentions,  type=True) for r in gold_label.interactions]), \
         set([full_relation_repr(r, guess_mentions, type=True) for r in guess_label.interactions]), \
         results.task2.full_type)
    # Full - TYPE  
    eval_f(set([full_relation_repr(r, gold_mentions,  type=False) for r in gold_label.interactions]), \
         set([full_relation_repr(r, guess_mentions, type=False) for r in guess_label.interactions]), \
        results.task2.full_notype)
 # Binary + TYPE
    eval_f(binary_relation_repr(gold_label.interactions,  gold_mentions,  type=True), \
           binary_relation_repr(guess_label.interactions, guess_mentions, type=True), \
           results.task2.binary_type)
    # Binary - TYPE
    eval_f(binary_relation_repr(gold_label.interactions,  gold_mentions,  type=False), \
         binary_relation_repr(guess_label.interactions, guess_mentions, type=False), \
         results.task2.binary_notype)
# Representation for binary matching of relations
def binary_relation_repr(interactions, mentions, type):
    reprs = set()
    binary_interactions = {}
    spans = set()
    for r in interactions:
        if r.precipitant not in binary_interactions:
            binary_interactions[r.precipitant] = []
            arg_repr = exact_mention_repr(mentions[r.precipitant], type=type)
            if type:
                arg_repr = r.rtype + ':' + arg_repr
            binary_interactions[r.precipitant].append(arg_repr)
        
    for key, value in binary_interactions.items():
        repr = exact_mention_repr(mentions[key], type=type)
        spans.add(repr)
        for item in sorted(value):
            repr += '::'
            repr += item
            reprs.add(repr)
        
    for _, mention in mentions.items():
        repr = exact_mention_repr(mention, type=type)
        if repr in spans and mention.mid not in binary_interactions:
            reprs.add(repr)
    return reprs

# Representation for matching of full relations
def full_relation_repr(interaction, mentions, type):
    if interaction.effect .startswith('C'):
        repr = exact_mention_repr(mentions[interaction.precipitant], type=type) + '::' +\
            interaction.effect
        if type:
            repr += '::' + interaction.rtype
    elif interaction.effect!='' and interaction.effect is not None:
        for anInt in interaction.effect.split(';'):
            repr = exact_mention_repr(mentions[interaction.precipitant], type=type) + '::' +\
            exact_mention_repr(mentions[anInt], type=type)
            if type:
                repr += '::' + interaction.rtype
    else:
        repr = exact_mention_repr(mentions[interaction.precipitant], type=type) 
        if type:
            repr += '::' + interaction.rtype
    return repr

# Evaluates Task 3 (Normaization)
def eval_task3(gold_label, guess_label, results):
    classification = Classification()
    eval_f(set([norm_repr(m) for m in gold_label.mentions]), \
           set([norm_repr(m) for m in guess_label.mentions]), \
           classification)
    results.task3.classifications.append(classification)
    
    

# Evaluates Task 4 (GlobalInteractions)
def eval_task4(gold_label, guess_label, results):
    classification = Classification()
    eval_f(set([reaction_repr(r) for r in gold_label.globalinteractions]), \
         set([reaction_repr(r) for r in guess_label.globalinteractions]), \
         classification)
    results.task4.classifications.append(classification)

# Representation for matching reaction strings
def reaction_repr(r):
    return str(r)

# Representation for matching norm
def norm_repr(m):
    return m.code
# Calculates statistics needed for F-measure (TP, FP, FN)
def eval_f(gold_set, guess_set, classification):
    c = {}
    for gold in gold_set:
        if not gold or gold=="":
            continue
        if gold in guess_set:
            classification.tp += 1
            assert gold not in c
            c[gold] = 'TP'
        else:
            classification.fn += 1
            assert gold not in c
            c[gold] = 'FN'
    for guess in guess_set:
        if guess not in gold_set:
            classification.fp += 1
            assert guess not in c
            c[guess] = 'FP'

# Prints various numbers related to F-measure
def print_f(name, classification, primary=False):
    print('  ' + name)
    print('    TP: {}  FP: {}  FN: {}'.format(classification.tp, classification.fp, classification.fn))
    print('    Precision: {:.2f}'.format(classification.precision()))
    print('    Recall:    {:.2f}'.format(classification.recall()))
    print('    F1:        {:.2f}  {}'.format(classification.f1(), '(PRIMARY)' if primary else ''))

# Prints various numbers related to macro F-measure
def print_macro_f(classifications):
    merge = merge_classifications(classifications)
    print('    TP: {}  FP: {}  FN: {}'.format(merge.tp,
      merge.fp, merge.fn))
    print('    Micro-Precision: {:.2f}'.format(merge.precision()))
    print('    Micro-Recall:    {:.2f}'.format(merge.recall()))
    print('    Micro-F1:        {:.2f}'.format(merge.f1()))
    print('    Macro-Precision  {:.2f}'.format(
      sum([c.precision() for c in classifications])/len(classifications)))
    print('    Macro-Recall     {:.2f}'.format(
      sum([c.recall() for c in classifications])/len(classifications)))
    print('    Macro-F1         {:.2f}  (PRIMARY)'.format(
      sum([c.f1() for c in classifications])/len(classifications)))
  
if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Evaluate TAC 2018 Drug Drug Interactions Extraction task')
        parser.add_argument('gold_dir', metavar='GOLD', type=str, help='path to directory containing system output')
        parser.add_argument('guess_dir', metavar='GUESS', type=str, help='path to directory containing system output')
        parser.add_argument('-1, --task1', action='store_true', dest='task1',
                    help='Evaluate Task 1')
        parser.add_argument('-2, --task2', action='store_true', dest='task2',
                    help='Evaluate Task 2')
        parser.add_argument('-3, --task3', action='store_true', dest='task3',
                    help='Evaluate Task 3')
        parser.add_argument('-4, --task4', action='store_true', dest='task4',
                    help='Evaluate Task 4')
        args = parser.parse_args()
        tasks = [args.task1, args.task2, args.task3, args.task4]
        if sum(tasks) == 0:
            args.task1, args.task2, args.task3, args.task4 = True, True, True, True
        
        results = Results(args.task1, args.task2, args.task3, args.task4)

        print('Gold Directory:  ' + args.gold_dir)
        print('Guess Directory: ' + args.guess_dir)
        compare_dirs(args.gold_dir, args.guess_dir, results)
        if args.task1:
            print('--------------------------------------------------')
            print('Task 1 Results:')
            print_f('Exact (+type)', results.task1.exact_type, primary=True)
            print_f('Exact (-type)', results.task1.exact_notype)
        
        if args.task2:
            print('--------------------------------------------------')
            print('Task 2 Results:')
            print_f('Full (+type)', results.task2.full_type, primary=True)
            print_f('Full (-type)', results.task2.full_notype)
            print_f('Binary (+type)', results.task2.binary_type)
            print_f('Binary (-type)', results.task2.binary_notype)
        
        if args.task3:
            print('--------------------------------------------------')
            print('Task 3 Results:')
            print_macro_f(results.task3.classifications)
        
        if args.task4:
            print('--------------------------------------------------')
            print('Task 4 Results:')
            print_macro_f(results.task4.classifications)

