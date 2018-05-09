import codecs
import json
import ast

def transfer_wordlist_2_idlist_with_maxlen(token_list, vocab_map, maxlen):
    '''
    From such as ['i', 'love', 'Munich'] to idlist [23, 129, 34], if maxlen is 5, then pad two zero in the left side, becoming [0, 0, 23, 129, 34]
    '''
    idlist=[]
    for word in token_list:

        id=vocab_map.get(word)
        if id is None: # if word was not in the vocabulary
            id=len(vocab_map)+1  # id of true words starts from 1, leaving 0 to "pad id"
            vocab_map[word]=id
        idlist.append(id)

    mask_list=[1.0]*len(idlist) # mask is used to indicate each word is a true word or a pad word
    pad_size=maxlen-len(idlist)
    if pad_size>0:
        idlist=[0]*pad_size+idlist
        mask_list=[0.0]*pad_size+mask_list
    else: # if actual sentence len is longer than the maxlen, truncate
        idlist=idlist[:maxlen]
        mask_list=mask_list[:maxlen]
    return idlist, mask_list

def load_fever_train(sent_len, claim_len, cand_size):
    '''
    sents_ids=T.itensor3() #(batch, cand_size, sent_len)
    sents_mask=T.ftensor3()
    sents_labels=T.imatrix() #(batch, cand_size)
    claim_ids = T.imatrix() #(batch, claim_len)
    claim_mask = T.imatrix()
    '''
    filename="/save/wenpeng/datasets/FEVER/train.reformat.2classes.support.refute.json"
    # files=['ReliefWeb.train.balanced.txt', 'ReliefWeb.test.balanced.txt', 'translated2017/E30_id_label_segment.txt']
    word2id={}  # store vocabulary, each word map to a id
    all_sents=[]
    all_sent_masks=[]
    all_sent_labels=[]
    all_claims = []
    all_claim_mask = []
    all_labels = []

    readfile=codecs.open(filename, 'r', 'utf-8')
    print 'loading ... ', filename
    co = 0
    for line in readfile:
        line2dict = json.loads(line) #ast.literal_eval(line.strip())
        claim_label = line2dict.get('label')
        sents = line2dict.get('sent_cands')
        claim = line2dict.get('claim')
        sent_labels = line2dict.get('sent_binary')
        sents = sents[:cand_size]
        sent_labels = sent_labels[:cand_size]

        sent_ins_ids = []
        sent_ins_mask = []
        for sent in sents:
            sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sent.split(), word2id, sent_len)
            sent_ins_ids.append(sent_idlist)
            sent_ins_mask.append(sent_masklist)
        # print 'len(sent_ins_ids):', len(sent_ins_ids)
        # print 'line:',line
        assert len(sent_ins_ids) == len(sent_labels)
        claim_idlist, claim_masklist=transfer_wordlist_2_idlist_with_maxlen(claim.split(), word2id, claim_len)

        all_sents.append(sent_ins_ids)
        all_sent_masks.append(sent_ins_mask)
        all_sent_labels.append(sent_labels)
        all_claims.append(claim_idlist)
        all_claim_mask.append(claim_masklist)
        all_labels.append(claim_label)
        co+=1
        # if co%5000==0:
        #     print 'load size ... ', co
        # if co == 1000:
        #     break

    readfile.close()
    print '\t\t\t size:', len(all_claims)
    print 'dataset loaded over, totally ', len(word2id), 'words'
    return all_sents, all_sent_masks, all_sent_labels, all_claims, all_claim_mask, all_labels, word2id

def load_fever_train_NoEnoughInfo(sent_len, claim_len, cand_size, word2id):
    '''
    sents_ids=T.itensor3() #(batch, cand_size, sent_len)
    sents_mask=T.ftensor3()
    sents_labels=T.imatrix() #(batch, cand_size)
    claim_ids = T.imatrix() #(batch, claim_len)
    claim_mask = T.imatrix()
    '''
    filename="/save/wenpeng/datasets/FEVER/train.reformat.3th.class.json"
    all_sents=[]
    all_sent_masks=[]
    all_sent_labels=[]
    all_claims = []
    all_claim_mask = []
    all_labels = []

    readfile=codecs.open(filename, 'r', 'utf-8')
    print 'loading ... ', filename

    co = 0
    for line in readfile:
        line2dict = json.loads(line)
        sents = line2dict.get('sent_cands')
        claim = line2dict.get('claim')
        claim_label = line2dict.get('label')
        # sent_labels = line2dict.get('sent_binary')
        # sents = sents[:cand_size]
        # sent_labels = sent_labels[:cand_size]
        assert len(sents) > 0
        sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sents[0].split(), word2id, sent_len)
        sent_ins_ids=[sent_idlist]*cand_size
        sent_ins_mask=[sent_masklist]*cand_size
        sent_labels = [1]+[0]*(cand_size-1)

        claim_idlist, claim_masklist=transfer_wordlist_2_idlist_with_maxlen(claim.split(), word2id, claim_len)

        all_sents.append(sent_ins_ids)
        all_sent_masks.append(sent_ins_mask)
        all_sent_labels.append(sent_labels)
        all_claims.append(claim_idlist)
        all_claim_mask.append(claim_masklist)
        all_labels.append(claim_label)
        co+=1
        # if co%5000==0:
        #     print 'load train 3th size ... ', co
        # if co == 1000:
        #     break

    readfile.close()
    print '\t\t\t size:', len(all_claims)
    print 'dataset loaded over, totally ', len(all_claims), ' claims'
    return all_sents, all_sent_masks, all_sent_labels, all_claims, all_claim_mask, all_labels, word2id

def load_fever_dev_NoEnoughInfo(sent_len, claim_len, cand_size, word2id):
    '''
    sents_ids=T.itensor3() #(batch, cand_size, sent_len)
    sents_mask=T.ftensor3()
    sents_labels=T.imatrix() #(batch, cand_size)
    claim_ids = T.imatrix() #(batch, claim_len)
    claim_mask = T.imatrix()
    '''
    filename="/save/wenpeng/datasets/FEVER/dev.jsonl.reformat.3th.class.json"
    all_sents=[]
    all_sent_masks=[]
    all_sent_labels=[]
    all_claims = []
    all_claim_mask = []
    all_labels = []

    readfile=codecs.open(filename, 'r', 'utf-8')
    print 'loading ... ', filename

    co = 0
    for line in readfile:
        line2dict = json.loads(line)
        sents = line2dict.get('sent_cands')
        claim = line2dict.get('claim')
        claim_label = line2dict.get('label')
        # sent_labels = line2dict.get('sent_binary')
        # sents = sents[:cand_size]
        # sent_labels = sent_labels[:cand_size]
        assert len(sents) > 0
        sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sents[0].split(), word2id, sent_len)
        sent_ins_ids=[sent_idlist]*cand_size
        sent_ins_mask=[sent_masklist]*cand_size
        sent_labels = [1]+[0]*(cand_size-1)

        claim_idlist, claim_masklist=transfer_wordlist_2_idlist_with_maxlen(claim.split(), word2id, claim_len)

        all_sents.append(sent_ins_ids)
        all_sent_masks.append(sent_ins_mask)
        all_sent_labels.append(sent_labels)
        all_claims.append(claim_idlist)
        all_claim_mask.append(claim_masklist)
        all_labels.append(claim_label)
        co+=1
        # if co%5000==0:
        #     print 'load dev 3th size ... ', co
        # if co == 10000:
        #     break

    readfile.close()
    print '\t\t\t size:', len(all_claims)
    print 'dataset loaded over, totally ', len(all_claims), ' claims'
    return all_sents, all_sent_masks, all_sent_labels, all_claims, all_claim_mask, all_labels, word2id

def load_fever_dev(sent_len, claim_len, cand_size, word2id):
    '''
    sents_ids=T.itensor3() #(batch, cand_size, sent_len)
    sents_mask=T.ftensor3()
    sents_labels=T.imatrix() #(batch, cand_size)
    claim_ids = T.imatrix() #(batch, claim_len)
    claim_mask = T.imatrix()
    '''
    filename="/save/wenpeng/datasets/FEVER/dev.jsonl.reformat.2classes.support.refute.json"

    all_sents=[]
    all_sent_word_masks=[]
    all_sent_sent_masks=[] #which sent cand is valid or fake
    # all_sent_labels=[]
    all_claims = []
    all_labels = []
    all_claim_mask = []
    all_sent_names = []
    all_ground_names = []

    readfile=codecs.open(filename, 'r', 'utf-8')
    print 'loading ... ', filename
    co=0
    for line in readfile:
        line2dict = json.loads(line)
        sents = line2dict.get('sent_cands')
        claim = line2dict.get('claim')
        claim_label = line2dict.get('label')
        sent_names = line2dict.get('sent_names')
        sent_names = [name[0]+' '+str(name[1]) for name in sent_names]
        ground_name_list = line2dict.get('ground_truth_names')
        ground_name_list = [name[0]+' '+str(name[1]) for name in ground_name_list]


        sent_ins_ids = []
        sent_ins_mask = []
        for sent in sents:
            sent_idlist, sent_masklist=transfer_wordlist_2_idlist_with_maxlen(sent.split(), word2id, sent_len)
            sent_ins_ids.append(sent_idlist)
            sent_ins_mask.append(sent_masklist)

        true_cand_size = len(sents)
        if true_cand_size >= cand_size:
            sent_ins_ids = sent_ins_ids[:cand_size]
            sent_ins_mask = sent_ins_mask[:cand_size]
            sent_names = sent_names[:cand_size]
            sent_sent_mask = [1.0]*cand_size
        else:
            fake_size = cand_size - true_cand_size
            fake_sent_id = sent_ins_ids[-1:]
            fake_sent_mask = sent_ins_mask[-1:]
            fake_sent_name = sent_names[-1:]
            sent_ins_ids = sent_ins_ids + fake_sent_id*fake_size
            sent_ins_mask = sent_ins_mask + fake_sent_mask*fake_size
            sent_names = sent_names + fake_sent_name*fake_size
            sent_sent_mask = [1.0]*true_cand_size+[0.0]*fake_size






        assert len(sent_ins_ids) == cand_size
        claim_idlist, claim_masklist=transfer_wordlist_2_idlist_with_maxlen(claim.split(), word2id, claim_len)

        all_sents.append(sent_ins_ids)
        all_sent_word_masks.append(sent_ins_mask)
        all_sent_sent_masks.append(sent_sent_mask)
        all_claims.append(claim_idlist)
        all_labels.append(claim_label)
        all_claim_mask.append(claim_masklist)
        all_sent_names.append(sent_names)
        all_ground_names.append(ground_name_list)
        co+=1
        # if co%5000==0:
        #     print 'load size ... ', co
        # if co == 1000:
        #     break
    readfile.close()
    print '\t\t\t size:', len(all_claims)
    print 'dataset loaded over, totally ', len(word2id), 'words'
    return all_sents, all_sent_word_masks, all_sent_sent_masks, all_claims, all_claim_mask, all_sent_names,all_ground_names,all_labels,word2id

if __name__ == '__main__':
    # load_fever_train(maxlen=40)
    load_fever_train_NoEnoughInfo(40, 40, 20, {})
