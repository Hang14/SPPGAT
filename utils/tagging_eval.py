import os 
import sys 

import math 
import numpy as np 


def cal_f1_score(pcs, rec):
    tmp = 2 * pcs * rec / (pcs + rec) 
    return round(tmp, 4) 


def extract_entities(labels_lst, start_label = "1_4"):
    def gen_entities(label_lst, start_label=1, dims=1):
        # rules -> if end_mark > start_label 
        entities = dict()
        
        if "_" in start_label:
            start_label = start_label.split("_")
            start_label = [int(tmp) for tmp in start_label]
            ind_func = lambda x: (bool(label in start_label) for label in x)
            indicator = sum([int(tmp) for tmp in ind_func(label_lst)])
        else:
            start_label = int(start_label)
            indicator = 1 if start_label in labels_lst else 0 

        if indicator > 0:
            if isinstance(start_label, list):
                ixs, _ = zip(*filter(lambda x: x[1] in start_label, enumerate(label_lst))) 
            elif isinstance(start_label, int):
                ixs, _ = zip(*filter(lambda x: x[1] == start_label, enumerate(label_lst)))
            else:
                raise ValueError("You Should Notice that The FORMAT of your INPUT")

            ixs = list(ixs)
            ixs.append(len(label_lst))
            for i in range(len(ixs) - 1):
                sub_label = label_lst[ixs[i]: ixs[i+1]]
                end_mark = max(sub_label)
                end_ix = ixs[i] + sub_label.index(end_mark) + 1 
                entities["{}_{}".format(ixs[i], end_ix)] = label_lst[ixs[i]: end_ix]
        return entities 


    if start_label == "1" :
        entities = gen_entities(labels_lst, start_label = int(start_label))
    elif start_label == "4":
        entities = gen_entities(labels_lst, start_label = int(start_label))
    elif "_" in start_label:
        entities = gen_entities(labels_lst, start_label = start_label) 
    else:
        raise ValueError("You Should Check The FOMAT Of your SPLIT NUMBER !!!!!")

    return entities 


def split_index(label_list):
    label_dict = {label: i for i, label in enumerate(label_list)}
    label_idx = [tmp_value for tmp_key, tmp_value in label_dict.items() if "S" in tmp_key.split("-")[0] or "B" in tmp_key]
    str_label_idx = [str(tmp) for tmp in label_idx]
    label_idx = "_".join(str_label_idx)
    return label_idx 


# def compute_performance(pred_label, gold_label, pred_mask, label_list, dims=2, macro=False):
#     start_label = split_index(label_list)
#     print(start_label)

#     if dims == 1:
#         mask_index = [tmp_idx for tmp_idx, tmp in enumerate(pred_mask) if tmp != 0]
#         pred_label = [tmp for tmp_idx, tmp in enumerate(pred_label) if tmp_idx in mask_index]
#         gold_label = [tmp for tmp_idx, tmp in enumerate(gold_label) if tmp_idx in mask_index]

#         pred_entities = extract_entities(pred_label, start_label = start_label)
#         truth_entities = extract_entities(gold_label, start_label = start_label)

#         # print("="*20)
#         # print("pred entities")
#         # print(pred_entities)
#         # print(truth_entities)
#         num_true = len(truth_entities)
#         num_extraction = len(pred_entities)

#         num_true_positive = 0 
#         for entity_idx in pred_entities.keys():
#             try:
#                 if truth_entities[entity_idx] == pred_entities[entity_idx]:
#                     num_true_positive += 1 
#             except:
#                 pass 

#         dict_match = list(filter(lambda x: x[0] == x[1], zip(pred_label, gold_label)))
#         acc = len(dict_match) / float(len(gold_label))

#         if not macro:
#             return acc, num_true_positive, float(num_extraction), float(num_true)

#         if num_extraction != 0:
#             pcs = num_true_positive / float(num_extraction)
#         else:
#             pcs = 0 

#         if num_true != 0:
#             recall = num_true_positive / float(num_true)
#         else:
#             recall = 0 

#         if pcs + recall != 0 :
#             f1 = 2 * pcs * recall / (pcs + recall)
#         else:
#             f1 = 0 

#         if num_extraction == 0 and num_true == 0:
#             acc, pcs, recall, f1 = 0, 0, 0, 0 
#         acc, pcs, recall, f1 = round(acc, 4), round(pcs, 4), round(recall, 4), round(f1, 4)

#         return acc, pcs, recall, f1 

#     elif dims == 2:
#         if not macro:
#             acc, posit, extra, true = 0, 0, 0, 0
#             for pred_item, truth_item, mask_item in zip(pred_label, gold_label, pred_mask):
#                 tmp_acc, tmp_posit, tmp_extra, tmp_true = compute_performance(pred_item, truth_item, mask_item, label_list, dims=1)
#                 posit += tmp_posit 
#                 extra += tmp_extra 
#                 true += tmp_true 
#                 acc += tmp_acc 

#             if extra != 0:
#                 pcs = posit / float(extra)
#             else:
#                 pcs = 0 

#             if true != 0:
#                 recall = posit / float(true)
#             else:
#                 recall = 0 

#             if pcs + recall != 0 :
#                 f1 = 2 * pcs * recall / (pcs + recall)
#             else:
#                 f1 = 0 
#             acc = acc / len(pred_label)
#             acc, pcs, recall, f1 = round(acc, 4), round(pcs, 4), round(recall, 4), round(f1, 4)
#             return acc, pcs, recall, f1 

#         acc_lst = []
#         pcs_lst = []
#         recall_lst = []
#         f1_lst = []

#         for pred_item, truth_item, mask_item in zip(pred_label, gold_label, pred_mask):
#             tmp_acc, tmp_pcs, tmp_recall, tmp_f1 = compute_performance(pred_item, truth_item, \
#                 mask_item, label_list, dims=1, macro=True)
#             if tmp_acc == 0.0 and tmp_pcs == 0 and tmp_recall == 0 and tmp_f1 == 0:
#                 continue 
#             acc_lst.append(tmp_acc)
#             pcs_lst.append(tmp_pcs)
#             recall_lst.append(tmp_recall)
#             f1_lst.append(tmp_f1)

#         aveg_acc = round(sum(acc_lst)/len(acc_lst), 4)
#         aveg_pcs = round(sum(pcs_lst)/len(pcs_lst), 4)
#         aveg_recall = round(sum(recall_lst)/len(recall_lst), 4)
#         aveg_f1 = round(sum(f1_lst)/len(f1_lst), 4)

#         return aveg_acc, aveg_pcs, aveg_recall, aveg_f1 

def compute_performance(pred_label, gold_label, label_list):
    start_label = split_index(label_list)
    pred_entities = extract_entities(pred_label, start_label = start_label)
    truth_entities = extract_entities(gold_label, start_label = start_label)
    num_true = len(truth_entities)
    num_extraction = len(pred_entities)

    num_true_positive = 0 
    for entity_idx in pred_entities.keys():
        try:
            if truth_entities[entity_idx] == pred_entities[entity_idx]:
                num_true_positive += 1 
        except:
            pass 

    dict_match = list(filter(lambda x: x[0] == x[1], zip(pred_label, gold_label)))
    acc = len(dict_match) / float(len(gold_label))

    if num_extraction != 0:
        pcs = num_true_positive / float(num_extraction)
    else:
        pcs = 0 

    if num_true != 0:
        recall = num_true_positive / float(num_true)
    else:
        recall = 0 

    if pcs + recall != 0 :
        f1 = 2 * pcs * recall / (pcs + recall)
    else:
        f1 = 0 

    if num_extraction == 0 and num_true == 0:
        acc, pcs, recall, f1 = 0, 0, 0, 0 
    acc, pcs, recall, f1 = round(acc, 4), round(pcs, 4), round(recall, 4), round(f1, 4)

    return acc, pcs, recall, f1 

def eval_ner(pred, gold,label_list):
    print('Evaluating...')
    eval_dict = {}    # value=[#match, #pred, #gold]
    for p_1sent, g_1sent in zip(pred, gold):
        in_correct_chunk = False
        last_pair = ['^', '$']
        for p, g in zip(p_1sent, g_1sent):
            p = label_list[p]
            g = label_list[g]
            tp = p.split('-')
            tg = g.split('-')
            if len(tp) == 2:
                if tp[1] not in eval_dict:
                    eval_dict[tp[1]] = [0]*3
                if tp[0] == 'B' or tp[0] == 'S':
                    eval_dict[tp[1]][1] += 1
            if len(tg) == 2:
                if tg[1] not in eval_dict:
                    eval_dict[tg[1]] = [0]*3 
                if tg[0] == 'B' or tg[0] == 'S':
                    eval_dict[tg[1]][2] += 1
        
            if p != g or len(tp) == 1:
                if in_correct_chunk and tp[0] != 'I' and tg[0] != 'I' and tp[0] != 'E' and tg[0] != 'E':
                    assert last_pair[0] == last_pair[1]
                    eval_dict[last_pair[0]][0] += 1
                in_correct_chunk = False
                last_pair = ['^', '$'] 
            else:
                if tg[0] == 'B' or tg[0] == 'S':
                    if in_correct_chunk:
                        assert (last_pair[0] == last_pair[1])
                        eval_dict[last_pair[0]][0] += 1
                    last_pair = [tp[-1], tg[-1]]
                if tg[0] == 'B':
                    in_correct_chunk = True
                if tg[0] == 'S':
                    eval_dict[last_pair[0]][0] += 1
                    in_correct_chunk = False
        if in_correct_chunk:
            assert last_pair[0] == last_pair[1]
            eval_dict[last_pair[0]][0] += 1
    agg_measure = [0.0]*3
    agg_counts = [0]*3
    for k, v in eval_dict.items():
        agg_counts = [sum(x) for x in zip(agg_counts, v)]
        prec = float(v[0])/v[1] if v[1] != 0 else 0.0 
        recall = float(v[0])/v[2] if v[2] != 0 else 0.0
        F1 = 2*prec*recall/(prec+recall) if prec != 0 and recall != 0 else 0.0
        agg_measure[0] += prec
        agg_measure[1] += recall
        agg_measure[2] += F1
        # print(k+':', v[0], '\t', v[1], '\t', v[2], '\t', prec, '\t', recall, '\t', F1)
    agg_measure = [v/len(eval_dict) if len(eval_dict) != 0 else 0.0 for v in agg_measure]
    # print('Macro average:', '\t', agg_measure[0], '\t', agg_measure[1], '\t', agg_measure[2])
    prec = float(agg_counts[0])/agg_counts[1] if agg_counts[1] != 0 else 0.0
    recall = float(agg_counts[0])/agg_counts[2] if agg_counts[2] != 0 else 0.0
    F1 = 2*prec*recall/(prec+recall) if prec != 0 and recall != 0 else 0.0
    # print('Micro average:', agg_counts[0], '\t', agg_counts[1], '\t', agg_counts[2], '\t', prec, '\t', recall, '\t', F1) 
    return 0, prec, recall, F1 # {'p': prec, 'r': recall, 'f1': F1}