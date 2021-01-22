
import os
import numpy as np
import random

from collections import OrderedDict

import tokenization


from model_wrapper import ModelForPrediction

from model_sequential import convert_quan_to_ban
from model_sequential import convert_single_example
from model_sequential import convert_single_example_tokens
from model_sequential import convert_list_examples_to_batch

from data_mrc_seq import load_data_mrc_seq_labelled
from data_mrc_seq import trans_data_labelled_to_bio
from data_mrc_seq import load_data_bio
from data_mrc_seq import parse_predicted_bio_result

from metrics_np import Metrics


#
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#
bert_config_file = "../chinese_L-12_H-768_A-12/bert_config.json"
vocab_file = "../chinese_L-12_H-768_A-12/vocab.txt"
#
#
# model
#
dir_model = "../model_excavation"
pb_filename = "model_excavation.pb"
#
#
# inputs, outputs
#
inputs_dict = {}
inputs_dict["input_ids"] = "input_ids:0"
inputs_dict["input_mask"] = "input_mask:0"
inputs_dict["segment_ids"] = "segment_ids:0"
#
outputs_dict = {}
outputs_dict["seq_preds"] = "crf_layer/cond/Merge:0"
# outputs_dict["seq_preds"] = "crf_layer/ReverseSequence_1:0"
#
labels_idx_to_str = {
    0: "O",
    1: "B-ENT",
    2: "I-ENT"
}
#

#
def prepare_model_and_tokenizer():
    """
    """
    # load
    model_for_prediction = ModelForPrediction({})
    model_for_prediction.prepare_for_prediction_with_pb(
        os.path.join(dir_model, pb_filename),
        inputs_dict, outputs_dict )
    #
    # tokenizer
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    #
    # predict
    #
    text_a = "是的"
    text_b = "是"
    #
    list_examples = [ convert_single_example(tokenizer, text_a, text_b,
                                            max_seq_length=10) ]
    batch_dict = convert_list_examples_to_batch(list_examples)
    #
    result_dict = model_for_prediction.predict(batch_dict)
    print("test result in prepare_model_and_tokenizer():")
    print(result_dict)
    #
    return model_for_prediction, tokenizer
    #
#
def do_prediction_for_list_pairs(model, tokenizer, list_pairs,
        labels_idx_to_str, with_label=False, idx_max=10000000):
    """
    """
    list_data = []
    for item in list_pairs:
        tokens_a = tokenizer.tokenize(item[0])
        tokens_b = tokenizer.tokenize(item[1])
        labels = [0]
        #
        list_data.append( (tokens_a, tokens_b, labels) )
        #
    #
    result = do_prediction_for_list_data(model, tokenizer, list_data,
        labels_idx_to_str, with_label=with_label, idx_max=idx_max)
    #
    return result
    #
#
def do_prediction_for_list_data(model, tokenizer, list_data,
        labels_idx_to_str, with_label=False, idx_max=10000000):
    """ list_data: list of (tokens_a, tokens_b, list_labels_str)
    """
    labels_str_to_idx = {}
    for key, value in labels_idx_to_str.items():
        labels_str_to_idx[value] = key
    #

    #
    batch_size = 64
    num_batches = len(list_data) // batch_size + 1
    #
    list_batches_with_pred = []
    #
    list_inputs_idx = []
    list_labels_idx = []
    list_preds_idx = []
    #
    def get_labels_idx(list_labels_str, seq_len=0):
        if seq_len:
            return [0] * seq_len
        else:
            return [labels_str_to_idx.get(item, 0) for item in list_labels_str]
    #
    for idx in range(0, len(list_data), batch_size):
        print("curr_batch: %d / %d" % (idx / batch_size, num_batches))
        #
        items = list_data[idx: idx + batch_size]
        #
        if with_label:
            list_examples = [ convert_single_example_tokens(tokenizer, 
                item[0], item[1], get_labels_idx(item[2], 0),
                max_seq_length=256) for item in items ]
        else:
            list_examples = [ convert_single_example_tokens(tokenizer, 
                item[0], item[1], get_labels_idx(0, len(item[0])),
                max_seq_length=256) for item in items ]
        #
        batch_dict = convert_list_examples_to_batch(list_examples)
        result_dict = model.predict(batch_dict)
        #
        seq_inputs = batch_dict["input_ids"]
        seq_labels = batch_dict["label_ids"]
        seq_preds = result_dict["seq_preds"].tolist()
        #
        batch_dict["seq_labels"] = seq_preds
        list_batches_with_pred.append(batch_dict)
        #
        list_inputs_idx.extend(seq_inputs)
        list_labels_idx.extend(seq_labels)
        list_preds_idx.extend(seq_labels)
        #
        # for eid in range(len(seq_labels)):
        #     print(seq_labels[eid])
        #     print(seq_preds[eid])
        #
        if idx > idx_max: break
        #
    #
    result_scores = {}
    #
    list_inputs_str = []
    list_labels_str = []
    list_preds_str = []
    #
    for seq in list_inputs_idx:
        seq_str = tokenizer.convert_ids_to_tokens(seq)
        list_inputs_str.append(seq_str)
    #
    for seq in list_labels_idx:
        seq_str = [labels_idx_to_str[item] for item in seq]
        list_labels_str.append(seq_str)
    #
    for seq in list_preds_idx:
        seq_str = [labels_idx_to_str[item] for item in seq]
        list_preds_str.append(seq_str)
    #
    results = Metrics.seq_tagging_f1score(list_preds_str, list_labels_str)
    #
    result_scores["f1_score"] = results[0]
    result_scores["precision"] = results[1]
    result_scores["recall"] = results[2]
    #
    result_bundle = result_scores, list_inputs_str, list_labels_str, list_preds_str
    #
    return list_batches_with_pred, result_bundle
    #
#
def write_result_pred_bio(file_path, result_bundle, with_label=False, num_sample=0):
    """
    """
    result_scores, list_inputs_str, list_labels_str, list_preds_str = result_bundle
    #
    list_spans = parse_predicted_bio_result(list_inputs_str, list_preds_str)
    #
    def get_pred_str(v, with_label):
        if with_label:
            return "label: %s, prediction: %s" % (str(v[0]), str(v[1]))
        else:
            return "%s" % str(v[0])
    #
    fp = open(file_path, "w", encoding="utf-8")
    fp.write("{}".format(result_scores))
    #
    for eid in range(len(list_inputs_str)):
        fp.write("%s\n" % ("=" * 80) )
        #
        tokens = list_inputs_str[eid]
        labels = list_labels_str[eid]
        preds = list_preds_str[eid]
        spans = list_spans[eid]
        #
        for tid in range(len(preds)):
            if with_label:
                v = labels[tid], preds[tid]
            else:
                v = preds[tid], "UNK"
            #
            token = tokens[tid]
            if token == "[PAD]": break
            #
            fp.write("%s, %s\n" % (token, get_pred_str(v, with_label)))
            #
        #
        for tid_s in range(tid, len(tokens)):
            #
            token = tokens[tid_s]
            if token == "[PAD]": break
            #
            fp.write("%s, \n" % token)
            #
        #
        fp.write("pred_spans: {}\n".format(spans))
        #
    #
    fp.write("%s\n" % ("=" * 80) )
    fp.close()
    #
#

#
def get_data_from_file_bio_form(list_files):
    """
    """
    pass
#
def get_data_from_file_plain_form(list_files):
    """
    """
    pass
#
 
#              
if __name__ == "__main__":
    """
    """
    model, tokenizer = prepare_model_and_tokenizer()
    #
    # dir_data
    dir_data = "../data"
    #

    #
    list_pairs = [
        ("是", "是")
    ]
    #
    result = do_prediction_for_list_pairs(model, tokenizer, list_pairs,
        labels_idx_to_str, with_label=False, idx_max=100000)
    #
    list_batches_with_pred, result_bundle = result
    #
    file_path = os.path.join(dir_data, "temp_result.txt")
    #
    write_result_pred_bio(file_path, result_bundle, with_label=False, num_sample=0)
    #




