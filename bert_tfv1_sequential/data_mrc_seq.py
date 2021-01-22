

import os
import random

from tokenization import FullTokenizer


#
dir_data = "../data"
#
filenames_raw = [
    "data_excavation_labelled.txt",
]
#
filename_labelled_all = "data_excavation_labelled_all.txt"
filename_bio_all = "data_excavation_bio_all.txt"
#
dataset_dict = {
    "train": "examples_train.txt",
    "valid": "examples_valid.txt",
    "test": "examples_valid.txt",
}
#
vocab_file = "../chinese_L-12_H-768_A-12/vocab.txt"
#


#
labels_map = {
    "B-ENT": "[be]",
    "I-ENT": "[ee]",
    "B-DIR": "[bd]",
    "I-DIR": "[ed]"
}
#
label_types = [
    "[be]", "[ee]",
    "[bd]", "[ed]"
]
#
def extract_seq_labels(line, label_types):
    """
    """
    list_results = []
    text_pre = ""
    text_post = line
    max_len = len(line)
    str_next = "[b"
    idx_prev = -1
    #
    while True:
        #
        list_posi = []
        len_pre = len(text_pre)
        #
        for item in label_types:
            posi = text_post.find(item)
            if posi < 0:
                list_posi.append(max_len)
            else:
                list_posi.append(posi)
            #
        #
        min_posi = min(list_posi)
        #
        if min_posi == max_len:
            break
        #
        idx_posi = list_posi.index(min_posi)
        label_str = label_types[idx_posi]
        #
        if not label_str.startswith(str_next):
            print("not label_str.startswith(str_next): %s, %s" % (label_str, str_next))
            return "", []
        #
        if str_next == "[b":
            str_next = "[e"
            idx_prev = idx_posi
        else:
            str_next = "[b"
            #
            if idx_posi != idx_prev + 1:
                print("idx_posi != idx_prev + 1, with label_str: %s" % label_str)
                return "", []
            #
        #
        text_pre = text_pre + text_post[0:min_posi]
        text_post = text_post[min_posi + len(label_str):]
        #
        list_results.append( (min_posi + len_pre, label_str) )
        #
    #
    text_pre = text_pre + text_post
    #
    return text_pre, list_results
    #
#
def load_data_mrc_seq_labelled(file_path, label_types):
    """
    """
    with open(file_path, "r", encoding="utf-8") as fp:
        lines = fp.readlines()
    #
    list_data = []
    set_texts = set()
    #
    text_a = ""
    text_b = ""
    labels = []
    #
    flag_content = 0
    #
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        #
        if line.startswith("text_a:"):
            if flag_content != 0:
                print("warning, flag_content != 0")
                text_a = ""
                text_b = ""
                labels = []
            #
            line_text_a = line[7:].strip()
            line_text_a = "".join(line_text_a.split())
            #
            flag_content = 1
            #
        elif line.startswith("text_b:"):
            if flag_content != 1:
                print("warning, flag_content != 1")
                continue
            #
            line_text_b = line[7:].strip()
            line_text_b = "".join(line_text_b.split())
            #
            flag_content = 0
            #
            # parse
            text_b = line_text_b
            text_a, labels = extract_seq_labels(line_text_a, label_types)
            #
            if len(labels) == 0:
                print("len(labels) == 0 when parsing labels with line_text:")
                print(line_text_a)
                # continue
            #
            # check
            text_ab = text_a + "[SEP]" + text_b
            if text_ab in set_texts:
                print("examples with same text_a and text_b")
                print(text_a)
                print(text_b)
                continue
            else:
                set_texts.add(text_ab)
            #
            # example
            example = {}
            example["text_a"] = line_text_a
            example["text_b"] = line_text_b if len(line_text_b) else None
            # example["text_b"] = text_b if len(text_b) else None
            # example["labels"] = labels
            #
            list_data.append(example)
            #
        #
    #
    return list_data
    #
#
def write_data_mrc_seq_labelled(file_path, list_data):
    """
    """
    fp = open(file_path, "w", encoding="utf-8")
    fp.write("\n")
    for item in list_data:
        text_a = item["text_a"]
        text_b = item["text_b"]
        # labels = item["labels"]
        #
        fp.write("text_a: %s\n" % text_a)
        fp.write("text_b: %s\n\n" % text_b)
        #
    #
    fp.close()
    #
#

#
def trans_data_labelled_to_bio(data_labelled, tokenizer, labels_map):
    """
    """
    labels_map_rev = {}
    for key, value in labels_map.items():
        labels_map_rev[value] = key
    #
    list_data_bio = []
    for item in data_labelled:
        text_a = item["text_a"]
        text_b = item["text_b"]
        #
        text_a_tokens = tokenizer.tokenize(text_a)
        text_b_tokens = tokenizer.tokenize(text_b)
        #
        # print(text_a_tokens)
        # print(text_b_tokens)
        #
        tokens_a = []
        labels_bio = []
        #
        num_tokens_a_all = len(text_a_tokens)
        idx = 0
        label_next = "O"
        flag_begin = 0
        #
        while idx < num_tokens_a_all:
            token = text_a_tokens[idx]
            if token == "[":
                if idx + 2 < num_tokens_a_all:
                    str_join = "".join(text_a_tokens[idx:idx+3])
                    # print(str_join)
                    if str_join in labels_map_rev.keys():
                        pass
                    else:
                        str_join = None
                    #
                #
                if str_join:
                    idx += 3
                    label_next = labels_map_rev[str_join]
                    #
                    if label_next.startswith("I-"):
                        label_next = "O"
                        flag_begin = 0
                    else:
                        flag_begin = 1
                    #
                    continue
                else:
                    pass
            #
            tokens_a.append(token)
            labels_bio.append(label_next)
            #
            if flag_begin == 1:
                label_next = "I" + label_next[1:]
                flag_begin = 0
                #
            #
            idx += 1
            #
        #
        str_info = "len(tokens_a) != len(labels_bio), item = {}".format(item)
        if len(tokens_a) != len(labels_bio):
            print(str_info)
            continue
        #
        example = {}
        example["tokens_a"] = tokens_a
        example["tokens_b"] = text_b_tokens
        example["labels"] = labels_bio
        example["text_a"] = text_a
        example["text_b"] = text_b
        #
        list_data_bio.append(example)
        #
    #
    return list_data_bio
    #
#
def trans_data_bio_to_labelled(data_bio, labels_map):
    """
    """
    data_labelled = []
    for item in data_bio:
        tokens_a = item["tokens_a"]
        tokens_b = item["tokens_b"]
        labels = item["labels"]
        #
        if len(tokens_a) != len(labels):
            print("len(tokens_a) != len(labels)")
            print(item)
            continue
        #
        text_a_tokens_labelled = []
        prev_label = "O"
        #
        for token, label in zip(tokens_a, labels):
            if token.startswith("##"):
                token = token[2:]
            #
            if label == "O":
                if prev_label == "O":
                    text_a_tokens_labelled.append(token)
                else:  # B, I
                    prev_label = "I" + prev_label[1:]
                    label_str = labels_map[prev_label]
                    #
                    text_a_tokens_labelled.append(label_str)
                    text_a_tokens_labelled.append(token)
                    #
            elif label.startswith("I"):
                text_a_tokens_labelled.append(token)
            else: # B
                label_str = labels_map[label]
                #
                text_a_tokens_labelled.append(label_str)
                text_a_tokens_labelled.append(token)
                #
            #
            prev_label = label
            #
        #
        text_a = "".join(text_a_tokens_labelled)
        #
        text_b_tokens = []
        for token in tokens_b:
            if token.startswith("##"):
                token = token[2:]
            #
            text_b_tokens.append(token)
        #
        text_b = "".join(text_b_tokens)
        #
        example = {}
        example["text_a"] = text_a
        example["text_b"] = text_b if len(text_a) else None
        #
        data_labelled.append(example)
        #
    #
    return data_labelled
    #
#

#
def write_data_bio(file_path, data_bio):
    """
    """
    fp = open(file_path, "w", encoding="utf-8")
    fp.write("\n")
    for item in data_bio:
        tokens_a = item["tokens_a"]
        tokens_b = item["tokens_b"]
        labels = item["labels"]
        text_a = item["text_a"]
        #
        if len(tokens_a) != len(labels):
            print("len(tokens_a) != len(labels)")
            print(item)
            continue
        #
        fp.write("text_a: %s\n" % text_a)
        #
        for token, label in zip(tokens_a, labels):
            fp.write("%s %s\n" % (token, label))
        #
        fp.write("tokens_b: %s\n\n" % (" ".join(tokens_b)))
        #
    #
    fp.close()
    #
#
def load_data_bio(file_path):
    """
    """
    fp = open(file_path, "r", encoding="utf-8")
    lines = fp.readlines()
    fp.close()
    #
    data_bio = []
    #
    idx = 0
    num_lines = len(lines)
    #
    list_pairs = []
    text_a = ""
    #
    while idx < num_lines:
        line = lines[idx].strip()
        if len(line) == 0:
            idx += 1
            continue
        #
        line = " ".join(line.split())
        #
        if line.startswith("tokens_b:"):
            tokens_a, labels = list(zip(*list_pairs))
            tokens_b = line.split()[1:]
            #
            example = {}
            example["tokens_a"] = tokens_a
            example["tokens_b"] = tokens_b
            example["labels"] = labels
            #
            data_bio.append(example)
            #
            idx += 1
            list_pairs = []
            text_a = ""
            #
            continue
            #
        elif line.startswith("text_a:"):
            #
            text_a = line[7:].strip()
            #
            idx += 1
            continue
            #
        #
        str_arr = line.split()
        list_pairs.append( (str_arr[0], str_arr[1]) )
        #
        idx += 1
        #
    #
    return data_bio
    #
#

#
def parse_predicted_bio_result(batch_input_tokens, batch_pred_tags):
    """
    """
    batch_spans = []
    #
    for eid in range(len(batch_input_tokens)):
        tokens = batch_input_tokens[eid]
        preds = batch_pred_tags[eid]
        #
        prev_tag = "O"
        span_posi_tag = []
        #
        spans_posi_tag = []
        #
        for tid in range(len(preds)):
            #
            curr_tag = preds[tid]
            #
            if curr_tag.startswith("O"):
                if prev_tag.startswith("O"):   # OO
                    pass
                else:         # BO, IO
                    if len(span_posi_tag):  
                        spans_posi_tag.append(span_posi_tag)
                    #
                #
                span_posi_tag = []
                prev_tag = "O"
                #
            elif curr_tag.startswith("B"):
                if prev_tag.startswith("O"):   # OB
                    pass
                else:       # IB, BB
                    if len(span_posi_tag):  
                        spans_posi_tag.append(span_posi_tag)
                    #
                #
                span_posi_tag = [ (tid, curr_tag) ]
                prev_tag = curr_tag
                #
            else:    # I
                if prev_tag.startswith("O"):   # OI, error
                    if tid >= 1:
                        span_posi_tag = [ (tid-1, "B" + curr_tag[1:]) ]
                    #
                else:                      # II, BI
                    pass
                #
                span_posi_tag.append( (tid, curr_tag) )
                prev_tag = curr_tag
                #
            #
        #
        list_spans = []
        #
        for item in spans_posi_tag:
            posi_s = item[0][0]
            posi_e = item[-1][0] + 1
            span_type = item[0][1][2:]
            #
            span_str = "".join(tokens[posi_s:posi_e])
            #
            list_spans.append( (span_str, span_type, posi_s) )
            #
        #
        batch_spans.append(list_spans)
        #
    #
    return batch_spans
    #
#



#
if __name__ == "__main__":
    """
    """
    data_labelled = []
    for item in filenames_raw:
        file_path = os.path.join(dir_data, item)
        list_curr = load_data_mrc_seq_labelled(file_path, label_types)
        data_labelled.extend(list_curr)
    #
    print("num_examples_all: %d" % len(data_labelled))
    # print(data_labelled)
    #
    file_path_all = os.path.join(dir_data, filename_labelled_all)
    # write_data_mrc_seq_labelled(file_path_all, data_labelled)
    #
    ## trans
    #
    tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    #
    data_bio = trans_data_labelled_to_bio(data_labelled, tokenizer, labels_map)
    # print(data_bio)
    #
    file_path_bio_all = os.path.join(dir_data, filename_bio_all)
    # write_data_bio(file_path_bio_all, data_bio)
    #
    ## split
    #
    random.shuffle(data_bio)
    #
    num_all = len(data_bio)
    num_train = int(num_all * 0.8)
    num_valid = num_all - num_train
    #
    list_train = data_bio[0:num_train]
    list_valid=  data_bio[num_train:]
    #
    file_path_train = os.path.join(dir_data, dataset_dict["train"])
    file_path_valid = os.path.join(dir_data, dataset_dict["valid"])
    #
    write_data_bio(file_path_train, list_train)
    write_data_bio(file_path_valid, list_valid)
    #
    print("num_all, num_train, num_valid: %d, %d, %d" % (num_all, num_train, num_valid))
    #
    """
    data_bio_loadded = load_data_bio(file_path_bio_all)
    print(data_bio_loadded)
    #
    data_labelled_trans = trans_data_bio_to_labelled(data_bio, labels_map)
    print(data_labelled_trans)
    """
    #

