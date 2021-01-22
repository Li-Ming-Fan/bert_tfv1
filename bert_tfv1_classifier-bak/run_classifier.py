
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import time
import logging
import collections

import csv
import pandas as pd

import tensorflow as tf

import tokenization
from modeling import BertConfig

from model_wrapper import create_logger, get_logger
from model_wrapper import ModelForTrain, ModelForPrediction, create_optimizer
from model_wrapper import train_op_dict_creator_simple
from model_wrapper import train_op_dict_creator_multibatch
from metrics_np import Metrics

from model_classifier import model_creator

#
task_name = "sim"
data_dir = "../data_sim"
output_dir = "../model_sim"
#
bert_config_file = "../chinese_L-12_H-768_A-12/bert_config.json"
vocab_file = "../chinese_L-12_H-768_A-12/vocab.txt"
init_checkpoint = "../chinese_L-12_H-768_A-12/bert_model.ckpt"
#
max_seq_length = 20
save_checkpoints_steps = 10
#
with_multibatch = 0
grad_accum_steps = 4
train_batch_size = 32
#
do_train = True
do_eval = True
do_predict = False
#
gpu_id = "0"
#


## 
flags = tf.flags
FLAGS = flags.FLAGS


## Required parameters
flags.DEFINE_string("task_name", task_name,
    "The name of the task to train.")

flags.DEFINE_string("data_dir", data_dir,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string("output_dir", output_dir,
    "The output directory where the model checkpoints will be written.")


flags.DEFINE_string("bert_config_file", bert_config_file,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", vocab_file,
                    "The vocabulary file that the BERT model was trained on.")


## Other parameters

flags.DEFINE_string("init_checkpoint", init_checkpoint,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool("do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("max_seq_length", max_seq_length,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

#
flags.DEFINE_bool("with_multibatch", with_multibatch,
    "Whether to use multibatch.")

flags.DEFINE_integer("grad_accum_steps", grad_accum_steps,
    "grad_accum_steps.")

#
flags.DEFINE_bool("do_train", do_train,
    "Whether to run training.")

flags.DEFINE_bool("do_eval", do_eval,
    "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", do_predict,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_string("gpu_id", gpu_id, "gpu_id")

#
flags.DEFINE_integer("train_batch_size", train_batch_size, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 10,
                   "Total number of training epochs to perform.")

#
flags.DEFINE_float("warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", save_checkpoints_steps,
                     "How often to save the model checkpoint.")
#


#
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
#
class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """
#
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example
#
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

#
class SimProcessor(DataProcessor):
    """Processor for the Sim task"""

    # read csv
    # def get_train_examples(self, data_dir):
    #   file_path = os.path.join(data_dir, 'train.csv')
    #   train_df = pd.read_csv(file_path, encoding='utf-8')
    #   train_data = []
    #   for index, train in enumerate(train_df.values):
    #       guid = 'train-%d' % index
    #       text_a = tokenization.convert_to_unicode(str(train[0]))
    #       # text_b = tokenization.convert_to_unicode(str(train[1]))
    #       label = str(train[1])
    #       train_data.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    #   return train_data

    # read txt
    #返回InputExample类组成的list
    #text_a是一串字符串，text_b则是另一串字符串。在进行后续输入处理后(BERT代码中已包含，不需要自己完成)
    # text_a和text_b将组合成[CLS] text_a [SEP] text_b [SEP]的形式传入模型
    def get_train_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'train_sentiment.txt')
        f = open(file_path, 'r', encoding='utf-8')
        train_data = []
        index = 0
        for line in f.readlines():
            guid = 'train-%d' % index#参数guid是用来区分每个example的
            line = line.replace("\n", "").split("\t")
            text_a = tokenization.convert_to_unicode(str(line[1]))#要分类的文本
            label = str(line[2])#文本对应的情感类别
            train_data.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))#加入到InputExample列表中
            index += 1
        return train_data

    # csv
    # def get_dev_examples(self, data_dir):
    #   file_path = os.path.join(data_dir, 'dev.csv')
    #   dev_df = pd.read_csv(file_path, encoding='utf-8')
    #   dev_data = []
    #   for index, dev in enumerate(dev_df.values):
    #       guid = 'dev-%d' % index
    #       text_a = tokenization.convert_to_unicode(str(dev[0]))
    #       # text_b = tokenization.convert_to_unicode(str(dev[1]))
    #       label = str(dev[1])
    #       dev_data.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    #   return dev_data

    def get_dev_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'test_sentiment.txt')
        f = open(file_path, 'r', encoding='utf-8')
        dev_data = []
        index = 0
        for line in f.readlines():
            guid = 'dev-%d' % index
            line = line.replace("\n", "").split("\t")
            text_a = tokenization.convert_to_unicode(str(line[1]))
            label = str(line[2])
            dev_data.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
            index += 1
        return dev_data

    def get_test_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'test.csv')
        test_df = pd.read_csv(file_path, encoding='utf-8')
        test_data = []
        for index, test in enumerate(test_df.values):
            guid = 'test-%d' % index
            text_a = tokenization.convert_to_unicode(str(test[0]))
            # text_b = tokenization.convert_to_unicode(str(test[1]))
            label = str(test[1])
            test_data.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return test_data

    def get_labels(self):
        return ['0', '1', '2']

    def get_class_weights(self):
        return [1, 1, 3]
#

#
def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer, logger=None):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5 and logger:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature
#
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
    #
#

#
def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, logger=None):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and logger:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer, logger)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()
#
def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d
    #
    return input_fn
    #
#


#
def main(args):
    """
    """
    str_datetime = time.strftime("%Y_%m_%d_%H_%M_%S")
    log_path = "log_%s_%s.txt" % (FLAGS.task_name, str_datetime)
    log_tag = log_path
    logger = create_logger(log_tag, log_path)

    settings_all = {}
    settings_all["logger"] = logger

    #
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id

    processors = {
        "sim": SimProcessor,
    }

    #
    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    try:
        class_weights = processor.get_class_weights()
    except:
        class_weights = None

    num_labels = len(label_list)

    upper_settings = {}
    upper_settings["num_labels"] = num_labels
    upper_settings["class_weights"] = class_weights

    #
    bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    #
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    #
    settings_all["bert_config"] = bert_config
    settings_all["upper_settings"] = upper_settings
    settings_all["model_dir"] = FLAGS.output_dir
    settings_all["model_name"] = FLAGS.task_name

    inputs_dict = {}
    inputs_dict["input_ids"] = "input_ids:0"
    inputs_dict["input_mask"] = "input_mask:0"
    inputs_dict["segment_ids"] = "segment_ids:0"
    #
    outputs_dict = {}
    outputs_dict["probabilities"] = "loss/Softmax:0"
    #
    list_pb_save_names = [ "loss/Softmax" ]
    pb_filepath = os.path.join(FLAGS.output_dir, "model_%s.pb" % FLAGS.task_name)

    #
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, logger)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", FLAGS.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        #
        if FLAGS.with_multibatch:
            train_op_dict_creator = train_op_dict_creator_multibatch
            settings_all["grad_accum_steps"] = FLAGS.grad_accum_steps
            settings_all["batch_size"] = FLAGS.train_batch_size // FLAGS.grad_accum_steps
        else:
            train_op_dict_creator = train_op_dict_creator_simple
            settings_all["batch_size"] = FLAGS.train_batch_size
        #
        settings_all["max_train_steps"] = num_train_steps
        settings_all["save_checkpoints_steps"] = FLAGS.save_checkpoints_steps
        #

        # file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder)
        dataset_creator = file_based_input_fn_builder(
            train_file, FLAGS.max_seq_length, True, True)

        model = ModelForTrain(settings_all)
        outputs = model.create_model_graph_and_sess(
            dataset_creator, model_creator, settings_all)

        if FLAGS.init_checkpoint and FLAGS.init_checkpoint != "None":
            model.load_vars_from_ckpt(FLAGS.init_checkpoint)

        loss_train = outputs[0]

        def optimizer_creator(loss_tensor, settings):
            learning_rate=FLAGS.learning_rate,
            opt = create_optimizer(loss_tensor, init_lr=learning_rate,
                num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps)
            return opt

        train_op_dict = model.create_train_op_dict(optimizer_creator,
                train_op_dict_creator, loss_train, settings_all)
        
        model.train(train_op_dict, settings_all)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file, logger)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        logger.info("  Batch size = %d", FLAGS.eval_batch_size)

        # file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder)
        dataset_creator = file_based_input_fn_builder(
            eval_file, FLAGS.max_seq_length, False, False)

        model = ModelForPrediction(settings_all)
        model.create_model_graph_and_sess(model_creator, settings_all)
        model.load_vars_from_ckpt(FLAGS.output_dir)

        model.set_inputs_outputs_pred(inputs_dict, outputs_dict)
        model.save_pb_file(list_pb_save_names, pb_filepath)

        graph, sess = model.get_model_graph_and_sess()
        settings_all["batch_size"] = FLAGS.eval_batch_size

        list_labels = []
        list_pred_logits = []

        with graph.as_default():
            dataset = dataset_creator(settings_all)
            data_iter = dataset.make_one_shot_iterator()
            batch_data_tensor = data_iter.get_next()
            #
            try:
                while True:
                    batch_data = sess.run(batch_data_tensor)
                    output = model.predict(batch_data)
                    # print(batch_data)
                    # print(output)
                    #
                    labels = batch_data["label_ids"].tolist()
                    probs = output["probabilities"].tolist()
                    #
                    list_labels.extend(labels)
                    list_pred_logits.extend(probs)
                    #
            except:
                print("eval finished.")
            #

        #
        acc = Metrics.accuracy(list_pred_logits, list_labels)
        macro_f1 = Metrics.macro_f1(list_pred_logits, list_labels,
                list_classes=list(range(num_labels)))
        #
        result = {}
        result["accuracy"] = acc
        result["macro_f1"] = macro_f1
        #
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("%s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        #

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file, logger)

        logger.info("***** Running prediction*****")
        logger.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        logger.info("  Batch size = %d", FLAGS.predict_batch_size)

        # file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder)
        dataset_creator = file_based_input_fn_builder(
            predict_file, FLAGS.max_seq_length, False, False)

        model = ModelForPrediction(settings_all)
        model.create_model_graph_and_sess(model_creator, settings_all)
        model.load_vars_from_ckpt(FLAGS.output_dir)

        model.set_inputs_outputs_pred(inputs_dict, outputs_dict)
        model.save_pb_file(list_pb_save_names, pb_filepath)

        graph, sess = model.get_model_graph_and_sess()
        settings_all["batch_size"] = FLAGS.eval_batch_size

        list_batches_with_pred = []

        with graph.as_default():
            dataset = dataset_creator(settings_all)
            data_iter = dataset.make_one_shot_iterator()
            batch_data_tensor = data_iter.get_next()
            #
            try:
                while True:
                    batch_data = sess.run(batch_data_tensor)
                    output = model.predict(batch_data)
                    # print(batch_data)
                    # print(output)
                    #
                    # labels = batch_data["label_ids"].tolist()
                    probs = output["probabilities"]
                    #
                    batch_data["pred_probs"] = probs
                    #
                    list_batches_with_pred.append(batch_data)
                    #
            except:
                print("prediction finished.")
            #
        #
        output_file = os.path.join(FLAGS.output_dir, "pred_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            for item in list_batches_with_pred:
                pass
            #
            for key in sorted(result.keys()):
                logger.info("%s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        #
#

if __name__ == "__main__":
    """
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    """
    tf.app.run()
