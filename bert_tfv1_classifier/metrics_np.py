
import numpy as np
import metrics_miulab
from sklearn import metrics

class Metrics(object):
    """
    """
    @staticmethod
    def accuracy(list_pred_logits, list_labels):
        """
        """
        pred_array = np.argmax(np.array(list_pred_logits), -1)
        label_array = np.array(list_labels)
        return (pred_array == label_array).sum() * 1.0 / len(label_array)

    @staticmethod
    def macro_f1(list_pred_logits, list_labels, list_classes=None):
        """
        """
        pred_array = np.argmax(np.array(list_pred_logits), -1)
        label_array = np.array(list_labels)
        f1 = metrics.f1_score(label_array, pred_array, labels=list_classes, average="macro")
        return f1

    @staticmethod
    def classification_scores(list_pred_logits, list_labels,
                list_classes=None, class_names=None, sample_weight=None,
                digits=2, output_dict=True):
        """
        """
        pred_array = np.argmax(np.array(list_pred_logits), -1)
        label_array = np.array(list_labels)
        """
        classification_report(y_true, y_pred, labels = None, 
                              target_names = None, sample_weight = None, digits=2)
        """
        report = metrics.classification_report(
            label_array, pred_array, labels=list_classes, target_names=class_names,
            sample_weight=sample_weight, digits=digits, output_dict=output_dict)
        #
        return report
        #

    @staticmethod
    def binary_classification_scores(list_pred_logits, list_labels, pos_label=1, beta=1.0):
        """
        """
        pred_array = np.argmax(np.array(list_pred_logits), -1)
        label_array = np.array(list_labels)
        """
        precision_score(y_true, y_pred, labels=None, pos_label=1, average=’binary’, sample_weight=None)
        recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
        """        
        precision = metrics.precision_score(label_array, pred_array, pos_label=pos_label)
        recall = metrics.recall_score(label_array, pred_array, pos_label=pos_label)
        fbeta_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        #
        return precision, recall, fbeta_score
        #

    @staticmethod
    def seq_tagging_f1score(pred_slots, correct_slots):
        """ batch of seqs, seq of str (slot labels)
        """
        return metrics_miulab.computeF1Score(correct_slots, pred_slots)

    @staticmethod
    def semantic_acc(pred_slot, real_slot, pred_intent, real_intent):
        """
        """
        total_count, correct_count = 0.0, 0.0
        for p_slot, r_slot, p_intent, r_intent in zip(pred_slot, real_slot, pred_intent, real_intent):
            total_count += 1.0
            # check intent
            if p_intent == r_intent:
                # check slot
                for p, r in zip(p_slot, r_slot):
                    if p != r:
                        # flag_right = 0
                        break
                else:
                    # flag_right = 1
                    correct_count += 1.0
                #
        #
        return 1.0 * correct_count / total_count

    
    