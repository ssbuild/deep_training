# @Time    : 2022/11/11 19:03
# @Author  : tk
# @FileName: pointer.py
from seqmetric.metrics import pt_class_report


def metric_for_pointer(trues_all,preds_all,id2label):
    y_trues = {
        label: [] for i, label in id2label.items()
    }
    y_preds = {
        label: [] for i, label in id2label.items()
    }

    for idx in range(len(preds_all)):
        one_trues = {
            label: [] for i, label in id2label.items()
        }
        one_preds = {
            label: [] for i, label in id2label.items()
        }

        pred = preds_all[idx]
        true = trues_all[idx]

        for l, s, e in true:
            str_label = id2label[l]
            one_trues[str_label].append((l, s, e))

        for l, s, e in pred:
            str_label = id2label[l]
            one_preds[str_label].append((l, s, e))


        for k, v in y_trues.items():
            v.append(one_trues[k])
        for k, v in y_preds.items():
            v.append(one_preds[k])

    str_report, f1 = pt_class_report(y_trues, y_preds, average='micro')
    return f1,str_report