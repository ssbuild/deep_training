# @Time    : 2022/11/11 19:03
# @Author  : tk
# @FileName: pointer.py
from seqmetric.metrics import pointer_report,spo_report,get_report_from_string

def metric_for_pointer(trues_all,preds_all,label2id,metric='micro'):
    str_report = pointer_report(trues_all,preds_all,label2id)
    report = get_report_from_string(str_report,metric=metric)
    f1 = float(report[-2])
    return f1,str_report


def metric_for_spo(trues_all, preds_all, label2id, metric='micro'):
    str_report = spo_report(trues_all, preds_all, label2id)
    report = get_report_from_string(str_report,metric=metric)
    f1 = float(report[-2])
    return f1, str_report