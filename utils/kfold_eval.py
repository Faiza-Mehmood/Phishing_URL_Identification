# !/usr/bin/env python
# coding:utf-8
"""
Tencent is pleased to support the open source community by making NeuralClassifier available.
Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the MIT License (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at
http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for thespecific language governing permissions and limitations under
the License.
"""

import sys
import torch
from torch.utils.data import DataLoader
from utils.multilabel_evaluationmetrics_final.examplebasedclassification import *
from utils.multilabel_evaluationmetrics_final.labelbasedclassification import *
from utils.multilabel_evaluationmetrics_final.examplebasedranking import *
from utils.multilabel_evaluationmetrics_final.labelbasedranking import *
from utils import util
import numpy as np
from utils.config import Config
from utils.dataset_preprocessing.classification_dataset import ClassificationDataset
from utils.dataset_preprocessing.collator import ClassificationCollator
from utils.dataset_preprocessing.collator import ClassificationType
from utils.dataset_preprocessing.collator import FastTextCollator
from utils.evaluate.classification_evaluate import \
    ClassificationEvaluator as cEvaluator
from model.classification.textrcnn import TextRCNN
from model.classification.transformer import Transformer

from model.model_util import get_optimizer
from utils.precision_test import take_values

ClassificationDataset, ClassificationCollator, FastTextCollator, cEvaluator,
TextRCNN, Transformer

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


def get_classification_model(model_name, dataset, conf):
    model = globals()[model_name](dataset, conf)
    model = model.cuda(conf.device) if conf.device.startswith("cuda") else model
    return model


def load_checkpoint(file_name, conf, model, optimizer):
    checkpoint = torch.load(file_name)
    conf.train.start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def eval(conf):
    logger = util.Logger(conf)
    model_name = conf.model_name
    dataset_name = "ClassificationDataset"
    collate_name = "FastTextCollator" if model_name == "FastText" \
        else "ClassificationCollator"

    test_dataset = globals()[dataset_name](conf, conf.data.test_json_files)
    collate_fn = globals()[collate_name](conf, len(test_dataset.label_map))
    test_data_loader = DataLoader(
        test_dataset, batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    empty_dataset = globals()[dataset_name](conf, [])
    model = get_classification_model(model_name, empty_dataset, conf)
    optimizer = get_optimizer(conf, model)
    load_checkpoint(conf.eval.model_dir, conf, model, optimizer)
    model.eval()
    is_multi = False
    if conf.task_info.label_type == ClassificationType.MULTI_LABEL:
        is_multi = True
    predict_probs = []
    standard_labels = []
    evaluator = cEvaluator(conf.eval.dir)
    for batch in test_data_loader:
        logits = model(batch)
        if not is_multi:
            result = torch.nn.functional.softmax(logits, dim=1).cpu().tolist()
        else:
            result = torch.sigmoid(logits).cpu().tolist()
        predict_probs.extend(result)
        standard_labels.extend(batch[ClassificationDataset.DOC_LABEL_LIST])

        #============================ EVALUATION API ============================================================================================
        y_test, predictions = [], []
        for i,j in zip(standard_labels,predict_probs):
            y_test.append(i)
            predictions.append(j)

        # y_test = np.asarray(y_test)
        # predictions = np.asarray(predictions)
        # a,b = take_values(predictions,y_test)
        # print(model_name)


        # print(y_test,predictions)

        actual,pred = take_values(predictions,y_test)
        print(all(len(sub) == 6 for sub in actual),all(len(sub) == 6 for sub in pred))
        # print(a,b)
        #
        # actual,pred = np.array(a),np.array(b)
        print(actual,pred)
        print("###################################ExampleBasedClassification###################################")
        print("Accuracy", accuracy(actual,pred)*100)
        print("Precision", precision(actual,pred)*100)
        print("Recall ",recall(actual, pred) * 100)
        print("F1 score ",f1_scor(actual, pred, ) * 100)
        print("Subset Accuracy ", subsetAccuracy(actual, pred) * 100)
        print("Hamming Loss     = {:.2f} %\t".format(hammingLoss(actual, pred)))
        print("f-beta           = {:.2f} %\t".format(fbeta(actual, pred) * 100))
        print("###################################LabelBasedClassification###################################")
        print("________________________________Macro____________________________________")
        print("Accuracy Macro        = {:.2f} %\t".format(accuracyMacro(actual, pred) * 100))
        print("Recall Macro          = {:.2f} %\t".format(recallMacro(actual, pred) * 100))
        print("Precision Macro = {:.2f} %\t".format(precisionMacro(actual, pred) * 100))
        print("f-beta Macro          = {:.2f} %\t".format(fbetaMacro(actual, pred)))
        print("f-1 Macro          = {:.2f} %\t".format(macroF1(actual, pred)))
        print("_________________________________Micro____________________________________")
        print("Accuracy Micro        = {:.2f} %\t".format(accuracyMicro(actual, pred) * 100))
        print("Recall Micro          = {:.2f} %\t".format(recallMicro(actual, pred) * 100))
        print("Precision Micro = {:.2f} %\t".format(precisionMicro(actual, pred) * 100))
        print("f-beta Micro          = {:.2f} %\t".format(fbetaMicro(actual, pred)))
        print("f-1 Micro          = {:.2f} %\t".format(microF1(actual, pred)))

        print("###################################LabelBasedRanking###################################")
        print("Auc Macro          = {:.2f} %\t".format(AucMacro(np.array(actual), np.array(pred)) * 100))
        print("Auc Micro          = {:.2f} %\t".format(AucMicro(np.array(actual), np.array(pred)) * 100))
        print("ROC AUC: ", roc_auc_score(np.array(actual), np.array(pred)) * 100)

        print("###################################ExampleBasedRanking###################################")
        print("averagePrecision        = {:.2f} %\t".format(averagePrecision(actual, pred) * 100))
        print("coverage          = {:.2f} %\t".format(coverage(actual, pred) * 100))
        print("oneError                = {:.2f} %\t".format(oneError(actual, pred) * 100))
        print("rankingLoss:            = {:.2f} %\t".format(rankingLoss(actual, pred) * 100))
        print("###################################ExampleBasedRanking###################################")
        print("Classification report: \n", classification_report(np.array(actual), np.array(pred)))
        print("F1 micro averaging:", f1_score(np.array(actual), np.array(pred), average='micro'))
        print("F1 macro averaging:", f1_score(np.array(actual), np.array(pred), average='macro'))
        #=========================================================================================================================================
        #OLD CODE
        # print("Accuracy         = {}\t".format(accuracy(b,a)))
        # print("Precision        = {}\t".format(precision(b,a)))
        # print("Recall           = {}\t".format(recall(b,a)))
        # print("Subset Accuracy  = {}\t".format(subsetAccuracy(b,a)))
        # print("f1_score         = {}\t".format(f1_score(b, a)))
        # print("f1_beta          = {}\t".format(fbeta(b, a)))
        # print("Hamming loss     = {}\t".format(hammingLoss(b, a)))
        # arraycf = multilabel_confusion_matrix(b, a)
        # print(multilabel_confusion_matrix(b, a))

        #=========================================================================================================================================


        # cm_df = pd.DataFrame(
        #     arraycf,
        #     index=["Business", "Entertainment", "Famous-Personality","Sports","Technology"],
        #     columns=["Business", "Entertainment", "Famous-Personality","Sports","Technology"])
        #
        # # cm_df = pd.DataFrame(
        # #         arraycf, index=['agriculture', 'business', 'entertainment' ,'health-science', 'sports', 'world'],
        # #                      columns=['agriculture', 'business', 'entertainment' ,'health-science', 'sports', 'world']
        # #     )
        #
        # plt.figure(figsize=(22, 22))
        # sns.set(font_scale=1.5)  # for label size
        # sns.heatmap(cm_df, cmap=plt.cm.Blues, annot=True, annot_kws={"size": 20}, fmt='g')  # font size
        # # sns.heatmap(cm_df, annot=True)
        # # plt.title('CLE1Mdf_Model_12')
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        # plt.show()

    (_, precision_list, recall_list, fscore_list, right_list,
     predict_list, standard_list) = \
        evaluator.evaluate(
            predict_probs, standard_label_ids=standard_labels, label_map=empty_dataset.label_map,
            threshold=conf.eval.threshold, top_k=conf.eval.top_k,
            is_flat=conf.eval.is_flat, is_multi=is_multi)
    logger.warn(
        "Performance is precision: %f, "
        "recall: %f, fscore: %f,fscore macro:  %f, right: %d, predict: %d, standard: %d." % (
            precision_list[0][cEvaluator.MICRO_AVERAGE],
            recall_list[0][cEvaluator.MICRO_AVERAGE],
            fscore_list[0][cEvaluator.MICRO_AVERAGE],
            fscore_list[0][cEvaluator.MACRO_AVERAGE],
            right_list[0][cEvaluator.MICRO_AVERAGE],
            predict_list[0][cEvaluator.MICRO_AVERAGE],
            standard_list[0][cEvaluator.MICRO_AVERAGE]))
    evaluator.save()
def kfold_eval(conf):
    logger = util.Logger(conf)
    model_name = conf.model_name
    dataset_name = "ClassificationDataset"
    collate_name = "FastTextCollator" if model_name == "FastText" \
        else "ClassificationCollator"

    test_dataset = globals()[dataset_name](conf, conf.data.test_json_files)
    collate_fn = globals()[collate_name](conf, len(test_dataset.label_map))
    test_data_loader = DataLoader(
        test_dataset, batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    empty_dataset = globals()[dataset_name](conf, [])
    model = get_classification_model(model_name, empty_dataset, conf)
    optimizer = get_optimizer(conf, model)
    load_checkpoint(conf.eval.model_dir, conf, model, optimizer)
    model.eval()
    is_multi = False
    if conf.task_info.label_type == ClassificationType.MULTI_LABEL:
        is_multi = True
    predict_probs = []
    standard_labels = []
    evaluator = cEvaluator(conf.eval.dir)
    for batch in test_data_loader:
        logits = model(batch)
        if not is_multi:
            result = torch.nn.functional.softmax(logits, dim=1).cpu().tolist()
        else:
            result = torch.sigmoid(logits).cpu().tolist()
        predict_probs.extend(result)
        standard_labels.extend(batch[ClassificationDataset.DOC_LABEL_LIST])

        # ============================ EVALUATION API ============================================================================================
    y_test, predictions = [], []


    print (len(standard_labels))
    for i, j in zip(standard_labels, predict_probs):
        y_test.append(i)
        predictions.append(j)
        #
        # y_test = np.asarray(y_test)
        # predictions = np.asarray(predictions)
    pred, actual = take_values(predictions, y_test)
    # print(pred)
    actual=np.array(actual)
    pred=np.array(pred)

        # print(metrics.roc_curve(actual.ravel(), pred.ravel()))
        # fpr, tpr, _= metrics.roc_curve(actual.ravel(), pred.ravel())
        #
        # plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        # plt.legend(loc=4)
        # plt.show()
        #
        # # fpr, tpr, _ = metrics.roc_curve(np.array(actual), np.array(pred))
        # auc = metrics.roc_auc_score(actual, pred,average='macro' )
        # # plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        # # plt.legend(loc=4)
        # # plt.show()

        # a, b = take_values(predictions, y_test)
        # actual = np.array(actual)
        # pred = np.array(pred)
    # pred_file = open(
    #     "/home/nabeel/NeuralAPI_multilabel/" + model_name + "pred.txt",
    #     "a")
    # for pred1 in pred:
    #     pred_file.write(str(pred1) + "\n")
    # act_file = open(
    #     "/home/nabeel/NeuralAPI_multilabel/" + model_name + "actual.txt",
    #     "a")
    # for act in actual:
    #     act_file.write(str(act) + "\n")
    evaluation_measures={"Accuracy": accuracy(actual, pred) ,
                        "Precision": precision(actual, pred) ,
                             "Recall": recall(actual, pred) ,
                             "F1 score": f1_scor(actual, pred) ,
                             "Subset Accuracy": subsetAccuracy(actual, pred) ,
                             "Hamming Loss":hammingLoss(actual, pred),
                             "f-beta"   :fbeta(actual, pred) ,
                             "Accuracy Macro":accuracyMacro(actual, pred) ,
                             "Recall Macro":recallMacro(actual, pred) ,
                             "Precision Macro":precisionMacro(actual, pred) ,
                             "f-beta Macro":fbetaMacro(actual, pred) ,
                             "f-1 Macro":macroF1(actual, pred) ,
                             "Accuracy Micro":accuracyMicro(actual, pred) ,
                             "Recall Micro":recallMicro(actual, pred) ,
                             "Precision Micro":precisionMicro(actual, pred) ,
                             "f-beta Micro":fbetaMicro(actual, pred),
                             "f-1 Micro":microF1(actual, pred),
                             "Auc Macro":AucMacro(actual, pred) ,
                             "Auc Micro":AucMicro(actual, pred) ,
                             "ROC AUC":Auc(actual,pred),
                             "averagePrecision":averagePrecision(actual, pred) ,
                             "coverage":coverage(actual, pred),
                             "oneError":oneError(actual, pred),
                             "rankingLoss":rankingLoss(actual, pred),
                             "F1 micro averaging": f1_score(actual, pred, average='micro') ,
                             "F1 macro averaging":f1_score(actual, pred, average='macro'),
                             "F1 weighted averaging": f1_score(actual, pred, average='weighted') ,

                             }

    # (_, precision_list, recall_list, fscore_list, right_list,
    # predict_list, standard_list) = \
    # evaluator.evaluate(
    #             predict_probs, standard_label_ids=standard_labels, label_map=empty_dataset.label_map,
    #             threshold=config.eval.threshold, top_k=config.eval.top_k,
    #             is_flat=config.eval.is_flat, is_multi=is_multi)
    # logger.warn(
    #         "Performance is precision: %f, "
    #         "recall: %f, fscore: %f,fscore macro:  %f, right: %d, predict: %d, standard: %d." % (
    #             precision_list[0][cEvaluator.MICRO_AVERAGE],
    #             recall_list[0][cEvaluator.MICRO_AVERAGE],
    #             fscore_list[0][cEvaluator.MICRO_AVERAGE],
    #             fscore_list[0][cEvaluator.MACRO_AVERAGE],
    #             right_list[0][cEvaluator.MICRO_AVERAGE],
    #             predict_list[0][cEvaluator.MICRO_AVERAGE],
    #             standard_list[0][cEvaluator.MICRO_AVERAGE]))
    # with open("probss/"+model_name+"_actual.txt", 'a') as f:
    #     for item in actual:
    #         f.write("%s\n" % item)
    # with open("probss/"+model_name+"probssssdddd.txt", 'a') as f:
    #     for item in predict_probs:
    #         f.write("%s\n" % item)
    # evaluator.save()
    return evaluation_measures





if __name__ == '__main__':
    config = Config(config_file=sys.argv[1])
    eval(config)
