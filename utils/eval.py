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

from utils import util
import numpy as np
from utils.config import Config
from utils.dataset.classification_dataset import ClassificationDataset
from utils.dataset.collator import ClassificationCollator
from utils.dataset.collator import ClassificationType
from utils.dataset.collator import FastTextCollator
from utils.evaluate.classification_evaluate import \
    ClassificationEvaluator as cEvaluator
from model.classification.textrcnn import TextRCNN
from model.classification.transformer import Transformer
from model.model_util import get_optimizer
from utils.precision_test import take_values
from utils.examplebasedclassification import accuracy,subsetAccuracy,precision,recall,f1_score,fbeta,hammingLoss

ClassificationDataset, ClassificationCollator, FastTextCollator, cEvaluator,
TextRCNN, Transformer


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

        y_test = np.asarray(y_test)
        predictions = np.asarray(predictions)

        a,b = take_values(predictions,y_test)
        pred_file = open(
            "/topmodelensembleresults/current_final/" + model_name + "pred.txt",
            "w")
        for pred in a:
            pred_file.write(str(pred) + "\n")
        act_file = open(
            "/topmodelensembleresults/current_final/" + model_name + "pred.txt",
            "w")
        for act in b:
            act_file.write(str(act) + "\n")

        a,b = np.array(a),np.array(b)
        print("Accuracy         = {}\t".format(accuracy(b,a)))
        print("Precision        = {}\t".format(precision(b,a)))
        print("Recall           = {}\t".format(recall(b,a)))
        print("Subset Accuracy  = {}\t".format(subsetAccuracy(b,a)))
        print("f1_score         = {}\t".format(f1_score(b, a)))
        print("f1_beta          = {}\t".format(fbeta(b, a)))
        print("Hamming loss     = {}\t".format(hammingLoss(b, a)))
        #=========================================================================================================================================
        # arraycf=multilabel_confusion_matrix(b,a)
        #print(multilabel_confusion_matrix(b,a))

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
        "recall: %f, fscore: %f, right: %d, predict: %d, standard: %d." % (
            precision_list[0][cEvaluator.MICRO_AVERAGE],
            recall_list[0][cEvaluator.MICRO_AVERAGE],
            fscore_list[0][cEvaluator.MICRO_AVERAGE],
            right_list[0][cEvaluator.MICRO_AVERAGE],
            predict_list[0][cEvaluator.MICRO_AVERAGE],
            standard_list[0][cEvaluator.MICRO_AVERAGE]))
    evaluator.save()


if __name__ == '__main__':
    config = Config(config_file=sys.argv[1])
    eval(config)
