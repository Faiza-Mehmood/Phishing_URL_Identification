#!/usr/bin/env python
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

from colorama import Fore
from utils.eval import *
import os
import shutil
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader
from utils import util
from utils.config import Config
from utils.dataset_preprocessing.classification_dataset import ClassificationDataset
from utils.dataset_preprocessing.collator import ClassificationCollator
from utils.dataset_preprocessing.collator import FastTextCollator
from utils.dataset_preprocessing.collator import ClassificationType
from utils.evaluate.classification_evaluate import ClassificationEvaluator as cEvaluator
from model.loss import ClassificationLoss
from model.model_util import get_optimizer, get_hierar_relations

from utils.kfold_eval import kfold_eval
from utils.util import ModeType

ClassificationDataset, ClassificationCollator, FastTextCollator, ClassificationLoss, cEvaluator
TextRCNN, Transformer


def get_data_loader(dataset_name, collate_name, conf):
    """Get data loader: Train, Validate, Test
    """
    train_dataset = globals()[dataset_name](
        conf, conf.data.train_json_files, generate_dict=True)
    collate_fn = globals()[collate_name](conf, len(train_dataset.label_map))

    train_data_loader = DataLoader(
        train_dataset, batch_size=conf.train.batch_size, shuffle=True,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    validate_dataset = globals()[dataset_name](
        conf, conf.data.validate_json_files)
    validate_data_loader = DataLoader(
        validate_dataset, batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    test_dataset = globals()[dataset_name](conf, conf.data.test_json_files)
    test_data_loader = DataLoader(
        test_dataset, batch_size=conf.eval.batch_size, shuffle=False,
        num_workers=conf.data.num_worker, collate_fn=collate_fn,
        pin_memory=True)

    return train_data_loader, validate_data_loader, test_data_loader


def get_classification_model(model_name, dataset, conf):
    """Get classification model from configuration
    """
    model = globals()[model_name](dataset, conf)
    model = model.cuda(conf.device) if conf.device.startswith("cuda") else model
    return model


class ClassificationTrainer(object):
    def __init__(self, label_map, logger, evaluator, conf, loss_fn):
        self.label_map = label_map
        self.logger = logger
        self.evaluator = evaluator
        self.conf = conf
        self.loss_fn = loss_fn
        if self.conf.task_info.hierarchical:
            self.hierar_relations = get_hierar_relations(
                self.conf.task_info.hierar_taxonomy, label_map)

    def train(self, data_loader, model, optimizer, stage, epoch):
        model.update_lr(optimizer, epoch)
        model.train()
        return self.run(data_loader, model, optimizer, stage, epoch,
                        ModeType.TRAIN)

    def eval(self, data_loader, model, optimizer, stage, epoch):
        model.eval()
        return self.run(data_loader, model, optimizer, stage, epoch)

    def run(self, data_loader, model, optimizer, stage,
            epoch, mode=ModeType.EVAL):
        is_multi = False
        # multi-label classifcation
        if self.conf.task_info.label_type == ClassificationType.MULTI_LABEL:
            is_multi = True
        predict_probs = []
        standard_labels = []
        num_batch = data_loader.__len__()
        total_loss = 0.
        for batch in data_loader:
            logits = model(batch)
            if self.conf.task_info.hierarchical:
                linear_paras = model.linear.weight
                is_hierar = True
                used_argvs = (self.conf.task_info.hierar_penalty, linear_paras, self.hierar_relations)
                loss = self.loss_fn(
                    logits,
                    batch[ClassificationDataset.DOC_LABEL].to(self.conf.device),
                    is_hierar,
                    is_multi,
                    *used_argvs)
            else:  # flat classification
                loss = self.loss_fn(
                    logits,
                    batch[ClassificationDataset.DOC_LABEL].to(self.conf.device),
                    False,
                    is_multi)
            if mode == ModeType.TRAIN:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                continue
            total_loss += loss.item()
            if not is_multi:
                result = torch.nn.functional.softmax(logits, dim=1).cpu().tolist()
            else:
                result = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(result)
            standard_labels.extend(batch[ClassificationDataset.DOC_LABEL_LIST])
        if mode == ModeType.EVAL:
            total_loss = total_loss / num_batch
            (_, precision_list, recall_list, fscore_list, right_list,
             predict_list, standard_list) = \
                self.evaluator.evaluate(
                    predict_probs, standard_label_ids=standard_labels, label_map=self.label_map,
                    threshold=self.conf.eval.threshold, top_k=self.conf.eval.top_k,
                    is_flat=self.conf.eval.is_flat, is_multi=is_multi)

            self.logger.warn(
                "%s performance at epoch %d is precision: %f, "
                "recall: %f, fscore: %f, macro-fscore: %f, right: %d, predict: %d, standard: %d.\n"
                "Loss is: %f." % (
                    stage, epoch, precision_list[0][cEvaluator.MICRO_AVERAGE],
                    recall_list[0][cEvaluator.MICRO_AVERAGE],
                    fscore_list[0][cEvaluator.MICRO_AVERAGE],
                    fscore_list[0][cEvaluator.MACRO_AVERAGE],
                    right_list[0][cEvaluator.MICRO_AVERAGE],
                    predict_list[0][cEvaluator.MICRO_AVERAGE],
                    standard_list[0][cEvaluator.MICRO_AVERAGE], total_loss))
            return precision_list[0][cEvaluator.MICRO_AVERAGE], precision_list[0][cEvaluator.MACRO_AVERAGE],recall_list[0][cEvaluator.MICRO_AVERAGE],recall_list[0][cEvaluator.MACRO_AVERAGE], fscore_list[0][cEvaluator.MICRO_AVERAGE],fscore_list[0][cEvaluator.MACRO_AVERAGE]


def load_checkpoint(file_name, conf, model, optimizer):
    checkpoint = torch.load(file_name)
    conf.train.start_epoch = checkpoint["epoch"]
    best_performance = checkpoint["best_performance"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return best_performance


def save_checkpoint(state, file_prefix):
    file_name = file_prefix + "_" + str(state["epoch"])
    torch.save(state, file_name)


def train(conf, dir, fold, algo_name,final_results_path):
    sum_precision = []
    sum_micro_precision = []
    sum_macro_precision = []
    sum_recall = []
    sum_micro_recall = []
    sum_macro_recall = []
    sum_accuracy = []
    sum_micro_accuracy = []
    sum_macro_accuracy = []
    sum_subset_accuracy = []
    sum_f1_scor = []
    sum_micro_fscore = []
    sum_macro_fscore = []
    sum_sk_f1_score = []
    sum_sk_micro_fscore = []
    sum_sk_macro_fscore = []
    sum_hamming_loss = []
    sum_fbeta = []
    sum_micro_fbeta = []
    sum_macro_fbeta = []
    sum_rocauc = []
    sum_micro_rocauc = []
    sum_macro_rocauc = []
    sum_averagePrecision = []
    sum_coverage = []
    sum_oneError = []
    sum_rankingLoss = []

    for i in range(fold):

       # i=2
        print("______________________Fold",i,"______________________")
        conf.data.train_json_files=[os.path.join(dir, str(i), "train.json").replace("\\", "/")]
        conf.data.test_json_files=[os.path.join(dir, str(i), "test.json").replace("\\", "/")]
        conf.data.validate_json_files=[os.path.join(dir, str(i), "valid.json").replace("\\", "/")]

        logger = util.Logger(conf)
        if not os.path.exists(conf.checkpoint_dir):
            os.makedirs(conf.checkpoint_dir)

        model_name = conf.model_name
        dataset_name = "ClassificationDataset"
        collate_name = "FastTextCollator" if model_name == "FastText" \
            else "ClassificationCollator"
        train_data_loader, validate_data_loader, test_data_loader = \
            get_data_loader(dataset_name, collate_name, conf)
        empty_dataset = globals()[dataset_name](conf, [])
        model = get_classification_model(model_name, empty_dataset, conf)
        loss_fn = globals()["ClassificationLoss"](
            label_size=len(empty_dataset.label_map), loss_type=conf.train.loss_type)
        optimizer = get_optimizer(conf, model)
        evaluator = cEvaluator(conf.eval.dir)
        trainer = globals()["ClassificationTrainer"](
            empty_dataset.label_map, logger, evaluator, conf, loss_fn)

        best_epoch = -1
        best_performance = 0
        model_file_prefix = conf.checkpoint_dir + "/" + model_name
        for epoch in range(conf.train.start_epoch,
                           conf.train.start_epoch + conf.train.num_epochs):
            start_time = time.time()
            trainer.train(train_data_loader, model, optimizer, "Train", epoch)
            trainer.eval(train_data_loader, model, optimizer, "Train", epoch)
            performance = trainer.eval(
                validate_data_loader, model, optimizer, "Validate", epoch)
            trainer.eval(test_data_loader, model, optimizer, "test", epoch)
            if performance[4] > best_performance:  # record the best model
                best_epoch = epoch
                best_performance = performance[4]
            save_checkpoint({
                'epoch': epoch,
                'model_name': model_name,
                'state_dict': model.state_dict(),
                'best_performance': best_performance,
                'optimizer': optimizer.state_dict(),
            }, model_file_prefix)
            time_used = time.time() - start_time
            logger.info("Epoch %d cost time: %d second" % (epoch, time_used))

        # best model on validateion set
        best_epoch_file_name = model_file_prefix + "_" + str(best_epoch)
        best_file_name = model_file_prefix + "_best"
        shutil.copyfile(best_epoch_file_name, best_file_name)

        load_checkpoint(model_file_prefix + "_" + str(best_epoch), conf, model,
                        optimizer)

        evaluation_measures = kfold_eval(config)

        sum_precision.append(evaluation_measures["Precision"])
        sum_micro_precision.append(evaluation_measures["Precision Micro"])
        sum_macro_precision.append(evaluation_measures["Precision Macro"])
        sum_recall.append(evaluation_measures["Recall"])
        sum_micro_recall.append(evaluation_measures["Recall Micro"])
        sum_macro_recall.append(evaluation_measures["Recall Macro"])
        sum_accuracy.append(evaluation_measures["Accuracy"])
        sum_micro_accuracy.append(evaluation_measures["Accuracy Micro"])
        sum_macro_accuracy.append(evaluation_measures["Accuracy Macro"])
        sum_subset_accuracy.append(evaluation_measures["Subset Accuracy"])
        sum_f1_scor.append(evaluation_measures["F1 score"])
        sum_micro_fscore.append(evaluation_measures["f-1 Micro"])
        sum_macro_fscore.append(evaluation_measures["f-1 Macro"])
        sum_sk_f1_score.append(evaluation_measures["F1 weighted averaging"])
        sum_sk_micro_fscore.append(evaluation_measures["F1 micro averaging"])
        sum_sk_macro_fscore.append(evaluation_measures["F1 macro averaging"])
        sum_fbeta.append(evaluation_measures["f-beta"])
        sum_micro_fbeta.append(evaluation_measures["f-beta Micro"])
        sum_macro_fbeta.append(evaluation_measures["f-beta Macro"])
        sum_rocauc.append(evaluation_measures["ROC AUC"])
        sum_micro_rocauc.append(evaluation_measures["Auc Micro"])
        sum_macro_rocauc.append(evaluation_measures["Auc Macro"])
        sum_hamming_loss.append(evaluation_measures["Hamming Loss"])
        sum_averagePrecision.append(evaluation_measures["averagePrecision"])
        sum_coverage.append(evaluation_measures["coverage"])
        sum_oneError.append(evaluation_measures["oneError"])
        sum_rankingLoss.append(evaluation_measures["rankingLoss"])

        shutil.rmtree(conf.eval.model_dir.split("/")[0])

    import csv
    with open('kfoldres.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            [algo_name, "\n"+"Precision "+ str(sum_precision),"\n"+"Recall "+ str(sum_recall),
             "\n"+"F1_score "+str(sum_f1_scor),
             "\n" + "Accuracy " + str(sum_accuracy),
             "\n"+"hamming_loss "+str(sum_hamming_loss)])
    print("_________________________________kfolds Metrics____________________________________")
    print(Fore.BLUE + "k-fold  precision", sum(sum_precision) / fold)
    print(Fore.RED + str(sum_precision))
    print(Fore.BLUE + "k-fold recall", sum(sum_recall) / fold)
    print(Fore.RED + str(sum_recall))
    print(Fore.BLUE + "k-fold fscore", sum(sum_f1_scor) / fold)
    print(Fore.RED + str(sum_f1_scor))
    print(Fore.BLUE + "k-fold Micro Fscore", sum(sum_micro_fscore) / fold)
    print(Fore.RED + str(sum_micro_fscore))
    print(Fore.BLUE + "k-fold Macro Fscore", sum(sum_macro_fscore) / fold)
    print(Fore.RED + str(sum_macro_fscore))
    print(Fore.BLUE + "k-fold Accuracy", sum(sum_accuracy) / fold)
    print(Fore.RED + str(sum_accuracy))
    print(Fore.BLUE + "k-fold SubsetAccuracy", sum(sum_subset_accuracy) / fold)
    print(Fore.RED + str(sum_subset_accuracy))
    print(Fore.BLUE + "k-fold rocauc", sum(sum_rocauc) / fold)
    print(Fore.RED + str(sum_rocauc))
    print(Fore.BLUE + "k-fold Hamming Loss", sum(sum_hamming_loss) / fold)
    print(Fore.RED + str(sum_hamming_loss))
    print(Fore.BLUE + "k-fold averagePrecision", sum(sum_averagePrecision) / fold)
    print(Fore.RED + str(sum_averagePrecision))

    results_df = pd.DataFrame()
    results_df["algo_name"] = [algo_name]
    results_df["emb_dim"] = [conf.embedding.dimension]
    results_df["bs"] = [conf.train.batch_size]
    results_df["num_epochs"] = [conf.train.num_epochs]
    results_df["lr"] = [conf.optimizer.learning_rate]
    results_df["Precision"] = [sum(sum_precision) / fold]
    results_df["Micro_Precision"] = [sum(sum_micro_precision) / fold]
    results_df["Macro_Precision"] = [sum(sum_macro_precision) / fold]
    results_df["rec"] = [sum(sum_recall) / fold]
    results_df["Micro_rec"] = [sum(sum_micro_recall) / fold]
    results_df["Macro_rec"] = [sum(sum_macro_recall) / fold]
    results_df["f1"] = [sum(sum_f1_scor) / fold]
    results_df["Micro_f1"] = [sum(sum_micro_fscore) / fold]
    results_df["Macro_f1"] = [sum(sum_macro_fscore) / fold]
    results_df["ACC"] = [sum(sum_accuracy) / fold]
    results_df["SubsetAccuracy"] = [sum(sum_subset_accuracy) / fold]
    results_df["Avg_Precision"] = [sum(sum_averagePrecision) / fold]
    results_df["rocauc"] = [sum(sum_rocauc) / fold]
    results_df["Macro rocauc"] = [sum(sum_macro_rocauc) / fold]
    results_df["Micro rocauc"] = [sum(sum_micro_rocauc) / fold]
    results_df["Hamming Loss"] = [sum(sum_hamming_loss) / fold]
    results_df["Ranking Loss"] = [sum(sum_rankingLoss) / fold]
    results_df["One Error"] = [sum(sum_oneError) / fold]
    if os.path.exists(final_results_path):
        res = pd.read_csv(final_results_path)
        results_df = pd.concat([res, results_df], ignore_index=True)
        results_df.to_csv(final_results_path, index=False)
    else:
        results_df.to_csv(final_results_path, index=False)


if __name__ == '__main__':
    config_file = "config/train_kfolds.json"
    config = Config(config_file=config_file)
    fold = config["num_folds"]
    final_results_path = config["results_saved_file_path"]
    algorithms = config["algorithms"]
    data_dir = config["data_dir"]

    for algo in algorithms:
            print("_________________________________" + algo + "____________________________________")
            os.environ['CUDA_VISIBLE_DEVICES'] = str(config.train.visible_device_list)
            torch.manual_seed(2019)
            torch.cuda.manual_seed(2019)
            config.model_name = algo
            config.eval.model_dir = config.eval.model_dir.split("/")[0] + "/" + algo + "_best"
            train(config, data_dir, fold,algo,final_results_path)


