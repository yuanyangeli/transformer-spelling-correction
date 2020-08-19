# -*- coding: utf-8 -*-

import argparse

import torch

from beaver.data import build_dataset
from beaver.infer import parallel_beam_search
from beaver.model import NMTModel
from beaver.utils import parseopt, get_device, get_logger, calculate_bleu, Loader

parser = argparse.ArgumentParser()

parseopt.translate_opts(parser)
parseopt.model_opts(parser)

opt = parser.parse_args()

device = get_device()
logger = get_logger()

loader = Loader(opt.model_path, opt, logger)


def translate(dataset, fields, model):

    total = len(dataset.examples)
    already, hypothesis, references = 0, [], []

    for batch in dataset:
        predictions = parallel_beam_search(opt, model, batch, fields)
        hypothesis += [fields["tgt"].decode(p) for p in predictions]
        already += len(predictions)
        logger.info("Translated: %7d/%7d" % (already, total))

    origin = sorted(zip(hypothesis, dataset.seed), key=lambda t: t[1])
    hypothesis = [h for h, _ in origin]
    with open(opt.output, "w", encoding="UTF-8") as out_file:
        out_file.write("\n".join(hypothesis))
        out_file.write("\n")

    logger.info("Translation finished. ")


def main():
    logger.info("Build dataset...")
    dataset = build_dataset(opt, [opt.input, opt.input], opt.vocab, device, train=False)

    logger.info("Build model...")
    model = NMTModel.load_model(loader, dataset.fields).to(device).eval()

    logger.info("Start translation...")
    with torch.set_grad_enabled(False):
        translate(dataset, dataset.fields, model)
    pass


def compute_prf(results):
    TP = 0
    FP = 0
    FN = 0
    all_predict_true_index = []
    all_gold_index = []
    for item in results:
        src, tgt, predict = item
        #print(src,tgt,predict)
        gold_index = []
        each_true_index = []
        for i in range(len(list(src))):
            if src[i] == tgt[i]:
                continue
            else:
                gold_index.append(i)
        all_gold_index.append(gold_index)
        predict_index = []
        for i in range(len(list(src))):
            if src[i] == predict[i]:
                continue
            else:
                predict_index.append(i)

        for i in predict_index:
            if i in gold_index:
                TP += 1
                each_true_index.append(i)
            else:
                FP += 1
        for i in gold_index:
            if i in predict_index:
                continue
            else:
                FN += 1
        all_predict_true_index.append(each_true_index)
    print(TP,FP,FN)
    
    # For the detection Precision, Recall and F1
    detection_precision = TP / (TP + FP) if (TP+FP) > 0 else 0
    detection_recall = TP / (TP + FN) if (TP+FN) > 0 else 0
    detection_f1 = 2 * (detection_precision * detection_recall) / (detection_precision + detection_recall) if (detection_precision + detection_recall) > 0 else 0
   # logging.info("The detection result is precision={}, recall={} and F1={}".format(detection_precision, detection_recall, detection_f1))
    print(detection_precision,detection_recall,detection_f1)

    TP = 0
    FP = 0
    FN = 0

    for i in range(len(all_predict_true_index)):
        # we only detect those correctly detected location, which is a different from the common metrics since
        # we wanna to see the precision improve by using the confusionset
        if len(all_predict_true_index[i]) > 0:
            predict_words = []
            for j in all_predict_true_index[i]:
                predict_words.append(results[i][2][j])
                if results[i][1][j] == results[i][2][j]:
                    TP += 1
                else:
                    FP += 1
            for j in all_gold_index[i]:
                if results[i][1][j] in predict_words:
                    continue
                else:
                    FN += 1
    print(TP,FP,FN)
    # For the correction Precision, Recall and F1
    correction_precision = TP / (TP + FP) if (TP+FP) > 0 else 0
    correction_recall = TP / (TP + FN) if (TP+FN) > 0 else 0
    correction_f1 = 2 * (correction_precision * correction_recall) / (correction_precision + correction_recall) if (correction_precision + correction_recall) > 0 else 0
    #logging.info("The correction  result is precision={}, recall={} and F1={}".format(correction_precision, correction_recall, correction_f1))
    return correction_precision,correction_recall,correction_f1

    
if __name__ == '__main__':
     main()
#    src = open("test.raw.src","r",encoding="utf-8")
 #   tgt = open("test.raw.tgt","r",encoding="utf-8")
  #  pre = open("output.txt","r",encoding="utf-8")
#    sentence_list = []
#    for sentence in pre.readlines():
#        sen = "".join(sentence.split(" "))
#        sentence_list.append(sen) 
#    results = [(x.strip(" "),y.strip(" "),z.strip(" ")) for x,y,z in zip(src.readlines(),tgt.readlines(),sentence_list) if len(x.strip(" ")) ==len(y.strip(" ")) == len(z.strip(" "))]
#    print(len(results))
#    #print(results)
#    correct_score = compute_prf(results)
#    print(correct_score)


