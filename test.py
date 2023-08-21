import torch
import time
import numpy as np
from loss_function import get_link_prediction_loss
from sklearn.metrics import precision_score, recall_score, f1_score

def get_score(y_pred, y_true):
    score = {
        "precision": precision_score(y_true, y_pred, labels=[1, 2, 3, 4, 5, 6, 7, 8], average='micro'),
        "recall": recall_score(y_true, y_pred, labels=[1, 2, 3, 4, 5, 6, 7, 8], average='micro'),
        "f1": f1_score(y_true, y_pred, labels=[1, 2, 3, 4, 5, 6, 7, 8], average='micro')
    }

    detailed_score = {
        "precision": precision_score(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8], average=None, zero_division=0),
        "recall": recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8], average=None, zero_division=0),
        "f1": f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8], average=None, zero_division=0)
    }
    return score, detailed_score

def evaluate(model, dataloader, labelemb, criterion, device):
    # print('evaluating ... ')
    evaluate_start = time.time()
    model.eval()
    total_loss = 0
    cosine_total_loss = 0
    link_prediction_total_loss = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    detailed_precision = 0
    detailed_recall = 0
    detailed_f1 = 0
    count = 0

    all_y_preds = None
    all_labels = None

    with torch.no_grad():
        for input_nodes, output_nodes, blocks in dataloader:

            blocks = [blk.to(device) for blk in blocks]

            # input
            input_instr = blocks[0].srcdata['instr_feature']['recipe']
            input_ingredient = blocks[0].srcdata['nutrient_feature']['ingredient']
            ingredient_of_dst_recipe = blocks[1].srcdata['nutrient_feature']['ingredient']
            input_compound = blocks[0].srcdata['com_feature']['compound']

            instrids = blocks[0].srcdata['_ID']['recipe']
            la_emb = labelemb[instrids]

            labels = blocks[-1].dstdata['label']['recipe']

            inputs = [input_instr, input_ingredient, input_compound, ingredient_of_dst_recipe]
            logits, secondToLast_ingre, last_instr, total_pos_score, total_neg_score = model(blocks, inputs,
                                                                                             output_nodes, la_emb)
            y_pred = np.argmax(logits.cpu(), axis=1)

            if all_y_preds is None:
                all_y_preds = y_pred
                all_labels = labels.cpu().numpy()
            else:
                all_y_preds = np.append(all_y_preds, y_pred, axis=0)
                all_labels = np.append(all_labels, labels.cpu().numpy(), axis=0)

            # Loss
            link_prediction_loss = get_link_prediction_loss(total_pos_score, total_neg_score)
            loss = criterion(logits, labels) + 0.1 * link_prediction_loss

            total_loss += loss.item()
            link_prediction_total_loss += link_prediction_loss.item()

            count += len(labels)

        score, detailed_score = get_score(all_y_preds, all_labels)
        total_precision = score['precision']
        total_recall = score['recall']
        total_f1 = score['f1']
        detailed_precision = detailed_score['precision']
        detailed_recall = detailed_score['recall']
        detailed_f1 = detailed_score['f1']

        total_loss /= count
        link_prediction_total_loss /= count
        evalutate_time = time.strftime("%M:%S min", time.gmtime(time.time() - evaluate_start))

    return total_loss, total_precision, total_recall, total_f1, evalutate_time, detailed_precision, detailed_recall, detailed_f1, link_prediction_total_loss