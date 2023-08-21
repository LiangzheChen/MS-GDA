import torch
import dgl
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time
import copy
from Data_loader import Get_RecipeGraph
from model import MS_GDA
from GAUG import GAug
from test import evaluate
from loss_function import XeLoss, HLoss, get_link_prediction_loss

if __name__ == "__main__":
    device = 'cuda:0'
    dataset_path = 'data/'
    graph = Get_RecipeGraph(dataset_path)
    graph.to(device)
    print('RecipeGraph: ', graph)

    # get train/val/test mask
    train_mask = graph.nodes['recipe'].data['train_mask'].to(device)
    val_mask = graph.nodes['recipe'].data['val_mask'].to(device)
    test_mask = graph.nodes['recipe'].data['test_mask'].to(device)
    labels = graph.nodes['recipe'].data['label'].to(device)

    n_classes = 9
    labelemb = torch.zeros([labels.shape[0], int(n_classes)]).to(device)
    labelemb[train_mask] = F.one_hot(labels[train_mask].to(torch.long), num_classes=n_classes).float().squeeze(1)

    print('train_mask: ', train_mask.size())
    print('val_mask: ', val_mask.size())
    print('test_mask: ', test_mask.size())
    print('labels: ', labels.size())

    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze()
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
    all_idx = torch.cat([train_idx, val_idx, test_idx])

    sampler = dgl.dataloading.MultiLayerNeighborSampler([{('compound', 'c-i', 'ingredient'): 20,
                                                          ('ingredient', 'i-c', 'compound'): 20,
                                                          ('recipe', 'r-i', 'ingredient'): 20,
                                                          ('ingredient', 'i-r', 'recipe'): 20,
                                                          ('recipe', 'r-r', 'recipe'): 20,
                                                          ('ingredient', 'i-i', 'ingredient'): 20
                                                          }] * 2)

    train_dataloader = dgl.dataloading.NodeDataLoader(
        graph, {'recipe': all_idx.cpu()}, sampler,
        batch_size=64, shuffle=True, drop_last=False, num_workers=0)

    val_dataloader = dgl.dataloading.NodeDataLoader(
        graph, {'recipe': val_idx.cpu()}, sampler,
        batch_size=64, shuffle=True, drop_last=False, num_workers=0)

    test_dataloader = dgl.dataloading.NodeDataLoader(
        graph, {'recipe': test_idx.cpu()}, sampler,
        batch_size=64, shuffle=True, drop_last=False, num_workers=0)

    model = MS_GDA(dataset_path, graph, device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)

    weights_class = torch.Tensor(9).fill_(1)
    criterion = nn.CrossEntropyLoss(weight=weights_class).to(device)
    criterion_cosine = torch.nn.CosineEmbeddingLoss(margin=0.1)
    soft_xe_loss_op = XeLoss()
    h_loss_op = HLoss()

    print('START TRAINING')
    delta_G_e = 0.2
    best_f1 = 0
    for epoch in range(2000):
        train_start = time.time()
        epoch_loss = 0
        cosine_epoch_loss = 0
        epoch_precision = 0
        epoch_recall = 0
        epoch_f1 = 0
        iteration_cnt = 0

        for batch, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            model.train()
            blocks1 = copy.deepcopy(blocks)
            blocks0 = copy.deepcopy(blocks)
            # batch_train_ids
            batch_train_ids = output_nodes['recipe']
            batch_train_ids = train_mask[batch_train_ids.to(device)]

            ###GAUG###
            blocks1[0] = GAug(blocks1[0], delta_G_e)
            delta_G_e_aug = 0.2
            blocks[0] = GAug(blocks[0], delta_G_e_aug)
            delta_G_e = 0.2

            blocks0 = [b.to(device) for b in blocks0]
            blocks = [b.to(device) for b in blocks]
            blocks1 = [b.to(device) for b in blocks1]

            # input
            input_instr = blocks[0].srcdata['instr_feature']['recipe']
            input_ingredient = blocks[0].srcdata['nutrient_feature']['ingredient']
            input_compound = blocks[0].srcdata['com_feature']['compound']
            ingredient_of_dst_recipe = blocks[1].srcdata['nutrient_feature']['ingredient']
            labels = blocks[-1].dstdata['label']['recipe']

            instrids = blocks[0].srcdata['_ID']['recipe']
            la_emb = labelemb[instrids]

            inputs = [input_instr, input_ingredient, input_compound, ingredient_of_dst_recipe]

            logits0, secondToLast_ingre, last_instr, total_pos_score, total_neg_score = model(blocks0, inputs,
                                                                                              output_nodes,
                                                                                              la_emb)
            logits, _, _, _, _ = model(blocks, inputs, output_nodes, la_emb)

            logits1, _, _, _, _ = model(blocks1, inputs, output_nodes, la_emb)

            # training scores
            y_pred = np.argmax(logits.detach().cpu(), axis=1)

            # compute loss
            link_prediction_loss = get_link_prediction_loss(total_pos_score, total_neg_score)
            loss = criterion(logits[batch_train_ids], labels[batch_train_ids]) + 0.1 * link_prediction_loss

            # semi loss
            loss_KL = soft_xe_loss_op(logits, logits1)
            loss_H = h_loss_op(logits0)

            loss = loss + 0.2 * loss_H + 2.0 * loss_KL

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            iteration_cnt += 1
            # break

        epoch_loss /= iteration_cnt
        cosine_epoch_loss /= iteration_cnt
        train_end = time.strftime("%M:%S min", time.gmtime(time.time() - train_start))

        print('Epoch: {0},  Loss: {l:.4f},  Time: {t}, LR: {lr:.6f}'
              .format(epoch, l=epoch_loss, t=train_end, lr=opt.param_groups[0]['lr']))

        scheduler.step()

        # Evaluation
        # For demonstration purpose, only test set result is reported here. Please use val_dataloader for comprehensiveness.
        test_loss, test_precision, test_recall, test_f1, test_time, test_detailed_precision, test_detailed_recall, test_detailed_f1, link_prediction_test_loss \
            = evaluate(model, test_dataloader, labelemb, criterion, device)
        print('Testing: ')
        print(
            'Total Loss: {l:.4f},  Precision: {precision:.4f},  Recall: {recall:.4f},  F1: {f1:.6f},  Time: {t}, Link Loss: {link_loss:.4f}'
                .format(l=test_loss, precision=test_precision, recall=test_recall, f1=test_f1, t=test_time,
                        link_loss=link_prediction_test_loss))
        print('detailed_precision: ', [float('{:.4f}'.format(i)) for i in list(test_detailed_precision)])
        print('detailed_recall: ', [float('{:.4f}'.format(i)) for i in list(test_detailed_recall)])
        print('detailed_f1: ', [float('{:.4f}'.format(i)) for i in list(test_detailed_f1)])
        print()
        if best_f1 < test_f1:
            best_f1 = test_f1
            print('upgrate')
            torch.save(model.state_dict(), 'model_MS.pth')

    print(best_f1)