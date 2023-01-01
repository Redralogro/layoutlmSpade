from spade_model import RelationTagger
from transformers import AutoModel, AutoTokenizer, AutoConfig, BatchEncoding, BertModel, LayoutLMModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from dataModel.datamd_ import DpDataSet
from graph_stuff import get_strings, get_qa
import networkx as nx
from datetime import datetime
from modeling.layoutlm import LayoutlmEmbeddings
from helpers import b_loss
import mlflow
from mlflow import log_metric, log_param, log_artifacts
from loss import BboxLoss
mlflow.set_tracking_uri("http://10.10.1.37:5000")
mlflow.set_experiment("eKyC/DP")

config = AutoConfig.from_pretrained("microsoft/layoutlm-base-uncased")
model = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased").cuda()
# model =  LayoutlmEmbeddings(config).cuda()
tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

lr = 1e-4

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
trainData = DpDataSet(path='./data/processed/data_cccd.jsonl')
train_loader = DataLoader(trainData, batch_size=1,
                          shuffle=False, num_workers=4)
val_loader = DataLoader(trainData, batch_size=1, shuffle=False, num_workers=1)

reduce_size = 256
ln = nn.Linear(config.hidden_size, reduce_size).cuda()


def extend_matrix(matrix):

    matrix_s = [[0] + list(x) + [0] for x in list(matrix)]
    t_m = list(np.zeros_like(matrix_s[0]))
    _s = [t_m] + list(matrix_s) + [t_m]

    return np.array(_s, dtype='int8')
# np.array(_s)


def extend_label(label_):
    label = [[0] + list(x) + [0] for x in list(label_)]
    return np.array(label, dtype='int8')


def reduce_shape(last_hidden_state, maps):
    i = 0
    # reduce = torch.zeros(768)
    reduce = []
    for g_token in maps:
        ten = torch.zeros(reduce_size).cuda()
        for ele in g_token:
            # print(ele)
            ten += last_hidden_state[0][i]
            # i+=1
            # print(last_hidden_state[0][i])
            i += 1
        ten = ten/len(g_token)
        # print(ten)
        reduce.append(ten)
        # reduce = torch.cat((reduce,ten),-1)
    # print(np.array(reduce).shape)
    # print(reduce)
    reduce = torch.stack(reduce)
    return reduce


rel_s = RelationTagger(
    hidden_size=reduce_size,
    n_fields=3,
).cuda()
rel_g = RelationTagger(
    hidden_size=reduce_size,
    n_fields=3,
).cuda()

dropout = nn.Dropout(0.1).cuda()


def grouth_truth(text_, label_):
    graph_s = np.array(label_, dtype='int')
    # print(graph_s.shape)
    label = graph_s[0, :3, :]
    S_ = graph_s[0, 3:, :]
    G_ = graph_s[1, 3:, :]
    question_heads = [i for i, ele in enumerate(label[0]) if ele != 0]
    answer_heads = [i for i, ele in enumerate(label[1]) if ele != 0]
    header_heads = [i for i, ele in enumerate(label[2]) if ele != 0]
    # print(S_,S_.shape)
    ques = get_strings(question_heads, text_, S_)
    ans = get_strings(answer_heads, text_, S_)
    print('ground truth', ques, ans)
    resul_ = {}
    for ques_idx in question_heads:
        q_, a_ = get_qa(ques_idx, ques, ans, G_)
        resul_ = merge(resul_, {q_: a_})
        # print(get_ques_ans(ques_idx, ques,ans))
    print('Grouth truth', resul_)
    return(resul_)


loss_clss = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1]).cuda())
bbox_loss_fn = BboxLoss().cuda()
epochs = 2000
for epoch in tqdm(range(epochs)):
    model.train()

    for i, sample_batched in enumerate((train_loader)):
        input_ids = sample_batched["input_ids"].squeeze(0).cuda()
        attention_mask = sample_batched["attention_mask"].squeeze(0).cuda()
        token_type_ids = sample_batched["token_type_ids"].squeeze(0).cuda()
        bbox = sample_batched["bbox"].squeeze(0).cuda()
        ex_bboxes = bbox.squeeze(0)/1000
        print(ex_bboxes.shape)
        maps = sample_batched['maps']
        # print(input_ids.shape,attention_mask.shape,bbox.shape,token_type_ids.shape, np.array (maps).shape)
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        last_hidden_state, maps = outputs.last_hidden_state, maps
        last_hidden_state = ln(last_hidden_state)
        reduce = reduce_shape(last_hidden_state, maps)

        # s part
        S = rel_s(dropout(reduce.unsqueeze(0)))
        s0, s1 = S[:, :, :3, :], S[:, :, 3:, :]
        graph = np.array(sample_batched['label'])
        # group part
        G = rel_g(dropout(reduce.unsqueeze(0)))
        g0, g1 = G[:, :, :3, :], G[:, :, 3:, :]

        # label = graph[0, :3, :].unsqueeze(0)
        # matrix_s = graph[0, 3:, :].unsqueeze(0)
        # matrix_g = graph[1, 3:, :].unsqueeze(0)

        label_s = torch.tensor(extend_label(
            graph[0, :3, :])).unsqueeze(0).cuda()
        label_g = torch.tensor(extend_label(
            graph[1, :3, :])).unsqueeze(0).cuda()
        matrix_s = torch.tensor(extend_matrix(
            graph[0, 3:, :])).unsqueeze(0).cuda()
        matrix_g = torch.tensor(extend_matrix(
            graph[1, 3:, :])).unsqueeze(0).cuda()

        # print()
        text = [tokenizer.cls_token] + [x[0]
                                        for x in sample_batched["text"]] + [tokenizer.sep_token]
        # print(text)
        label_actual = label_s.squeeze(0)
        S_ = extend_matrix(graph[0, 3:, :])
        G_ = extend_matrix(graph[1, 3:, :])
        question_heads = [i for i, ele in enumerate(
            label_actual[0]) if ele != 0]
        answer_heads = [i for i, ele in enumerate(label_actual[1]) if ele != 0]
        header_heads = [i for i, ele in enumerate(label_actual[2]) if ele != 0]

        pred_matrix_s = torch.softmax(s1, dim=1)
        pred_matrix_s = torch.argmax(pred_matrix_s, dim=1).squeeze(0)

        pred_label = torch.argmax(s0, dim=1).squeeze(0)
        pred_S = np.array([list(x)
                          for x in np.array(pred_matrix_s.cpu().numpy())])
        # pred_G = np.array([list(x) for x in np.array(pred_matrix_g.cpu().numpy())])
        pred_question_heads = [
            i for i, ele in enumerate(pred_label[0]) if ele != 0]
        pred_answer_heads = [
            i for i, ele in enumerate(pred_label[1]) if ele != 0]
        bbox_loss = bbox_loss_fn(S_, pred_S, ex_bboxes, (question_heads,
                                                         answer_heads,
                                                         pred_answer_heads,
                                                         pred_answer_heads)).cuda()

        loss_label_s = loss_clss(s0, label_s.long())
        loss_label_g = loss_clss(g0, label_g.long())
        loss_matrix_s = loss_clss(s1, matrix_s.long())
        loss_matrix_g = loss_clss(g1, matrix_g.long())
        loss = loss_label_s + loss_matrix_s + loss_label_g + loss_matrix_g + bbox_loss
        loss.backward()
        # loss_matrix_s.backward()
        # loss_matrix_g.backward()
        print('loss', loss.detach())
        print('loss_label_s', loss_label_s.detach())
        print('loss_label_g', loss_label_g.detach())
        print('loss_matrix_s', loss_matrix_s.detach())
        print('loss_matrix_g', loss_matrix_g.detach())
        print('loss_bboxes', bbox_loss.detach())
        log_metric('loss', loss.detach(), epoch)
        log_metric('loss_label_s', loss_label_s.detach(), epoch)
        log_metric('loss_label_g', loss_label_g.detach(), epoch)
        log_metric('loss_matrix_s', loss_matrix_s.detach(), epoch)
        log_metric('loss_matrix_g', loss_matrix_g.detach(), epoch)
        log_metric('loss_bboxes', bbox_loss.detach(), epoch)

        with torch.no_grad():
            ############[GROUND TRUTH]####################
            label_actual = label_s.squeeze(0)
            S_ = extend_matrix(graph[0, 3:, :])
            G_ = extend_matrix(graph[1, 3:, :])
            question_heads = [i for i, ele in enumerate(
                label_actual[0]) if ele != 0]
            answer_heads = [i for i, ele in enumerate(
                label_actual[1]) if ele != 0]
            header_heads = [i for i, ele in enumerate(
                label_actual[2]) if ele != 0]

            ques = get_strings(question_heads, text, S_)
            ans = get_strings(answer_heads, text, S_)
            print(len(ques), len(ans))
            print(f'[GROUND TRUTH]: Ques:{ques} \n Ans: {ans}')

            print('\n ###############################')
            ############[PREDICT]####################

            pred_label = torch.argmax(s0, dim=1).squeeze(0)

            pred_matrix_s = torch.softmax(s1, dim=1)
            pred_matrix_s = torch.argmax(pred_matrix_s, dim=1).squeeze(0)
            pred_matrix_g = torch.softmax(g1, dim=1)
            pred_matrix_g = torch.argmax(pred_matrix_g, dim=1).squeeze(0)
            # pred_label.shape
            # pred_matrix_s
            pred_question_heads = [
                i for i, ele in enumerate(pred_label[0]) if ele != 0]
            pred_answer_heads = [
                i for i, ele in enumerate(pred_label[1]) if ele != 0]
            pred_header_heads = [
                i for i, ele in enumerate(pred_label[2]) if ele != 0]

            pred_S = np.array([list(x)
                              for x in np.array(pred_matrix_s.cpu().numpy())])
            pred_G = np.array([list(x)
                              for x in np.array(pred_matrix_g.cpu().numpy())])
            # G = nx.Graph(pred_S)

            pred_ques = get_strings(pred_question_heads, text, pred_S)
            # print(np.shape(ques))

            pred_ans = get_strings(pred_answer_heads, text, pred_S)
            # print(np.shape(ans))
            print(f'[PREDICT]: Ques:{pred_ques} \n Ans: {pred_ans}')

            for ques_idx in pred_question_heads:
                G_pred = nx.Graph(pred_G)  # group
                dfs = list(nx.dfs_edges(G_pred, source=int(ques_idx)))
                # print(dfs)
                if len(dfs) != 0:
                    q, a = dfs[0]
                    qu_s = [qs[1] for qs in pred_ques if q in qs]
                    an_s = [as_[1] for as_ in pred_ans if a in as_]
                    # if len(qu_s)== len(an_s):
                    #     print(qu_s[0], an_s[0])
                    print(f'[PREDICT]: ques: {qu_s} \n ans: {an_s}')

                    # if len(qu_s) !=0:
                    #     print("QUES",qu_s[0], "\n")
                    # else : print("QUES",qu_s, "\n")
                    # if len(an_s) !=0:
                    #     print("ANS",an_s[0], "\n")
                    # else : print("ANS",an_s, "\n")

        optimizer.step()
    scheduler.step()
    if (epoch + 1) % 10 == 0:
        lr *= 0.5

    # if epoch % 5 == 0:    # test every 5 epochs
    #     model.eval()
    #     # with torch.no_grad():
    #     for i, val_batched in enumerate((val_loader)):
    #         input_ids = val_batched["input_ids"].squeeze(0).cuda()
    #         attention_mask = val_batched["attention_mask"].squeeze(0).cuda()
    #         token_type_ids = val_batched["token_type_ids"].squeeze(0).cuda()
    #         bbox = val_batched["bbox"].squeeze(0).cuda()
    #         maps = val_batched['maps']
    #         outputs = model(
    #             input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids
    #         )
    #         last_hidden_state, maps = outputs.last_hidden_state, maps
    #         last_hidden_state = ln(last_hidden_state)
    #         reduce = reduce_shape(last_hidden_state, maps)

    #             #s part
    #         S_val = rel_s(dropout(reduce.unsqueeze(0)))
    #         val_s0,val_s1 = S_val[:,:,:3,:],S_val[:,:,3:,:]
    #         graph = np.array(val_batched['label'])
    #         #group part
    #         G_val = rel_g(dropout(reduce.unsqueeze(0)))
    #         val_g0,val_g1 = G_val[:,:,:3,:],G_val[:,:,3:,:]
    #         val_text = [tokenizer.cls_token] + [x[0] for x in val_batched["text"]] + [tokenizer.sep_token]
    #         with torch.no_grad():
    #             val_pred_label = torch.argmax(val_s0,dim=1).squeeze(0)
    #             val_pred_matrix_s = torch.softmax(val_s1,dim=1)
    #             val_pred_matrix_s = torch.argmax(val_pred_matrix_s,dim=1).squeeze(0)
    #             val_pred_matrix_g = torch.softmax(val_g1,dim=1)
    #             val_pred_matrix_g = torch.argmax(val_pred_matrix_g,dim=1).squeeze(0)
    #             # pred_label.shape
    #             # pred_matrix_s
    #             val_pred_question_heads = [i for i, ele in enumerate(val_pred_label[0]) if ele != 0]
    #             val_pred_answer_heads = [i for i, ele in enumerate(val_pred_label[1]) if ele != 0]
    #             val_pred_header_heads = [i for i, ele in enumerate(val_pred_label[2]) if ele != 0]

    #             val_pred_S = np.array([list(x) for x in np.array(val_pred_matrix_s.cpu().numpy())])
    #             val_pred_G = np.array([list(x) for x in np.array(val_pred_matrix_g.cpu().numpy())])
    #             # G = nx.Graph(pred_S)

    #             val_pred_ques = get_strings(val_pred_question_heads, val_text, val_pred_S)
    #             # print(np.shape(ques))

    #             val_pred_ans = get_strings(val_pred_answer_heads, val_text, val_pred_S)
    #             # print(np.shape(ans))
    #             print(f'[PREDICT VAL]: Ques:{val_pred_ques} \n Ans: {val_pred_ans}')

    if epoch % 100 == 0:
        now = datetime.now()
        now = now.strftime("%d-%m-%Y_%H-%M-%S")
        with open(f"resources/checkpoints/DP_model{now}.pt", "wb") as f:
            torch.save(model.state_dict(), f)
