from transformers import AutoModel, AutoTokenizer, AutoConfig, BatchEncoding, BertModel,LayoutLMModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from dataModel.datamd_ import DpDataSet
from jsonmerge import merge
from graph_stuff import get_strings, get_qa
model = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased").cuda()
config = AutoConfig.from_pretrained("microsoft/layoutlm-base-uncased")
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.00005)
trainData = DpDataSet(path='./data/processed/data_train.jsonl')
train_loader =  DataLoader(trainData, batch_size= 1, shuffle= False, num_workers= 4)
from spade_model import RelationTagger

reduce_size = 256
ln = nn.Linear(config.hidden_size, reduce_size).cuda()
def reduce_shape(last_hidden_state, maps):
    i = 0
    # reduce = torch.zeros(768)
    reduce =[]
    for g_token in  maps:
        ten = torch.zeros(reduce_size).cuda()
        for ele in g_token:
            # print(ele)
            ten += last_hidden_state[0][i]
            # i+=1
            # print(last_hidden_state[0][i])
            i+=1
        ten = ten/len(g_token)
        # print(ten)
        reduce.append(ten)
        # reduce = torch.cat((reduce,ten),-1)
    # print(np.array(reduce).shape)
    # print(reduce)
    reduce=  torch.stack(reduce)
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

def grouth_truth(text_,label_):
    graph_s = np.array(label_, dtype='int8')
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
    print('ground truth', ques,ans)
    resul_ = {}
    for ques_idx in question_heads:
        q_, a_ = get_qa(ques_idx, ques, ans, G_)
        resul_ = merge(resul_, {q_: a_})
        # print(get_ques_ans(ques_idx, ques,ans))
    print('Grouth truth',resul_)
    return(resul_)


for epoch in tqdm(range(0,10)):
    model.train()
    for i, sample_batched in enumerate((train_loader)):
        input_ids = sample_batched["input_ids"].squeeze(0).cuda()
        attention_mask = sample_batched["attention_mask"].squeeze(0).cuda()
        token_type_ids = sample_batched["token_type_ids"].squeeze(0).cuda()
        bbox = sample_batched["bbox"].squeeze(0).cuda()
        maps = sample_batched['maps']
        print(input_ids.shape,attention_mask.shape,bbox.shape,token_type_ids.shape, np.array (maps).shape)
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        last_hidden_state, maps = outputs.last_hidden_state, maps
        last_hidden_state = ln(last_hidden_state)
        reduce = reduce_shape(last_hidden_state, maps)
        loss_clss = nn.CrossEntropyLoss()
        #s part
        S = rel_s(dropout(reduce.unsqueeze(0)))
        # S = rel_s
        # S = torch.argmax(rel_s,dim =1)
        s0,s1 = S[:,:,:3,:],S[:,:,3:,:]
        # torch.argmax(s0,dim =1).numpy()
        s1 =  s1[:,:,1:-1,1:-1]#reduce
        s0 = s0[:,:,:,1:-1]# reduce

        
        graph = torch.tensor(sample_batched['label']).cuda()
        # graph.copy()
        label = graph[0, :3, :].unsqueeze(0)
        # graph = np.array(label)

        
        # matrix_s = graph[0, 3:, :].unsqueeze(0)
        # matrix_g = graph[1, 3:, :].unsqueeze(0)
        # loss_label_s = loss_clss(s0, label.long())

        G = rel_g(dropout(reduce.unsqueeze(0)))
        # rel_s.shape
        # G = rel_g
        # S = torch.argmax(rel_s,dim =1)
        g0,g1 = G[:,:,:3,:],G[:,:,3:,:]
        # torch.argmax(s0,dim =1).numpy()
        g1 =  g1[:,:,1:-1,1:-1]#reduce
        g0 = g0[:,:,:,1:-1]# reduce
        # graph = torch.tensor(batch['label']).cuda()
        # label = graph[0, :3, :].unsqueeze(0)
        # graph = np.array(label)

        matrix_s = graph[0, 3:, :].unsqueeze(0)
        matrix_g = graph[1, 3:, :].unsqueeze(0)
        loss_label_s = loss_clss(s0, label.long())
        loss_matrix_s = loss_clss(s1,matrix_s.long())
        loss_matrix_g = loss_clss(g1,matrix_g.long())
        loss = loss_label_s + loss_matrix_s + loss_matrix_g
        loss.backward()
        loss_matrix_s.backward()
        loss_matrix_g.backward()
        print ('loss', loss.detach())
        print ('loss_label_s', loss_label_s.detach())
        print ('loss_matrix_s', loss_matrix_s.detach())
        print ('loss_matrix_g', loss_matrix_g.detach())
        # print()
        text = [x[0] for x in sample_batched["text"]]
        # print(text)
        with torch.no_grad():
            label_s = torch.argmax(s0,dim  =1).squeeze(0).cpu().numpy()
            pred_s = torch.argmax(s1,dim  =1).squeeze(0).cpu().numpy()
            pred_g = torch.argmax(g1,dim  =1).squeeze(0).cpu().numpy()
            S_ = pred_s
            G_ = pred_g
            question_heads = [i for i, ele in enumerate(label_s[0]) if ele != 0]
            answer_heads = [i for i, ele in enumerate(label_s[1]) if ele != 0]
            header_heads = [i for i, ele in enumerate(label_s[2]) if ele != 0]
            # S_ = list(s1[:,0, 3:, :].squeeze(0).cpu().numpy())
            ques = get_strings(question_heads, text, S_)
            ans = get_strings(answer_heads, text, S_)
            print('pred', ques,ans)
            try:
                for ques_idx in question_heads:
                    q_, a_ = get_qa(ques_idx, ques, ans, G_)
                    print('pred', q_, a_)
                    # resul_ = merge(resul_, {q_: a_})
            except Exception:
                print('something gone wrong')
                    # print(get_ques_ans(ques_idx, ques,ans))
            # if(len(ques) == len(ans)):
            #     resul_ = {}
            #     for ques_idx in question_heads:
            #         q_, a_ = get_qa(ques_idx, ques, ans, G_)
            #         resul_ = merge(resul_, {q_: a_})
            #         # print(get_ques_ans(ques_idx, ques,ans))
            #     print('pred', resul_)
            
            try:
                grouth_truth(text,sample_batched['label'])
            except Exception:
                print(Exception)
            # print([i for i, ele in enumerate(pred[0]) if ele != 0])
            # print(pred.shape)
        # print(outputs.last_hidden_state)
        optimizer.step()


