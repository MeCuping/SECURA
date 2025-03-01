import os
import time
import json
import torch
from src import Evaluation
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction

#自主加入了三个table记得减3
def to_one_hot(tokens,vocab_size = 151646,isUseCuda = True):
    vocab_size += 3
    if(isUseCuda == True):
        out = torch.zeros([1,len(tokens),vocab_size]).cuda()
    else:
        out = torch.zeros([1,len(tokens),vocab_size])
    for j,i in enumerate(tokens):
        out[0,j,i] = 1
    return out

def t_softmax(logits,tempreature = 1.0):
    #就是原值除去了一个温度
    temp = logits/tempreature
    prob = torch.softmax(temp,dim=-1)
    return prob

def top_p_decoding(logits, temperature=1.0, p=0.7):
    prob = t_softmax(logits, temperature)
    sorted_prob, sorted_index = torch.sort(prob, descending=True)
    cumulative_prob = torch.cumsum(sorted_prob, dim=-1)
    mask = cumulative_prob <= p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = 1
    top_p_prob = sorted_prob[mask]
    top_p_index = sorted_index[mask]
    top_p_prob /= top_p_prob.sum()
    next_token = torch.multinomial(top_p_prob, num_samples=1)
    return top_p_index[next_token]

def top_k_decoding(logits,temperature = 1.0,k = 3):
    prob = t_softmax(logits,temperature)
    sorted_prob, sorted_index = torch.sort(prob, descending=True)
    sorted_index = sorted_index[:,:,:k]
    sorted_prob = sorted_prob[:,:,k]
    sorted_prob /= sorted_prob.sum()
    next_token = torch.multinomial(sorted_prob, num_samples=1)
    return sorted_index[next_token]

def greedy_decoding(logits):
    return torch.argmax(logits, dim=-1)

def generatetext(mode,tokenizer,pk,model,inp,mx_Gen = 50,temp = 0.7):
    """
    mode:topp和topk和greedy
    pk为对应模式下的参数
    """
    #print(tokenizer.decode(tokenizer.encode("[EMPTY]")[1]))
    try:
        inp.append(tokenizer.encode("[EMPTY]")[1])
    except IndexError:
        inp.append(tokenizer.encode("[EMPTY]")[0])
    with torch.no_grad():
        out_ = []
        for _ in range(mx_Gen):
            #我了个豆啊，不用的赶紧清除啊...
            torch.cuda.empty_cache()
            inp_ = torch.tensor([inp]).cuda()
            out = model(inp_)
            out = out.logits
            if(mode == "greedy"):
                out = greedy_decoding(out[:,-1,:])[0]
                out_.append(out.item())
            elif(mode == "topp"):
                out = top_p_decoding(out[:,-1,:],temp,pk)[0]
                out_.append(out.item())
            elif(mode == "topk"):
                out = top_k_decoding(out[:,-1,:],temp,pk)[0]
                out_.append(out.item())
            inp.append(out_[-1])
            #只能说，记得改。。。
            try:
                if(inp[-1] == 151645 or inp[-1] == tokenizer.encode("[END]")[1]):
                    break
            except IndexError:
                if(inp[-1] == 151645 or inp[-1] == tokenizer.encode("[END]")[0]):
                    break
    return out_

def get_next_filename(directory = "./TestOutputs"):
    files = os.listdir(directory)
    json_files = [f for f in files if f.endswith('.json')]
    
    max_number = -1
    for file in json_files:
        try:
            number = int(file.split('.')[0])  
            if number > max_number:
                max_number = number
        except ValueError:
            pass  
    next_number = max_number + 1
    return os.path.join(directory, f"{next_number}.json"),next_number

def test(model,testStep,DataSetsTest,tokenizer,rougescorer,testName,SavePath = None):
    """
    SavePath填入的是文件夹
    """
    model.eval()
    smoothing = SmoothingFunction().method4
    sm = 0
    sm_bleu = 0
    sm_rouge1 = 0
    sm_rougeL = 0
    selfbleu = 0
    selfrouge = 0
    time1 = time.time()
    for i in range(testStep):
        inp,lbl,str_inp,str_label,start_idx = DataSetsTest.preprocess(tokenizer)
        inp = torch.tensor([inp]).cuda()
        out = model(inp)
        out = out.logits
        lbl = to_one_hot(lbl)
        sm += 1
        selfbleu += Evaluation.BLEU(out[:,start_idx:,:],lbl[:,start_idx:,:])
        selfrouge += Evaluation.Rouge_1(out[:,start_idx:,:],lbl[:,start_idx:,:])
        out = tokenizer.decode(greedy_decoding(out[:,start_idx:,:])[0])
        sm_bleu += sentence_bleu([str_label],out,smoothing_function=smoothing)
        rg = rougescorer.score("".join(str_label),out)
        sm_rouge1 += rg["rouge1"][2]
        sm_rougeL += rg["rougeL"][2] 
        torch.cuda.empty_cache()
    time2 = time.time()
    avg_selfrouge = selfrouge/sm
    avg_selfbleu = selfbleu/sm
    avg_bleu = sm_bleu/sm
    avg_rouge1 = sm_rouge1/sm
    avg_rougeL = sm_rougeL/sm
    print(f"{testStep}次测试,\navg_selfrouge:{avg_selfrouge},avg_selfbleu:{avg_selfbleu},avg_bleu:{avg_bleu},avg_rouge1:{avg_rouge1},avg_rougeL:{avg_rougeL}")
    if(SavePath != None):
        dic = {"tstName":testName,"time":time2-time1,"avg_selfrouge":avg_selfrouge,"avg_selfbleu":avg_selfbleu,"avg_bleu":avg_bleu,"avg_rouge1":avg_rouge1,"avg_rougeL":avg_rougeL}
        filename,__ = get_next_filename(SavePath)
        with open(filename, 'w', encoding="utf-8") as file:
            json.dump(dic, file, indent=4)
            print(f"Combined data saved to {filename}")
    #记得写一个实验记录保存
    model.train()
    #完善test

