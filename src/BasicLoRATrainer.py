import os
import torch
import json
import random
import torch.nn as nn
from src import Evaluation
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
from src.Processer import to_one_hot
from src.DataLoader import DataLoader
from src import MinePeft
from src import CABR_DoRA_Sep as CABR_DoRA
from src import Processer
from peft import LoraConfig,get_peft_model,PeftModel,AdaLoraConfig
from transformers import AutoModelForCausalLM,AutoTokenizer,AutoConfig
from src.tools import GradBiasOut

random.seed(114514)
torch.manual_seed(114514)
# random.seed(123)
# torch.manual_seed(123)
smoothing = SmoothingFunction().method4

def trainer(ParamDic,basicModelpth,trainpath,tstName = None,loraPth = None,testpath = None,mx = 800,isUseLoRA = True,expID = -1,change = None):
    """
    ParamDic 训练基础参数
    basicModelpth 模型本体地址
    trainpath 训练数据集地址
    tstName = None 测试名称
    loraPth = None lora载入地址（如果有)
    testpath = None 测试集地址
    mx = 800 可接受上下文上限
    isUseLoRA = True 是否采用lora
    expID = -1 实验id
    change = None 实验五的切换速度
    """
    Traintype = ParamDic["Traintype"]
    ParamDic["tstName"] = tstName
    CURdivideRank = ParamDic["CURdivideRank"]
    ABdivideRank = ParamDic["ABdivideRank"]
    lora_dropout = ParamDic["lora_dropout"]
    target_modules = ParamDic["target_modules"]
    BasicLoRARank = ParamDic["BasicLoRARank"]
    isNeedBias = ParamDic["isNeedBias"]
    EdgeNum = ParamDic["EdgeNum"]
    decayRatio = ParamDic["decayRatio"]
    Percent = ParamDic["Percent"]
    lr = ParamDic["lr"]
    if(expID != 2 and expID != 3 and expID != 0):
        testDir = ParamDic["testDir"]
    else:
        testDir = None

    epoch = ParamDic["epoch"]
    tstStep = ParamDic["tstStep"]
    saveStep = ParamDic["saveStep"]
    ckStep = ParamDic["ckStep"]
    Is_CABR_Only = False
    if(Traintype == "CABR-LoRA_Only"):
        Traintype = "CABR-LoRA"
        Is_CABR_Only = True
    if(Traintype == "CABR-LoRA" or Traintype == "CABR-LoRA_L"):
        FusionStep = ParamDic["FusionStep"]

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    model = AutoModelForCausalLM.from_pretrained(
            basicModelpth,
            torch_dtype="auto",
            device_map="cuda:0",
        )
    hidden = AutoConfig.from_pretrained(basicModelpth).num_hidden_layers
    tokenizer = AutoTokenizer.from_pretrained(basicModelpth)
    special_tokens = ["[EMPTY]","[START]","[END]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    if(expID != 5):
        Datasets_Train = DataLoader(trainpath,mx)
    elif(expID == 5):
        DT = []
        changeidx = 0
        for i in trainpath:
            DT.append(DataLoader(i,mx))
        Datasets_Train = DT[changeidx]
    if(testpath != None):
        Datasets_Test = DataLoader(testpath,mx)
    if(testDir != None):
        Datasets_Cat = DataLoader(testDir,mx)
    if(isNeedBias == True):
        bias = "lora_only"
    else:
        bias = "none"
    if(isUseLoRA == True):
        if(Traintype == "LoRA" and loraPth == None):
            peftConfig = LoraConfig(target_modules = target_modules,
                                    r = BasicLoRARank,
                                    lora_alpha=16,
                                    lora_dropout=lora_dropout,
                                    bias=bias)
        elif(Traintype == "DoRA" and loraPth == None):
            peftConfig = LoraConfig(target_modules = target_modules,
                                    r = BasicLoRARank,
                                    lora_alpha=16,
                                    lora_dropout=lora_dropout,
                                    bias=bias,
                                    use_dora=True)
        elif(Traintype == "CUR-LoRA"):
            peftConfig = LoraConfig(
                target_modules = target_modules,  # 只对atten.c_proj微调
                r=BasicLoRARank,  # LoRA rank
                lora_alpha=16,  # LoRA scaling factor
                lora_dropout=lora_dropout,  # LoRA dropout
                bias=bias,  # 只微调LoRA的Bias
            )

            aConfig = CABR_DoRA.CABDoRAConfig(CURdivideRank,ABdivideRank,isNeedBias,EdgeNum,decayRatio,Percent,Type=Traintype)
            MinePeft.RefactoryLoRALinear(aConfig)
            model = MinePeft.Newget_peft_model(model,peftConfig,aConfig)
        #验证AB有效性时用
        elif(Traintype == "CABR-LoRA"):
            peftConfig = LoraConfig(
                target_modules = target_modules,  # 只对atten.c_proj微调
                r=BasicLoRARank,  # LoRA rank
                lora_alpha=16,  # LoRA scaling factor
                lora_dropout=lora_dropout,  # LoRA dropout
                bias=bias  # 只微调LoRA的Bias
            )
            if(Is_CABR_Only == True):
                aConfig = CABR_DoRA.CABDoRAConfig(CURdivideRank,ABdivideRank,isNeedBias,EdgeNum,decayRatio,Percent,Type=Traintype,Is_CABR_Only = Is_CABR_Only)
            else:

                aConfig = CABR_DoRA.CABDoRAConfig(CURdivideRank,ABdivideRank,isNeedBias,EdgeNum,decayRatio,Percent,Type=Traintype)
            MinePeft.RefactoryLoRALinear(aConfig)
            model = MinePeft.Newget_peft_model(model,peftConfig,aConfig)
        elif(Traintype == "CABR-LoRA_L"):
            peftConfig = LoraConfig(
                target_modules = target_modules,  # 只对atten.c_proj微调
                r=BasicLoRARank,  # LoRA rank
                lora_alpha=16,  # LoRA scaling factor
                lora_dropout=lora_dropout,  # LoRA dropout
                bias=bias  # 只微调LoRA的Bias
            )

            aConfig = CABR_DoRA.CABDoRAConfig(CURdivideRank,ABdivideRank,isNeedBias,EdgeNum,decayRatio,Percent,Type=Traintype)
            MinePeft.RefactoryLoRALinear(aConfig)
            model = MinePeft.Newget_peft_model(model,peftConfig,aConfig)
        elif(Traintype == "I-LoRA"):
            peftConfig = LoraConfig(
                target_modules = target_modules,  # 只对atten.c_proj微调
                r=BasicLoRARank,  # LoRA rank
                lora_alpha=16,  # LoRA scaling factor
                lora_dropout=lora_dropout,  # LoRA dropout
                bias=bias,  # 只微调LoRA的Bias
            )

            aConfig = CABR_DoRA.CABDoRAConfig(CURdivideRank,ABdivideRank,isNeedBias,EdgeNum,decayRatio,Percent,Type=Traintype)
            MinePeft.RefactoryLoRALinear(aConfig)
            model = MinePeft.Newget_peft_model(model,peftConfig,aConfig)
        elif(Traintype == "CABR-DoRA"):
            peftConfig = LoraConfig(
                target_modules = target_modules,  # 只对atten.c_proj微调
                r=BasicLoRARank,  # LoRA rank
                lora_alpha=16,  # LoRA scaling factor
                lora_dropout=lora_dropout,  # LoRA dropout
                bias=bias,  # 只微调LoRA的Bias
            )

            aConfig = CABR_DoRA.CABDoRAConfig(CURdivideRank,ABdivideRank,isNeedBias,EdgeNum,decayRatio,Percent,Type=Traintype)
            MinePeft.RefactoryLoRALinear(aConfig)
            model = MinePeft.Newget_peft_model(model,peftConfig,aConfig)
        elif(Traintype == "LoRAXS"):
            peftConfig = LoraConfig(
                target_modules = target_modules,  # 只对atten.c_proj微调
                r=BasicLoRARank,  # LoRA rank
                lora_alpha=16,  # LoRA scaling factor
                lora_dropout=lora_dropout,  # LoRA dropout
                bias=bias,  # 只微调LoRA的Bias
            )

            aConfig = CABR_DoRA.CABDoRAConfig(CURdivideRank,ABdivideRank,isNeedBias,EdgeNum,decayRatio,Percent,Type=Traintype)
            MinePeft.RefactoryLoRALinear(aConfig)
            model = MinePeft.Newget_peft_model(model,peftConfig,aConfig)
        elif(Traintype == "BiasOnly_5" or Traintype == "BiasOnly_1"):
            peftConfig = LoraConfig(
                target_modules = target_modules,  # 只对atten.c_proj微调
                r=BasicLoRARank,  # LoRA rank
                lora_alpha=16,  # LoRA scaling factor
                lora_dropout=lora_dropout,  # LoRA dropout
                bias=bias,  # 只微调LoRA的Bias
            )

            aConfig = CABR_DoRA.CABDoRAConfig(CURdivideRank,ABdivideRank,isNeedBias,EdgeNum,decayRatio,Percent,Type=Traintype)
            MinePeft.RefactoryLoRALinear(aConfig)
            model = MinePeft.Newget_peft_model(model,peftConfig,aConfig)
        if(loraPth != None):
            print("获取LoRA模型中...")
            if(Traintype == "LoRA"):
                model = PeftModel.from_pretrained(model,loraPth)
                print(f"{Traintype}获取完毕")
            elif(Traintype == "CUR-LoRA"):
                MinePeft.load_selected_parameters(model,loraPth)
            elif(Traintype == "CABR-LoRA"):
                model.load_state_dict(torch.load("model.pth"))
            elif(Traintype == "CABR-LoRA_L"):
                MinePeft.load_selected_parameters(model,loraPth)
            elif(Traintype == "I-LoRA"):
                MinePeft.load_selected_parameters(model,loraPth)
            elif(Traintype == "DoRA"):
                model = PeftModel.from_pretrained(model,loraPth)
                print(f"{Traintype}获取完毕")
            else:
                print("未知的LoRA类型，将加载基础LoRA初始化")
                model = get_peft_model(model,peftConfig)
                for param in model.base_model.parameters():
                    param.requires_grad = False
        else:
            if(Traintype == "LoRA"):
                model = get_peft_model(model,peftConfig)
                for param in model.base_model.parameters():
                    param.requires_grad = False
                for name,param in model.named_parameters():
                    if("lora" in name):
                        param.requires_grad = True
                        print()
            elif(Traintype == "CUR-LoRA"):
                #应在此处加入自定义lora修改的部分，对应上述的UseLoRAtype
                for param in model.base_model.parameters():
                    param.requires_grad = False
                for name,param in model.named_parameters():
                    if("CUR" in name):
                        param.requires_grad = True
            elif(Traintype == "CABR-LoRA"):
                #应在此处加入自定义lora修改的部分，对应上述的UseLoRAtype
                for param in model.base_model.parameters():
                    param.requires_grad = False
                for name,param in model.named_parameters():
                    if("CABR" in name):
                        param.requires_grad = True
            elif(Traintype == "CABR-LoRA_L"):
                #应在此处加入自定义lora修改的部分，对应上述的UseLoRAtype
                for param in model.base_model.parameters():
                    param.requires_grad = False
                for name,param in model.named_parameters():
                    if("CABR" in name and not "_L" in name):
                        param.requires_grad = True
            elif(Traintype == "I-LoRA"):
                #应在此处加入自定义lora修改的部分，对应上述的UseLoRAtype
                for param in model.base_model.parameters():
                    param.requires_grad = False
                for name,param in model.named_parameters():
                    if("W_work" in name):
                        param.requires_grad = True
                ILoss = nn.MSELoss()
                Ilambda = model.base_model.model.model.layers[0].self_attn.v_proj.I_LoRA.return_lambda()
            elif(Traintype == "DoRA"):
                #应在此处加入自定义lora修改的部分，对应上述的UseLoRAtype
                model = get_peft_model(model,peftConfig)
                for param in model.base_model.parameters():
                    param.requires_grad = False
                for name,param in model.named_parameters():
                    if("lora" in name):
                        param.requires_grad = True

    filepath = rf"./outputs/EXP{expID}/{Traintype}/{Traintype}_{tstName}_{lr}"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        print(f"Created folder: {filepath}")
    print(model)
    model = model.cuda()
    opt = torch.optim.AdamW(model.parameters(),lr)
    #这是基础loss,更多的lora变体loss应被我封装在了lora变体类内部，直接调用即可
    baseLoss = nn.CrossEntropyLoss()
    sm = 0
    sm_bleu = 0
    sm_rouge1 = 0
    sm_rougeL = 0
    selfbleu = 0
    selfrouge = 0
    sm_Loss = 0
    if(expID == 0):
        saveName = f"./outputs/EXP0/Basic/{Traintype}_{tstName}_{lr}"
        weights = []
        for name,param in model.named_parameters():
            if(("v_proj" in name or "q_proj" in name) and "base_layer" in name and "bias" not in name):
                name
                weights.append(param)
        weigtNorm = 0
        saveWeigtNorm = {}
        for w_ in weights:
            weigtNorm += torch.norm(torch.svd(w_.to(torch.float))[1])
        saveWeigtNorm["WeightSVDNorm"] = (weigtNorm/(2*model.config.num_hidden_layers)).item()
        if not os.path.exists(saveName):
            os.makedirs(saveName)
        with open(os.path.join(saveName, "grad.json"),'w', encoding="utf-8") as w:
            json.dump(saveWeigtNorm,w,indent=4)
        print(f"SVD范数已保存到 {saveName} 文件中。")
    for i in range(epoch):
        if(sm == ckStep):
            avg_selfrouge = selfrouge/sm
            avg_selfbleu = selfbleu/sm
            avg_bleu = sm_bleu/sm
            avg_rouge1 = sm_rouge1/sm
            avg_rougeL = sm_rougeL/sm
            avg_Loss = sm_Loss/sm
            sm = 0
            sm_bleu = 0
            sm_rouge1 = 0
            sm_rougeL = 0
            selfbleu = 0
            selfrouge = 0
            sm_Loss = 0
            inp,lbl,str_inp,str_label,start_idx = Datasets_Train.preprocess(tokenizer)
            print("测试输入:",str_inp)
            out = Processer.generatetext("topp",tokenizer,0.7,model,inp,200,0.7)
            print("测试输出:",tokenizer.decode(out))
            print("对应标签:",str_label)
            print(f"第{i}次训练,avgLoss:{avg_Loss}\navg_selfrouge:{avg_selfrouge},avg_selfbleu:{avg_selfbleu},avg_bleu:{avg_bleu},avg_rouge1:{avg_rouge1},avg_rougeL:{avg_rougeL}")
            #下部嵌入测试函数
            if(expID == 2):
                Processer.test(model,tstStep,Datasets_Test,tokenizer,scorer,tstName,SavePath=rf"./outputs/EXP2/{Traintype}/{Traintype}_{tstName}_{lr}")
            elif(expID == 4):
                Processer.test(model,tstStep,Datasets_Cat,tokenizer,scorer,tstName,SavePath=rf"./outputs/EXP4/{Traintype}/{Traintype}_{tstName}_{lr}")
            elif(expID == 5):
                Processer.test(model,tstStep,Datasets_Cat,tokenizer,scorer,tstName,SavePath=rf"./outputs/EXP5/{Traintype}/{Traintype}_{tstName}_{lr}")
            elif(expID == 0):
                if(Traintype == "LoRA"):
                    saveName = f"./outputs/EXP0/LoRA/{Traintype}_{tstName}_{lr}_{i}"
                elif(Traintype == "DoRA"):
                    saveName = f"./outputs/EXP0/DoRA/{Traintype}_{tstName}_{lr}_{i}"
                else:
                    saveName = rf"./outputs/EXP0/{Traintype}/{Traintype}_{tstName}_{lr}_{i}_OnlyCABR_{Is_CABR_Only}"
                weights = []
                if(Traintype == "LoRA" or Traintype == "DoRA"):
                    model = model.merge_and_unload()
                    print(model)
                    for name,param in model.named_parameters():
                        if(("v_proj" in name or "q_proj" in name) and "bias" not in name):
                            name
                            weights.append(param)
                elif(Traintype == "CABR-LoRA"):
                    MinePeft.mergeSelfModel(model,Traintype,hidden)
                    #按照layer输入
                    for i in range(28):
                        param_1 = model.base_model.model.model.layers[i].self_attn.v_proj.CABRLoRA.baseWeight
                        param_2 = model.base_model.model.model.layers[i].self_attn.q_proj.CABRLoRA.baseWeight
                        weights.append(param_1)
                        weights.append(param_2)
                weigtNorm = 0
                saveWeigtNorm = {}
                for w_ in weights:
                    weigtNorm += torch.norm(torch.svd(w_.to(torch.float))[1])
                saveWeigtNorm["WeightSVDNorm"] = (weigtNorm/(2*model.config.num_hidden_layers)).item()
                if not os.path.exists(saveName):
                    os.makedirs(saveName)
                with open(os.path.join(saveName, "grad.json"),'w', encoding="utf-8") as w:
                    json.dump(saveWeigtNorm,w,indent=4)
                print(f"SVD范数已保存到 {saveName} 文件中。")
                break
            else:
                Processer.test(model,tstStep,Datasets_Test,tokenizer,scorer,tstName)
        while True:
            try:
                inp,lbl,str_inp,str_label,start_idx = Datasets_Train.preprocess(tokenizer)
                inp = torch.tensor([inp]).cuda()
                out = model(inp)
                # print("Output shape:", out[0, start_idx:, :].shape)
                # print("Target shape:", lbl[0, start_idx:, :].shape)
                #print(tokenizer.decode(Processer.greedy_decoding(out)[0]))
                if(Traintype != "I-LoRA"):
                    out = out.logits
                    lbl = to_one_hot(lbl)
                    loss = baseLoss(out[0,start_idx:,:],lbl[0,start_idx:,:])
                else:
                    out = out.logits
                    #缺点，跑双倍
                    for p in range(hidden):
                        model.base_model.model.model.layers[p].self_attn.q_proj.I_LoRA.isOld = True
                        model.base_model.model.model.layers[p].self_attn.v_proj.I_LoRA.isOld = True
                    outBefor = model.forward(inp)
                    outBefor = outBefor.logits
                    lbl = to_one_hot(lbl)
                    loss1 = baseLoss(out[0,start_idx:,:],lbl[0,start_idx:,:])
                    loss2 = ILoss(out[0,start_idx:,:],outBefor[0,start_idx:,:])
                    loss = loss1 + Ilambda*loss2
                    for p in range(hidden):
                        model.base_model.model.model.layers[p].self_attn.q_proj.I_LoRA.isOld = False
                        model.base_model.model.model.layers[p].self_attn.v_proj.I_LoRA.isOld = False
                break
            except ValueError:
                print("ValueError,ReTry")
        loss.backward()
        #print(torch.sum(model.base_model.model.model.layers[0].self_attn.v_proj.CABRLoRA.CABR_B.weight))
        #print(model.base_model.model.model.layers[0].self_attn.v_proj.CABRDoRA.CABRLA.CABR_B.weight)
        opt.step()
        sm += 1
        selfbleu += Evaluation.BLEU(out[:,start_idx:,:],lbl[:,start_idx:,:])
        selfrouge += Evaluation.Rouge_1(out[:,start_idx:,:],lbl[:,start_idx:,:])
        out = tokenizer.decode(Processer.greedy_decoding(out[:,start_idx:,:])[0])
        sm_bleu += sentence_bleu([str_label],out,smoothing_function=smoothing)
        rg = scorer.score("".join(str_label),out)
        sm_rouge1 += rg["rouge1"][2]
        sm_rougeL += rg["rougeL"][2] 
        sm_Loss += loss
        if(expID == 5 and (i+1)%change == 0):
            changeidx += 1
            try:
                Datasets_Train = DT[changeidx]
            except IndexError:
                pass
        
        if(expID == 3):
            if(Traintype == "LoRA"):
                saveName = f"./outputs/EXP3/LoRA/{Traintype}_{tstName}_{lr}_{i}"
            elif(Traintype == "DoRA"):
                saveName = f"./outputs/EXP3/DoRA/{Traintype}_{tstName}_{lr}_{i}"
                #model.save_pretrained(f"./outputs/Bitsfit/{Traintype}_{tstName}_{lr}")
            else:
                saveName = rf"./outputs/EXP3/{Traintype}/{Traintype}_{tstName}_{lr}_{i}_CABROnly_{Is_CABR_Only}"
        if(expID == 3 and (i+1)%ParamDic["GradSaveStep"] == 0):
            gradients = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad  # 转换为列表
            
            q_grad = 0
            v_grad = 0
            saveGrad = {}
            for layer in range(model.config.num_hidden_layers):
                # q_grad_B_name = f"base_model.model.model.layers.{layer}.self_attn.q_proj.BiasOnly.FTBiasB"
                # v_grad_B_name = f"base_model.model.model.layers.{layer}.self_attn.v_proj.BiasOnly.FTBiasB"
                # q_grad_BiasOut_name = f"base_model.model.model.layers.{layer}.self_attn.q_proj.BiasOnly.FTBiasB.BiasOut"
                # v_grad_BiasOut_name = f"base_model.model.model.layers.{layer}.self_attn.v_proj.BiasOnly.FTBiasB.BiasOut"
                # q_FTBiasA = model.base_model.model.model.layers[layer].self_attn.q_proj.BiasOnly.FTBiasA.data
                # v_FTBiasA = model.base_model.model.model.layers[layer].self_attn.v_proj.BiasOnly.FTBiasA.data
                q_grad_name = f"base_model.model.model.layers[{layer}].self_attn.q_proj"
                v_grad_name = f"base_model.model.model.layers[{layer}].self_attn.v_proj"
                
                if(Traintype == "CABR-LoRA"):
                    A_q_grad_name = f"base_model.model.model.layers.{layer}.self_attn.q_proj.CABRLoRA.CABR_A.weight"
                    A_v_grad_name = f"base_model.model.model.layers.{layer}.self_attn.v_proj.CABRLoRA.CABR_A.weight"
                    
                    B_q_grad_name = f"base_model.model.model.layers.{layer}.self_attn.q_proj.CABRLoRA.CABR_B.weight"
                    B_v_grad_name = f"base_model.model.model.layers.{layer}.self_attn.v_proj.CABRLoRA.CABR_B.weight"
                elif(Traintype == "LoRA"):
                    A_q_grad_name = f"base_model.model.model.layers.{layer}.self_attn.q_proj.lora_A.default.weight"
                    A_v_grad_name = f"base_model.model.model.layers.{layer}.self_attn.v_proj.lora_A.default.weight"
                    
                    B_q_grad_name = f"base_model.model.model.layers.{layer}.self_attn.q_proj.lora_A.default.weight"
                    B_v_grad_name = f"base_model.model.model.layers.{layer}.self_attn.v_proj.lora_A.default.weight"


                s = torch.norm(gradients[A_q_grad_name])
                m = torch.norm(gradients[B_q_grad_name])
                q_grad += s+m
                v_grad += torch.norm(gradients[A_v_grad_name])+torch.norm(gradients[B_v_grad_name])
            # avg_q_grad_BiasOut = GradBiasOut(q_FTBiasA, gradients[q_grad_B_name])
            # avg_v_grad_BiasOut = GradBiasOut(v_FTBiasA, gradients[v_grad_B_name])
            saveGrad[q_grad_name] = (q_grad/model.config.num_hidden_layers).item()
            saveGrad[v_grad_name] = (v_grad/model.config.num_hidden_layers).item()
            if not os.path.exists(saveName):
                os.makedirs(saveName)
            with open(os.path.join(saveName, "grad.json"),'w', encoding="utf-8") as w:
                json.dump(saveGrad,w,indent=4)
            print(f"梯度已保存到 {saveName} 文件中。")

        if((i+1)%saveStep == 0):
            if(expID == 2):
                if(Traintype == "LoRA"):
                    saveName = f"./outputs/EXP2/LoRA/{Traintype}_{tstName}_{lr}"
                    model.save_pretrained(saveName)
                elif(Traintype == "DoRA"):
                    saveName = f"./outputs/EXP2/DoRA/{Traintype}_{tstName}_{lr}"
                    model.save_pretrained(saveName)
                    #model.save_pretrained(f"./outputs/Bitsfit/{Traintype}_{tstName}_{lr}")
                else:
                    saveName = rf"./outputs/EXP2/{Traintype}/{Traintype}_{tstName}_{lr}"
                    MinePeft.savePretrained(model,saveName,ParamDic,Traintype,hidden)
        opt.zero_grad()
        if(Traintype == "I-LoRA"):
            MinePeft.ILoRAUpdate(model,hidden)
        if(Traintype == "CABR-LoRA" and Is_CABR_Only == False):
            if((i+1)%FusionStep == 0):
                MinePeft.mergeSelfModel(model,Traintype,hidden)
        if(Traintype == "CABR-LoRA_L"):
            if((i+1)%FusionStep == 0):
                MinePeft.mergeSelfModel(model,Traintype,hidden)
        if(expID == 5 and (i+1)%change == 0):
            if(Traintype == "I-LoRA" or Traintype == "CUR-LoRA"):
                MinePeft.mergeSelfModel(model,Traintype,hidden)
        torch.cuda.empty_cache()
    return model