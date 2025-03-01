import torch
import torch.nn as nn

class I_LoRA(nn.Module):
    def __init__(self, baselayer, R = 8, lambda_ema=0.99):
        super().__init__()
        self.lambda_ema = lambda_ema

        self.isOld = False
        self.baseWeight = baselayer.weight.data
        dev = self.baseWeight.device
        input_dim,output_dim = self.baseWeight.shape[1],self.baseWeight.shape[0]
        # 工作记忆参数
        self.W_work_A = nn.Linear(input_dim,R,bias = False)
        self.W_work_B = nn.Linear(R,output_dim,bias = False)
        self.W_work_B.weight.data = torch.zeros_like(self.W_work_B.weight.data).to(dev)

        # 长期记忆参数
        self.W_long_A = self.W_work_A.weight.data.clone().to(dev)
        self.W_long_B = self.W_work_B.weight.data.clone().to(dev)
        self.W_long_A.requires_grad = False
        self.W_long_B.requires_grad = False

    def forward(self,x):
        dty = x.dtype
        if(not self.isOld):
            W_combined = torch.matmul(self.W_work_B.weight,self.W_work_A.weight) + self.baseWeight
            outNow = torch.matmul(x, W_combined.to(dty).T)
            return outNow
        elif(self.isOld):
            self.eval()
            #outbefor
            W_combined_ = torch.matmul(self.W_long_B,self.W_long_A) + self.baseWeight
            outbefor = torch.matmul(x, W_combined_.to(dty).T)
            self.train()
            return outbefor

    def fusionWeight(self):
        self.baseWeight = torch.matmul(self.W_work_B.weight,self.W_work_A.weight).detach() + self.baseWeight
        self.baseWeight.requires_grad = False
        self.W_work_B.weight.data = torch.zeros(self.W_work_B.weight.shape,dtype=self.W_work_B.weight.dtype,requires_grad=True).to(self.baseWeight.device)
        self.W_long_A = self.W_work_A.weight.data.clone()
        self.W_long_B = self.W_work_B.weight.data.clone()
        self.W_long_A.requires_grad = False
        self.W_long_B.requires_grad = False

    def return_lambda(self):
        return self.lambda_ema

    def update_long_memory(self):
        with torch.no_grad():
            self.W_long_A.data = self.lambda_ema * self.W_long_A + (1 - self.lambda_ema) * self.W_work_A.weight.data
            self.W_long_B.data = self.lambda_ema * self.W_long_B + (1 - self.lambda_ema) * self.W_work_B.weight.data

#CURLoRA论文复现
#小瞧CUR了，效果不错啊，也就是说M也可以减小
class CURLoRA(nn.Module):
    def __init__(self,baselayer,CURRank):
        super().__init__()
        #获取原始冻结参数
        self.baseWeight = baselayer.weight
        self.bias = baselayer.bias
        #获取整体Frobenius范数--（1）
        self.FNorm = torch.norm(self.baseWeight,p='fro')
        #获取行/列的L2范数向量-- （2）
        self.RowP_ = torch.norm(self.baseWeight,p=2,dim=-1)
        self.ColumnP_ = torch.norm(self.baseWeight,p=2,dim=-2)
        #通过倒置正则化并归一化得到参数的不重要程度(越大约不重要)-- (3)
        self.ColumnP = (1/self.ColumnP_)/torch.sum(1/self.ColumnP_,dim=-1)
        self.RowP = (1/self.RowP_)/torch.sum(1/self.RowP_,dim=-1)
        #通过(3)不重要归一化矩阵的到前CURRank个不重要行/列
        _,topC = torch.topk(self.ColumnP,CURRank,dim=-1)
        _,topR = torch.topk(self.RowP,CURRank,dim=-1)
        #选取从原矩阵中选取不重要行列-- （4）
        self.ColumnP = self.baseWeight[torch.arange(0,self.baseWeight.size(0)).unsqueeze(1),topC].detach()
        self.RowP = self.baseWeight.T[torch.arange(0,self.baseWeight.T.size(0)).unsqueeze(1),topR].detach()
        #构建可学习参数
        self.U = nn.Linear(CURRank,CURRank,False)
        self.U.weight.data = torch.zeros([CURRank,CURRank],dtype=self.baseWeight.dtype)
        self.U = self.U.to(self.ColumnP.device)

    def returnImportance(self):
        #返回重要参数
        return [self.ColumnP_,self.RowP_]

    def fusionWeight(self):
        CURLoRAOut = torch.matmul(torch.matmul(self.ColumnP,self.U.weight),self.RowP.T).detach()
        self.baseWeight = self.baseWeight + CURLoRAOut
        self.baseWeight.requires_grad = False
        self.U.weight.data = torch.zeros(self.U.weight.shape,dtype=self.U.weight.dtype,requires_grad=True).to(self.baseWeight.device)


    def forward(self,x):
        #还原参数
        CURLoRAOut = torch.matmul(torch.matmul(self.ColumnP,self.U.weight),self.RowP.T)
        #W' = Wfroze(x) + CUR(x) = Wfroze@w + CUR@w = (Wfroze + CUR)(x) (6)
        CURLoRAOut = self.baseWeight + CURLoRAOut

        if(self.bias != None):
            CURLoRAOut = torch.matmul(x,CURLoRAOut.T) + self.bias
        else:
            CURLoRAOut = torch.matmul(x,CURLoRAOut.T)
        return CURLoRAOut

class CABRLoRA(CURLoRA):
    def __init__(self,baselayer,CURRank,ABRank = None,SVD = None,isNeedBias = False,mn = 1.2,decayRatio = 0.99,Ratio = 0.9):
        #SVD输入[U,S,V]
        #先将基础的CURLoRA构造初始化
        super().__init__(baselayer,CURRank)
        self.mn = mn
        self.decayRatio = decayRatio
        self.CURrank =CURRank
        self.Ratio = Ratio
        if(ABRank == None):
            #ABRank默认值将大于4倍的CURRank，为了调整参数能更精细
            ABRank = 4*CURRank
        if(SVD == None):
            #通过SVD拆解得到相关性参数并用于初始化A矩阵 (1)
            U,S,V = torch.svd(self.baseWeight.clone().float())
            U,S,V = U.to(self.baseWeight.dtype), S.to(self.baseWeight.dtype), V.to(self.baseWeight.dtype)
            SVD = [U,S,V]
        #768,768->CURRank,ABRank
        #通过SVD拆解将值压制为CURRank,ABRank形状 （2）
        U = SVD[0][:CURRank,:]
        S = torch.diag(SVD[1][:])
        V = SVD[2][:ABRank,:]
        #删除原有U矩阵
        del self.U
        self.CABR_A = nn.Linear(CURRank,ABRank,False)
        self.CABR_B = nn.Linear(ABRank,CURRank,False)
        #初始化A矩阵，保证A矩阵拥有大多数原矩阵重要信息 （2）
        #print(self.A.weight.data.shape)
        self.CABR_A.weight.data = torch.matmul(torch.matmul(U,S),V.T).T
        self.CABR_B.weight.data = torch.zeros(CURRank,ABRank,dtype=self.baseWeight.dtype)
        self.CABR_B.to(self.baseWeight.device)
        #print(self.B.weight.data.shape)
        if(isNeedBias == True):
            #初始化偏置
            self.LoRAbias = nn.Parameter(torch.zeros(self.baseWeight.shape[0],dtype=self.baseWeight.dtype),requires_grad=True)
        else:
            self.LoRAbias = None
            
    def returnImportance(self):
        return super().returnImportance()

    def get_mergeWeight(self):
        #这里的转置是必要的，不然都变成行了
        merged = torch.matmul(torch.matmul(self.ColumnP,torch.matmul(self.CABR_A.weight.T,self.CABR_B.weight.T)),self.RowP.T)
        return merged

    def SigNorm(self,LossMatrix,eplision = 1e-10,scale = 6):
        #是了，scale绝对不能选太大，否则上下限分的太开4-8即可，暂定6
        mx = torch.max(LossMatrix)
        if((LossMatrix == 0).all()):
            #print("May cause error,max-min == 0")
            return (torch.ones(LossMatrix.shape,dtype=self.baseWeight.dtype)*2).to(LossMatrix.device)
        else:
            #归一化计算差异矩阵，并压制在-0.5-0.5之间 （4）
            Norm = (LossMatrix)/(mx+eplision)-0.5
            #这个scale为的是让整体能够在0-1之间计算
            Norm = LossMatrix*scale
        #将正则值映射在1-2之间用于分辨参数变化大小 (5)
        SigNorm = 2-torch.nn.functional.sigmoid(Norm)
        #这个同样无需保存，只要计算CABRDoRA的计算图即可
        return SigNorm.detach()

    def merge(self,baseOut,CABROut,mn = 1.5,decayRatio = 0.99,Ratio = 0.5,eplision = 1e-10):
        #mn为记录的参数（接近1）
        #Ratio为被记录参数占总百分比
        #merge加入检测抑制机制,CABRWeight是融合的
        self.CABROut = CABROut
        #经过深思熟虑，暂定差异矩阵为倍率差异矩阵矩阵，通过与(4)的联动，可以将整体变化和自身变化总和起来自身变化/最大整体变化->变化大的一定趋近于1反之趋近于2并且是相对于整个矩阵而言 （3）
        LossMatrix = torch.abs((self.CABROut-baseOut)/(baseOut+eplision)).detach()
        LossMatrix = self.SigNorm(LossMatrix,1e-10,6)
        #计算差异小于的值的个数和倍率 (6) (7)
        delta = LossMatrix <= torch.tensor([mn]).to(LossMatrix.device)
        delta = delta.sum().item()
        total = CABROut.shape[-2]*CABROut.shape[-1]
        partition = delta/total
        #为了防止梯度反复计算和异常更新，他必须放在融合之前，decayRatio不能太小，否则最后一步的merge LossMatrix会很麻烦
        #当小于倍率
        if(partition>Ratio):
            #通过已经处于CABRLoRA中计算的重要度行/列（CURLoRA) (2)
            IMTC,IMTR = self.returnImportance()
            #选取最重要的前（hidden-CURrank）//并抑制decayRatio倍 （8）
            topCP,_ = torch.topk(IMTR,k = baseOut.shape[-1],dim=-1)
            a = (baseOut.shape[-1]-self.CURrank)//2
            b = topCP.shape[0]//4
            choosed = torch.multinomial(topCP,max(a,b))
            self.CABROut[:,:,choosed] = self.CABROut[:,:,choosed]*decayRatio
        #融合并相除
        MergeOut = (baseOut + self.CABROut)/LossMatrix
        return MergeOut
    #/2是在相近时，/1是在差异大时，MSE在相近时趋于0，不相近时趋于1,sig增长率与mse相同

    def MergeBasic(self):
        self.baseWeight.data.copy_(self.get_mergeWeight().detach().to(torch.bfloat16) + self.baseWeight.data)
        self.baseWeight.requires_grad = False
        self.CABR_B.weight.data = torch.zeros(self.CABR_B.weight.shape,dtype=self.CABR_B.weight.dtype,requires_grad=True).to(self.baseWeight.device)

    def forward_sepMerged(self, x):
        #获取CABRLoRA的实际值
        CABROut = torch.matmul(x,self.get_mergeWeight().T)
        if(self.bias != None):
            baseOut = torch.matmul(x,self.baseWeight.T) + self.bias
        else:
            baseOut = torch.matmul(x,self.baseWeight.T)
        if(self.LoRAbias != None):
            CABROut = CABROut+self.LoRAbias
        mergeOut = baseOut+CABROut
        ConOut = self.merge(baseOut,mergeOut,self.mn,self.decayRatio,self.Ratio)

        return ConOut
    
    def forward_sep(self, x):
        #NoneMerge
        #获取CABRLoRA的实际值
        CABROut = torch.matmul(x,self.get_mergeWeight().T)
        if(self.bias != None):
            baseOut = torch.matmul(x,self.baseWeight.T) + self.bias
        else:
            baseOut = torch.matmul(x,self.baseWeight.T)
        if(self.LoRAbias != None):
            CABROut = CABROut+self.LoRAbias
        mergeOut = baseOut+CABROut
        #ConOut = self.merge(baseOut,mergeOut,self.mn,self.decayRatio,self.Ratio)

        return mergeOut

    def forward_Merge(self,x):
        #还原参数
        CABRLoRAWeight = self.get_mergeWeight()
        SumWeight = self.baseWeight + CABRLoRAWeight

        #W' = Wfroze(x) + CUR(x) = Wfroze@w + CUR@w = (Wfroze + CUR)(x) (6)
        mergeWeight = self.merge(SumWeight,self.mn,self.decayRatio,self.Ratio)

        if(self.LoRAbias != None):
            if(self.bias != None):
                CURLoRAOut = torch.matmul(x,mergeWeight.T) + self.bias + self.LoRAbias
            else:
                CURLoRAOut = torch.matmul(x,mergeWeight.T) + self.LoRAbias
        else:
            if(self.bias != None):
                CURLoRAOut = torch.matmul(x,mergeWeight.T) + self.bias
            else:
                CURLoRAOut = torch.matmul(x,mergeWeight.T)
        return CURLoRAOut
    
    def forward_CABR_Only(self, x):
        #获取CABRLoRA的实际值
        CABROut = torch.matmul(x,self.get_mergeWeight().T)
        if(self.bias != None):
            baseOut = torch.matmul(x,self.baseWeight.T) + self.bias
        else:
            baseOut = torch.matmul(x,self.baseWeight.T)
        if(self.LoRAbias != None):
            CABROut = CABROut+self.LoRAbias
        mergeOut = baseOut+CABROut
    
        return mergeOut

class CABRLoRA_L(CURLoRA):
    def __init__(self,baselayer,CURRank,ABRank = None,SVD = None,isNeedBias = False,mn = 1.2,decayRatio = 0.99,Ratio = 0.9):
        #SVD输入[U,S,V]
        #先将基础的CURLoRA构造初始化
        super().__init__(baselayer,CURRank)
        self.mn = mn
        self.decayRatio = decayRatio
        self.CURrank =CURRank
        self.Ratio = Ratio
        if(ABRank == None):
            #ABRank默认值将大于4倍的CURRank，为了调整参数能更精细
            ABRank = 4*CURRank
        if(SVD == None):
            #通过SVD拆解得到相关性参数并用于初始化A矩阵 (1)
            U,S,V = torch.svd(self.baseWeight.clone().float())
            U,S,V = U.to(self.baseWeight.dtype), S.to(self.baseWeight.dtype), V.to(self.baseWeight.dtype)
            SVD = [U,S,V]
        #768,768->CURRank,ABRank
        #通过SVD拆解将值压制为CURRank,ABRank形状 （2）
        U = SVD[0][:CURRank,:]
        S = torch.diag(SVD[1][:])
        V = SVD[2][:ABRank,:]
        #删除原有U矩阵
        del self.U
        self.CABR_A = nn.Linear(CURRank,ABRank,False)
        self.CABR_B = nn.Linear(ABRank,CURRank,False)

        self.CABR_A_L = nn.Linear(CURRank,ABRank,False)
        self.CABR_B_L = nn.Linear(ABRank,CURRank,False)

        self.CABR_A_L.weight.data = torch.zeros(CURRank,ABRank,dtype=self.baseWeight.dtype,requires_grad=False)
        self.CABR_B_L.weight.data = torch.zeros(CURRank,ABRank,dtype=self.baseWeight.dtype,requires_grad=False)
        #初始化A矩阵，保证A矩阵拥有大多数原矩阵重要信息 （2）
        #print(self.A.weight.data.shape)
        self.CABR_A.weight.data = torch.matmul(torch.matmul(U,S),V.T).T

        #初始化B矩阵，全为0，近似为一种归一化，同时将有效利用倍率增长的效果
        self.CABR_B.weight.data = torch.zeros(CURRank,ABRank,dtype=self.baseWeight.dtype)
        self.CABR_B.to(self.baseWeight.device)
        #print(self.B.weight.data.shape)
        if(isNeedBias == True):
            #初始化偏置
            self.LoRAbias = nn.Parameter(torch.zeros(self.baseWeight.shape[0],dtype=self.baseWeight.dtype),requires_grad=True)
        else:
            self.LoRAbias = None
            
    def returnImportance(self):
        return super().returnImportance()

    def get_mergeWeight(self):
        #这里的转置是必要的，不然都变成行了
        merged = torch.matmul(torch.matmul(self.ColumnP,torch.matmul(self.CABR_A.weight.T,self.CABR_B.weight.T)),self.RowP.T)
        return merged

    def get_mergeWeightL(self):
        #这里的转置是必要的，不然都变成行了
        merged = torch.matmul(torch.matmul(self.ColumnP,torch.matmul(self.CABR_A_L.weight.T,self.CABR_B_L.weight.T)),self.RowP.T)
        return merged

    def SigNorm(self,LossMatrix,eplision = 1e-10,scale = 6):
        #是了，scale绝对不能选太大，否则上下限分的太开4-8即可，暂定6
        mx = torch.max(LossMatrix)
        if((LossMatrix == 0).all()):
            #print("May cause error,max-min == 0")
            return (torch.ones(LossMatrix.shape,dtype=self.baseWeight.dtype)*2).to(LossMatrix.device)
        else:
            #归一化计算差异矩阵，并压制在-0.5-0.5之间 （4）
            Norm = (LossMatrix)/(mx+eplision)-0.5
            #这个scale为的是让整体能够在0-1之间计算
            Norm = LossMatrix*scale
        #将正则值映射在1-2之间用于分辨参数变化大小 (5)
        SigNorm = 2-torch.nn.functional.sigmoid(Norm)
        #这个同样无需保存，只要计算CABRDoRA的计算图即可
        return SigNorm.detach()

    def merge(self,baseOut,CABROut,mn = 1.5,decayRatio = 0.99,Ratio = 0.5,eplision = 1e-10):
        #mn为记录的参数（接近1）
        #Ratio为被记录参数占总百分比
        #merge加入检测抑制机制,CABRWeight是融合的
        self.CABROut = CABROut
        #经过深思熟虑，暂定差异矩阵为倍率差异矩阵矩阵，通过与(4)的联动，可以将整体变化和自身变化总和起来自身变化/最大整体变化->变化大的一定趋近于1反之趋近于2并且是相对于整个矩阵而言 （3）
        LossMatrix = torch.abs((self.CABROut-baseOut)/(baseOut+eplision)).detach()
        #LossMatrix = torch.abs((self.CABRWeight-self.baseWeightIMT)/(self.baseWeightIMT+eplision)).detach()
        LossMatrix = self.SigNorm(LossMatrix,1e-10,6)
        #计算差异小于的值的个数和倍率 (6) (7)
        delta = LossMatrix <= torch.tensor([mn]).to(LossMatrix.device)
        delta = delta.sum().item()
        total = CABROut.shape[-2]*CABROut.shape[-1]
        partition = delta/total
        #为了防止梯度反复计算和异常更新，他必须放在融合之前，decayRatio不能太小，否则最后一步的merge LossMatrix会很麻烦
        #当小于倍率
        if(partition>Ratio):
            #通过已经处于CABRLoRA中计算的重要度行/列（CURLoRA) (2)
            IMTC,IMTR = self.returnImportance()
            #选取最重要的前（hidden-CURrank）//并抑制decayRatio倍 （8）
            topCP,_ = torch.topk(IMTR,k = baseOut.shape[-1],dim=-1)
            a = (baseOut.shape[-1]-self.CURrank)//2
            b = topCP.shape[0]//4
            choosed = torch.multinomial(topCP,max(a,b))
            self.CABROut[:,:,choosed] = self.CABROut[:,:,choosed]*decayRatio
        #融合并相除
        MergeOut = (baseOut + self.CABROut)/LossMatrix
        return MergeOut
    #/2是在相近时，/1是在差异大时，MSE在相近时趋于0，不相近时趋于1,sig增长率与mse相同

    def MergeBasic(self):
        self.CABR_A_L.weight.data = self.CABR_A.weight.data.clone()
        self.CABR_B_L.weight.data += self.CABR_B.weight.data.clone()
        self.CABR_A_L.requires_grad = False
        self.CABR_B_L.requires_grad = False
        
        self.CABR_B.weight.data = torch.zeros(self.CABR_B.weight.shape,dtype=self.CABR_B.weight.dtype,requires_grad=True).to(self.baseWeight.device)

    def forward_sepMerged(self, x):
        #获取CABRLoRA的实际值
        CABROut = torch.matmul(x,self.get_mergeWeight().T)
        CABRLw = self.get_mergeWeight()+self.baseWeight

        if(self.bias != None):
            baseOut = torch.matmul(x,CABRLw.T) + self.bias
        else:
            baseOut = torch.matmul(x,CABRLw.T)
        if(self.LoRAbias != None):
            CABROut = CABROut+self.LoRAbias
        mergeOut = baseOut+CABROut
        ConOut = self.merge(baseOut,mergeOut,self.mn,self.decayRatio,self.Ratio)

        return ConOut
    
    def forward_sep(self, x):
        #NoneMerge
        #获取CABRLoRA的实际值
        CABROut = torch.matmul(x,self.get_mergeWeight().T)
        if(self.bias != None):
            baseOut = torch.matmul(x,self.baseWeight.T) + self.bias
        else:
            baseOut = torch.matmul(x,self.baseWeight.T)
        if(self.LoRAbias != None):
            CABROut = CABROut+self.LoRAbias
        mergeOut = baseOut+CABROut
        #ConOut = self.merge(baseOut,mergeOut,self.mn,self.decayRatio,self.Ratio)

        return mergeOut

    def forward_Merge(self,x):
        #还原参数
        CABRLoRAWeight = self.get_mergeWeight()
        SumWeight = self.baseWeight + CABRLoRAWeight

        #W' = Wfroze(x) + CUR(x) = Wfroze@w + CUR@w = (Wfroze + CUR)(x) (6)
        mergeWeight = self.merge(SumWeight,self.mn,self.decayRatio,self.Ratio)

        if(self.LoRAbias != None):
            if(self.bias != None):
                CURLoRAOut = torch.matmul(x,mergeWeight.T) + self.bias + self.LoRAbias
            else:
                CURLoRAOut = torch.matmul(x,mergeWeight.T) + self.LoRAbias
        else:
            if(self.bias != None):
                CURLoRAOut = torch.matmul(x,mergeWeight.T) + self.bias
            else:
                CURLoRAOut = torch.matmul(x,mergeWeight.T)
        return CURLoRAOut


class CABDoRAConfig:
    """
    不仅用于CABRDORA
    """
    def __init__(self,CURdivideRank,ABdivideRank,isNeedBias = False,mn = 1.5,decayRatio = 0.99,Ratio = 0.5,Type = "CABR-DoRA",Is_CABR_Only = False):
        self.Type = Type
        self.CURdivideRank = CURdivideRank
        self.ABdivideRank = ABdivideRank
        self.isNeedBias = isNeedBias
        self.Is_CABR_Only = Is_CABR_Only
        self.mn = mn
        self.decayRatio = decayRatio
        self.Ratio = Ratio
        

# 整体测试
# torch.manual_seed(114514)
# inp = nn.Linear(768,768)
# # #规定，只输入weight
# aConfig = CABDoRAConfig(4,1,True,1.2,0.99,0.5)
# aConfig.baseWeight = inp.weight
# aConfig.CURRank = (inp.weight.shape[0])//aConfig.CURdivideRank
# aConfig.ABRank = (inp.weight.shape[0])//aConfig.ABdivideRank
# a = CABRDoRA(aConfig)
# a(torch.randn(1,768))

#print(a.merge(torch.ones(768,768)*390),1.5)
#print(a.merge(inp+torch.nn.functional.dropout(torch.ones(768,768),0.5)),1.2)
#问题不是出在signorm上了，现在是差异太小，没有特别大的变化则老是一致
#print(a(torch.randn(768,768)))
# b = CURLoRA(torch.randn(768,768),16)
# b(torch.randn(1,3,768))
# c = CABRLoRA(torch.randn(768,768),16,400)
# c(torch.randn(1,3,768))