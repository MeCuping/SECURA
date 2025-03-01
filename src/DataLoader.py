import json
import random

class DataLoader:
    def __init__(self,pth,max):
        self.pth = pth
        self.data = self.datas()
        self.max = max

    def datas(self):
        a = []
        b = []
        with open(self.pth,'r') as r:
            datas = json.load(r)
        for i in datas:
            a.append(i["input"])
            b.append(i["output"])
        return (a,b)
    
    def GetData(self):
        rand = random.randint(0,len(self.data[0])-1)
        return self.data[0][rand],self.data[1][rand]

    def preprocess(self,tokenizer,commands = None):
        #检查问题是不是因为输入...
        NoneTemp = False
        if getattr(tokenizer, "chat_template", None) is None:
            NoneTemp = True
        #为了检测输入不超限
        while True:
            DataProcessing = self.GetData()
            input = DataProcessing[0]
            label = DataProcessing[1]
            if(commands == None):
                commands = "You are a helpful AI assistant."
            messages_Q = [
            {"role": "system", "content": commands},
            {"role": "user", "content": DataProcessing[0]}
            ]
            messages_A = [
                {"role": "system", "content": commands},
                {"role": "user", "content": DataProcessing[0]},
                {"role": "assistant", "content": DataProcessing[1]}
            ]
            # if NoneTemp == False:
            if True:
                check = tokenizer.apply_chat_template(
                    messages_Q,
                    tokenize=True,
                    add_generation_prompt=True
                )
            else:
                check = tokenizer.encode("System:[START]"+commands+"[END]\n"+"User:[START]"+DataProcessing[0]+"[END]\n"+"Assistent:[START]")
            if(len(check) < self.max):
                start_idx = len(check)
                break
            else:
                print("warning, input bigger than max may cause memory run out!")
        # if(NoneTemp == False):
        messages_Q = [
            {"role": "system", "content": commands},
            {"role": "user", "content": DataProcessing[0]},
            #TMD更改模型这里还得改我真是操了要改特殊标记
            #{"role": "assistant", "content": "[EMPTY]" + DataProcessing[1]}
            {"role": "assistant", "content": "[EMPTY]" + DataProcessing[1]}
            ]
        input_total = tokenizer.apply_chat_template(
            messages_Q,
            tokenize=True,
            add_generation_prompt=False
        )
        #预处理为因果类格式
        input_total = input_total[:-1]
        label_total = tokenizer.apply_chat_template(
            messages_A,
            tokenize=True,
            add_generation_prompt=False
        )
        #此处要调整成到最大范围
        return input_total[:self.max],label_total[:self.max],input,label,start_idx
