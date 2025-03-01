import os
from src import BasicLoRATrainer,MinePeft

def list_folders_and_files(root_folder):
    folder_contents = {}
    
    # 获取根文件夹下的所有子文件夹列表
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    
    # 使用 for 循环遍历每个子文件夹
    for folder in subfolders:
        # 获取子文件夹内的文件列表
        files = os.listdir(folder)
        folder_contents[folder] = [os.path.join(folder, file) for file in files]
        
    return subfolders,folder_contents

#现在只是叫dividen了，实际上就是Rank
#还是得休息休息的，训完一个大模型缓个半天
def BatchTrainingACEXP2(llm,train_path,dics):
    """
    用于批量实验实验2,train_paths接受字符串数组
    """
    #0->MultiChoice 1->QA
    for m,dic in enumerate(dics): 
        for j,pths in enumerate(train_path):
            paths,pathDic = list_folders_and_files(pths)
            for n,i in enumerate(paths):
                trapath = pathDic[i][0]
                tespath = pathDic[i][1]
                BasicLoRATrainer.trainer(dic,llm,trapath,tstName=f"{j}_{n}",testpath=tespath,expID=2) 


def BatchTrainingACEXP3(llm,train_path,dics):
    """
    用于批量实验实验3,存grad
    """
    #0->MultiChoice 1->QA
    for m,dic in enumerate(dics): 
        for j,pths in enumerate(train_path):
            paths,pathDic = list_folders_and_files(pths)
            for n,i in enumerate(paths):
                trapath = pathDic[i][0]
                tespath = pathDic[i][1]
                BasicLoRATrainer.trainer(dic,llm,trapath,tstName=f"{j}_{n}",testpath=tespath,expID=3) 

def BatchTrainingACEXP4(llm,train_path,dics,testDir = "None"):
    """
    用于批量实验实验4,灾难性遗忘
    """
    #0->MultiChoice 1->QA
    for m,dic in enumerate(dics): 
        dic["testDir"] = testDir
        for j,pths in enumerate(train_path):
            paths,pathDic = list_folders_and_files(pths)
            for n,i in enumerate(paths):
                trapath = pathDic[i][0]
                tespath = pathDic[i][1]
                BasicLoRATrainer.trainer(dic,llm,trapath,tstName=f"{j}_{n}",testpath=tespath,expID=4,mx=600) 

def BatchTrainingACEXP5(llm,train_path,dics,testDir = "None",change = 2000):
    """
    用于批量实验实验5，过拟合
    """
    #0->MultiChoice 1->QA
    tra = []
    tes = []
    for m,dic in enumerate(dics): 
        dic["testDir"] = testDir
        for j,pths in enumerate(train_path):
            paths,pathDic = list_folders_and_files(pths)
            for n,i in enumerate(paths):
                trapath = pathDic[i][0]
                tespath = pathDic[i][1]
                tra.append(trapath)
                tes.append(tespath)
        BasicLoRATrainer.trainer(dic,llm,tra,tstName=f"{j}_{n}",testpath=None,expID=5,change = change,mx=600) 

def AddingTestSVDAnalysis(llm,train_path,dics):
    """
    测量知识保留程度
    """
    #0->MultiChoice 1->QA
    for m,dic in enumerate(dics): 
        for j,pths in enumerate(train_path):
            paths,pathDic = list_folders_and_files(pths)
            for n,i in enumerate(paths):
                trapath = pathDic[i][0]
                tespath = pathDic[i][1]
                BasicLoRATrainer.trainer(dic,llm,trapath,tstName=f"{j}_{n}",testpath=tespath,expID=0) 