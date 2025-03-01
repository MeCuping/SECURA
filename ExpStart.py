from src import ExpGradioFunctions
from src import tools

#M1
dic1 = {
"Traintype" : "CABR-LoRA",
#CABR时BasicLoRARank=0
"BasicLoRARank" : 0,
"lora_dropout" : 0.1,
"FusionStep" : 1,
"lr" : 1e-3,
"target_modules" : ["q_proj","v_proj"],
#以下部分为CABR专属，不用CABR训练时随意填入即可
"isNeedBias" : True,
"epoch" : 2010,
"tstStep" : 500,
"saveStep" : 2000,
"ckStep" : 2000,
"EdgeNum" : 1.2,
"CURdivideRank" : 150,
"ABdivideRank" : 200,
"decayRatio" : 0.99,
"Percent" : 0.9
}

#M2
dic2 = {
"Traintype" : "CABR-LoRA_L",
#CABR时BasicLoRARank=0
"BasicLoRARank" : 0,
"lora_dropout" : 0.1,
"FusionStep" : 1,
"lr" : 1e-3,
"target_modules" : ["q_proj","v_proj"],
#以下部分为CABR专属，不用CABR训练时随意填入即可
"isNeedBias" : True,
"epoch" : 2010,
"tstStep" : 500,
"saveStep" : 2000,
"ckStep" : 500,
"EdgeNum" : 1.2,
"CURdivideRank" : 150,
"ABdivideRank" : 200,
"decayRatio" : 0.99,
"Percent" : 0.9
}

#e.g.Usage
dic = [dic1]
llm = r"E:\tch\0 LLMs\Qwen2-1.5B-Instruct"

"""
!!!!
注意，跑完记得将outputs内变体文件整理到LoRAs/对应LLM名称/内
并清空outputs各个变体文件内的数据
"""

"""
此处
EXP0/AddingTestSVDAnalysis->SVD Norm 
EXP2->cabr实验1——性能
EXP3-> grad
EXP4->cabr实验2——极端lora忘却实验
EXP5->cabr实验3——对应灾难性遗忘metrics对比
"""

trainpathsEXP2 = [r"E:\tch\CABR-DoRA\DataSets\EXP1\Tst"]
#e.g. Useage
#ExpGradioFunctions.BatchTrainingACEXP2(llm,trainpathsEXP2,dic)
#ExpGradioFunctions.BatchTrainingACEXP4(llm,trainpathsEXP4,dic,r"E:\tch\CABR-DoRA\DataSets\EXP5\test\fusionTest.json")
#ExpGradioFunctions.BatchTrainingACEXP3(llm,trainpathsEXP3,dic)
#ExpGradioFunctions.BatchTrainingACEXP5(llm,trainpathsEXP5,dic,r"E:\tch\CABR-DoRA\DataSets\EXP5\test\fusionTest.json",change = 2000)
ExpGradioFunctions.AddingTestSVDAnalysis(llm,trainpathsEXP2,dic)
