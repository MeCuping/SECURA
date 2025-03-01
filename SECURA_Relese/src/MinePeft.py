import re
import os
import json
import torch
import math
import logging
import warnings
from typing import Any, Optional
import torch.nn as nn
from peft import LoraConfig
from transformers import AutoModelForCausalLM
#测试seq
#from src import CABR_DoRA
from src import CABR_DoRA_Sep as CABR_DoRA
from itertools import chain
from typing import Any, Union
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.tuners.lora.layer import LoraLayer
from peft.utils.integrations import gather_params_ctx
from peft.tuners.lora.layer import Linear as LoRALinear
from peft.utils import get_quantization_config
from peft.config import PeftConfig
from peft.mixed_model import PeftMixedModel
from peft.peft_model import PeftModel
from peft.mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING
from peft.utils import _prepare_prompt_learning_config
from peft.tuners.lora.model import LoraModel

logger = logging.getLogger(__name__)

def NewLoraModel__init__(self, model, config, adapter_name,CABRDoRAConfig = None) -> None:
        super(LoraModel,self).__init__(model, config, adapter_name)
        self.CABRDoRAConfig = CABRDoRAConfig

def NewLoraModel_create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "ephemeral_gpu_offload": False,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        quant_methods = ["gptq", "aqlm", "awq"]
        for quant_method in quant_methods:
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                kwargs[f"{quant_method}_quantization_config"] = quantization_config

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from peft.tuners.adalora import AdaLoraLayer

        if isinstance(target, LoraLayer) and not isinstance(target, AdaLoraLayer):
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
                CABRDoRAConfig = self.CABRDoRAConfig
            )
        else:
            new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

def NewPeftModel__init__(
        self,
        model,
        peft_config,
        adapter_name: str = "default",
        autocast_adapter_dtype: bool = True,
        CABRDoRAConfig = None
    ) -> None:
        super(PeftModel,self).__init__()
        self.modules_to_save = None
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        # These args are special PEFT arguments that users can pass. They need to be removed before passing them to
        # forward.
        self.special_peft_forward_args = {"adapter_names"}

        self._is_prompt_learning = False
        if self._is_prompt_learning:
            self._peft_config = {adapter_name: peft_config}
            self.base_model = model
            self.add_adapter(adapter_name, peft_config)
        else:
            self._peft_config = None
            cls = LoraModel
            self.base_model = cls(model, {adapter_name: peft_config}, adapter_name,CABRDoRAConfig)
            self.set_additional_trainable_modules(peft_config, adapter_name)

        if hasattr(self.base_model, "_cast_adapter_dtype"):
            self.base_model._cast_adapter_dtype(
                adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype
            )

        if getattr(model, "is_gradient_checkpointing", True):
            model = self._prepare_model_for_gradient_checkpointing(model)

        # the `pretraining_tp` is set for some models to simulate Tensor Parallelism during inference to avoid
        # numerical differences, https://github.com/pytorch/pytorch/issues/76232 - to avoid any unexpected
        # behavior we disable that in this line.
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
            self.base_model.config.pretraining_tp = 1

def NewLinear__init__(
    self,
    base_layer,
    adapter_name: str,
    r: int = 0,
    lora_alpha: int = 1,
    lora_dropout: float = 0.0,
    fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
    is_target_conv_1d_layer: bool = False,
    init_lora_weights: Union[bool, str] = True,
    use_rslora: bool = False,
    use_dora: bool = False,
    CABRDoRAConfig = None,
    **kwargs,
) -> None:
    super(LoRALinear,self).__init__()
    LoraLayer.__init__(self, base_layer, **kwargs)
    self.fan_in_fan_out = fan_in_fan_out

    self._active_adapter = adapter_name
    self.update_layer(
        adapter_name,
        r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        init_lora_weights=init_lora_weights,
        use_rslora=use_rslora,
        use_dora=use_dora,
        CABRDoRAConfig = CABRDoRAConfig
    )
    self.is_target_conv_1d_layer = is_target_conv_1d_layer

def NewLoraLayer_update_layer(
    self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora: bool = False,CABRDoRAConfig = None
):
    # This code works for linear layers, override for other layer types
    if r <= 0:
        if(self.CABRDoRAConfig.Type == "CABR-LoRA" or self.CABRDoRAConfig.Type == "CABR-LoRA_Only"):
            #修改测试
            baselayer = self.base_layer
            #CURRank = int(self.base_layer.weight.shape[0]//self.CABRDoRAConfig.CURdivideRank)
            CURRank = self.CABRDoRAConfig.CURdivideRank
            #ABRank = int(self.base_layer.weight.shape[1]//self.CABRDoRAConfig.ABdivideRank)
            ABRank = self.CABRDoRAConfig.ABdivideRank
            isNeedBias = self.CABRDoRAConfig.isNeedBias
            mn = self.CABRDoRAConfig.mn
            decayR = self.CABRDoRAConfig.decayRatio
            Ratio = self.CABRDoRAConfig.Ratio
            self.CABRLoRA = CABR_DoRA.CABRLoRA(baselayer,CURRank,ABRank,isNeedBias=isNeedBias,mn=mn,decayRatio=decayR,Ratio=Ratio)
        elif(self.CABRDoRAConfig.Type == "CABR-LoRA_L"):
            #修改测试
            baselayer = self.base_layer
            #CURRank = int(self.base_layer.weight.shape[0]//self.CABRDoRAConfig.CURdivideRank)
            CURRank = self.CABRDoRAConfig.CURdivideRank
            #ABRank = int(self.base_layer.weight.shape[1]//self.CABRDoRAConfig.ABdivideRank)
            ABRank = self.CABRDoRAConfig.ABdivideRank
            isNeedBias = self.CABRDoRAConfig.isNeedBias
            mn = self.CABRDoRAConfig.mn
            decayR = self.CABRDoRAConfig.decayRatio
            Ratio = self.CABRDoRAConfig.Ratio
            self.CABRLoRA = CABR_DoRA.CABRLoRA_L(baselayer,CURRank,ABRank,isNeedBias=isNeedBias,mn=mn,decayRatio=decayR,Ratio=Ratio)
        elif(self.CABRDoRAConfig.Type == "CUR-LoRA"):
            baselayer = self.base_layer
            CURRank = self.CABRDoRAConfig.CURdivideRank
            if(CURRank > baselayer.weight.shape[0]):
                CURRank = baselayer.weight.shape[0]
            self.CURLoRA = CABR_DoRA.CURLoRA(baselayer,CURRank)
        elif(self.CABRDoRAConfig.Type == "I-LoRA"):
            baselayer = self.base_layer
            Rank = self.CABRDoRAConfig.CURdivideRank
            self.I_LoRA = CABR_DoRA.I_LoRA(baselayer,Rank)
    else:
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)

    self.r[adapter_name] = r
    self.lora_alpha[adapter_name] = lora_alpha
    if lora_dropout > 0.0:
        lora_dropout_layer = nn.Dropout(p=lora_dropout)
    else:
        lora_dropout_layer = nn.Identity()

    self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        
    if use_rslora:
        self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
    else:
        #self.scaling[adapter_name] = lora_alpha / r
        pass

    # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
    if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
        with gather_params_ctx(self.get_base_layer().weight):
            self.pissa_init(adapter_name, init_lora_weights)
    elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
        with gather_params_ctx(self.get_base_layer().weight):
            self.olora_init(adapter_name)
    elif init_lora_weights == "loftq":
        with gather_params_ctx(self.get_base_layer().weight):
            self.loftq_init(adapter_name)
    elif init_lora_weights:
        self.reset_lora_parameters(adapter_name, init_lora_weights)
    # call this before dora_init
    #self._move_adapter_to_device_of_base_layer(adapter_name)

    if use_dora:
        self.dora_init(adapter_name)
        self.use_dora[adapter_name] = True
    else:
        self.use_dora[adapter_name] = False

    self.set_adapter(self.active_adapters)

def NewLinear_CABRDoRA_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            # lora_A = self.lora_A[active_adapter]
            # lora_B = self.lora_B[active_adapter]
        #dropout = self.lora_dropout[active_adapter]
        #x = dropout(x)
        if(self.CABRDoRAConfig.Type == "CABR-LoRA"):
            #result = self.CABRLoRA.forward_sep(x)
            if(self.CABRDoRAConfig.Is_CABR_Only == True):
                result = self.CABRLoRA.forward_CABR_Only(x)
            else:
                result = self.CABRLoRA.forward_sepMerged(x)
        elif(self.CABRDoRAConfig.Type == "CABR-LoRA_L"):
            #result = self.CABRLoRA.forward_sep(x)
            result = self.CABRLoRA.forward_sepMerged(x)
        elif(self.CABRDoRAConfig.Type == "CUR-LoRA"):
            result = self.CURLoRA(x)
    return result

#必须先初始化他
def RefactoryLoRALinear(CABRConfig):
    LoRALinear.CABRDoRAConfig = CABRConfig
    LoraLayer.CABRDoRAConfig = CABRConfig
    LoraLayer.update_layer = NewLoraLayer_update_layer
    LoRALinear.__init__ = NewLinear__init__
    LoRALinear.forward = NewLinear_CABRDoRA_forward
    LoraModel.__init__ = NewLoraModel__init__
    LoraModel._create_and_replace = NewLoraModel_create_and_replace
    PeftModel.__init__ = NewPeftModel__init__

def savePretrained(model,filepath,ParamDic,Traintype = None,hidden = None):
    TrainType = ParamDic["Traintype"]
    save_Dicts = {}
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        print(f"Created folder: {filepath}")
    saveName = os.path.join(filepath, "Config.json")
    with open(saveName, 'w', encoding="utf-8") as file:
            json.dump(ParamDic, file, indent=4)
    if(TrainType == "CABR-LoRA"):
        merge_peft_model(model,Traintype,hidden)
        torch.save(model.state_dict(), "model.pth")
        return
    if(TrainType == "CABR-LoRA_L"):
        for name, param in model.named_parameters():
            if "CABR" in name:
                save_Dicts[name] = param
    elif(TrainType == "CUR-LoRA"):
        for name, param in model.named_parameters():
            if "CUR" in name:
                save_Dicts[name] = param
    torch.save(save_Dicts, os.path.join(filepath, "model.pth"))
    print(f"Selected parameters saved to {filepath}")

def load_selected_parameters(model, filepath):
    ParamPath = os.path.join(filepath, "model.pth")
    ConfigPath = os.path.join(filepath, "Config.json")
    with open(ConfigPath,"r",encoding="utf-8") as r:
        Config = json.load(r)
    selected_params = torch.load(ParamPath,weights_only=True)
    peftConfig = LoraConfig(
                target_modules=Config["target_modules"],  # 只对atten.c_proj微调
                r=Config["BasicLoRARank"],  # LoRA rank
                lora_alpha=16,  # LoRA scaling factor
                lora_dropout=Config["lora_dropout"],  # LoRA dropout
                bias="lora_only",  # 不微调bias
            )
    aConfig = CABR_DoRA.CABDoRAConfig(Config["CURdivideRank"],Config["ABdivideRank"],Config["isNeedBias"],Config["EdgeNum"],Config["decayRatio"],Config["Percent"],Config["Traintype"])
    RefactoryLoRALinear(aConfig)
    model = Newget_peft_model(model,peftConfig,aConfig)
    for name, param in selected_params.items():
        if name in model.state_dict():
            model.state_dict()[name].copy_(param)
            #print(f"Loaded parameter: {name}, shape: {param.shape}")
        else:
            print(f"Parameter {name} not found in the model.")
    model.to(model.device)

def Newget_peft_model(
    model,
    peft_config: PeftConfig,
    CABR_DoRAConfig = None,
    adapter_name: str = "default",
    mixed: bool = False,
    autocast_adapter_dtype: bool = True,
    revision: Optional[str] = None,
):
    """
    Returns a Peft model object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]):
            Model to be wrapped.
        peft_config ([`PeftConfig`]):
            Configuration object containing the parameters of the Peft model.
        adapter_name (`str`, `optional`, defaults to `"default"`):
            The name of the adapter to be injected, if not provided, the default adapter name is used ("default").
        mixed (`bool`, `optional`, defaults to `False`):
            Whether to allow mixing different (compatible) adapter types.
        autocast_adapter_dtype (`bool`, *optional*):
            Whether to autocast the adapter dtype. Defaults to `True`. Right now, this will only cast adapter weights
            using float16 or bfloat16 to float32, as this is typically required for stable training, and only affect
            select PEFT tuners.
        revision (`str`, `optional`, defaults to `main`):
            The revision of the base model. If this isn't set, the saved peft model will load the `main` revision for
            the base model
    """
    model_config = getattr(model, "config", {"model_type": "custom"})
    if hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()

    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)

    if revision is not None:
        if peft_config.revision is not None and peft_config.revision != revision:
            warnings.warn(
                f"peft config has already set base model revision to {peft_config.revision}, overwriting with revision {revision}"
            )
        peft_config.revision = revision

    if mixed:
        # note: PeftMixedModel does not support autocast_adapter_dtype, so don't pass it
        return PeftMixedModel(model, peft_config, adapter_name=adapter_name)

    if peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys() and not peft_config.is_prompt_learning:
        return PeftModel(model, peft_config, adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype,CABRDoRAConfig = CABR_DoRAConfig)

    if peft_config.is_prompt_learning:
        peft_config = _prepare_prompt_learning_config(peft_config, model_config)
    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](
        model, peft_config, adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype
    )

def merge_peft_model(model, lora_model_path):
    
    # 加载并应用 LoRA 微调权重
    model = PeftModel.from_pretrained(model, lora_model_path)
    
    # 将 LoRA 权重合并到基础模型
    model.merge_and_unload()
    
    return model

def ILoRAUpdate(model,layer = 28):
    for i in range(layer):
        model.base_model.model.model.layers[i].self_attn.q_proj.I_LoRA.update_long_memory()
        model.base_model.model.model.layers[i].self_attn.v_proj.I_LoRA.update_long_memory()

def mergeSelfModel(model,traintyp,layer = 28):
    if(traintyp == "CABR-LoRA"):
        for i in range(layer):
            model.base_model.model.model.layers[i].self_attn.q_proj.CABRLoRA.MergeBasic()
            model.base_model.model.model.layers[i].self_attn.v_proj.CABRLoRA.MergeBasic()
    elif(traintyp == "CABR-LoRA_L"):
        for i in range(layer):
            model.base_model.model.model.layers[i].self_attn.q_proj.CABRLoRA.MergeBasic()
            model.base_model.model.model.layers[i].self_attn.v_proj.CABRLoRA.MergeBasic()
    elif(traintyp == "I-LoRA"):
        for i in range(layer):
            model.base_model.model.model.layers[i].self_attn.q_proj.I_LoRA.fusionWeight()
            model.base_model.model.model.layers[i].self_attn.v_proj.I_LoRA.fusionWeight()
    elif(traintyp == 'CUR-LoRA'):
        for i in range(layer):
            model.base_model.model.model.layers[i].self_attn.q_proj.CURLoRA.fusionWeight()
            model.base_model.model.model.layers[i].self_attn.v_proj.CURLoRA.fusionWeight()
