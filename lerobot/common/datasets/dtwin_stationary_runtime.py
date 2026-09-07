"""017799 stationary-task deployment contract; distinct from SimXR 004321.

DTwin episode metadata marks raw 0:45 valid, including roll/pitch/height.
Only raw velocities45:51 are invalid for the stationary Cup episodes. Preserve
all 41 joint/hand outputs AND trained base height/roll/pitch. No learned adapter.
"""
import dataclasses
import json
import os
from pathlib import Path
import numpy as np
import torch
from lerobot.common.datasets.data_config import LeRobotHumanoidS0S1DataConfig
from lerobot.common.datasets.torch_transforms import Group

@dataclasses.dataclass(frozen=True)
class StationaryInput:
    transform: object
    def __call__(self,data):
        data=dict(data)
        mask=np.ones(51,dtype=bool)
        mask[45:51]=False
        data['state_valid_mask']=mask
        result=self.transform(data)
        audit=os.environ.get('CONTRACT_AUDIT_DIR')
        if audit:
            with (Path(audit)/'inputs.jsonl').open('a') as stream:
                stream.write(json.dumps({'prompt':data.get('prompt'),
                    'raw_state':np.asarray(data['observation/state']).tolist(),
                    'state50':result['state'].detach().cpu().tolist(),
                    'valid50':result['state_valid_mask'].tolist()})+'\n')
        return result

@dataclasses.dataclass(frozen=True)
class StationaryOutput:
    def __call__(self,data):
        if 'actions' not in data:return data
        result=dict(data);original=data['actions']
        actions=original.clone() if isinstance(original,torch.Tensor) else np.array(original,copy=True)
        assert actions.shape[-1]==50
        audit=os.environ.get('CONTRACT_AUDIT_DIR')
        if audit:
            values=actions.detach().cpu().numpy() if isinstance(actions,torch.Tensor) else actions
            with (Path(audit)/'outputs.jsonl').open('a') as stream:
                stream.write(json.dumps({'body_before':values[...,:25].tolist(),
                    'base_before':values[...,25:34].tolist()})+'\n')
        actions[...,26:32]=0
        result['actions']=actions
        return result

@dataclasses.dataclass(frozen=True)
class DtwinStationaryRuntimeConfig(LeRobotHumanoidS0S1DataConfig):
    def create(self,assets_dirs,model_config):
        cfg=super().create(assets_dirs,model_config)
        inputs=list(cfg.data_transforms.inputs)
        inputs[0]=StationaryInput(inputs[0])
        outputs=list(cfg.data_transforms.outputs)+[StationaryOutput()]
        return dataclasses.replace(cfg,data_transforms=Group(inputs=tuple(inputs),outputs=tuple(outputs)))
