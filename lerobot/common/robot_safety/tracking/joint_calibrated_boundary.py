"""Command feedback boundary, with optional learned tracking correction."""
import copy
import numpy as np

class JointCalibratedBoundary:
    def __init__(self,wire,model=None):
        if wire.proprio not in ('all','measured'):raise ValueError('Unsupported feedback mode')
        self.wire=wire;self.model=model;self.last_model_body=None;self.last_correction=None

    def decode_observation(self,observation):
        t=float(observation['t'])
        if observation.get('reset') or self.wire.last_t is None or t<self.wire.last_t:
            self.last_model_body=None
        result=self.wire.decode_observation(observation)
        # Physical measurements already include the robot's actual response to
        # compensation. Never subtract command offsets or replace them with intent.
        if self.wire.proprio=='measured':return result
        if self.last_model_body is None:return result
        # Stock echoes corrected wire commands for torso but idle legs. Return
        # the *uncorrected*, already-issued model command for both. Never feed
        # the added correction back as intent or compensate measured attitude.
        result=dict(result);state=dict(result['state'])
        torso=np.asarray(state['torso_joint_pos']).copy()
        if torso.shape!=(13,):raise ValueError('Invalid torso state')
        torso[:11]=self.last_model_body[12:23]
        state['torso_joint_pos']=torso;state['legs_joint_pos']=self.last_model_body[:12].copy()
        result['state']=state
        return result

    def encode_response(self,response):
        if not response['actions_list']:raise ValueError('Empty response')
        encoded=dict(response);rows=[]
        for row in response['actions_list']:
            body=np.r_[row['legs_joint_pos'],row['torso_joint_pos']]
            base=row['base_command']
            delta=(np.zeros(23,dtype=float) if self.model is None else
                   self.model.correction(body,base['root_height'],base['roll'],base['pitch']))
            if delta.shape!=(23,) or not np.isfinite(delta).all():raise ValueError('Invalid correction')
            corrected=body.copy();corrected[:23]+=delta
            action=copy.deepcopy(row)
            action['legs_joint_pos']=corrected[:12].tolist();action['torso_joint_pos']=corrected[12:25].tolist()
            rows.append(action);self.last_model_body=body.copy();self.last_correction=delta.copy()
        encoded['actions_list']=rows
        return self.wire.encode_response(encoded)
