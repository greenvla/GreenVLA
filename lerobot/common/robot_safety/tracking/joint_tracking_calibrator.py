"""One pose-conditioned learned command bias; no task, phase or trajectory lookup."""
import json
from pathlib import Path
import numpy as np

def features(body25,height,roll,pitch):
    q=np.asarray(body25,dtype=float)
    if q.shape!=(25,) or not np.isfinite(q).all():raise ValueError('Invalid body25')
    values=np.r_[q[:23],np.sin(q[:23]),float(height),float(roll),float(pitch)]
    if not np.isfinite(values).all():raise ValueError('Invalid base pose')
    return values

class JointTrackingCalibrator:
    def __init__(self,path):
        self.meta=json.loads(Path(path).read_text())
        self.center=np.asarray(self.meta['center']);self.scale=np.asarray(self.meta['scale'])
        self.coef=np.asarray(self.meta['coef']);self.intercept=np.asarray(self.meta['intercept'])
        self.caps=np.asarray(self.meta['correction_caps_rad'])
        if self.center.shape!=(49,) or self.scale.shape!=(49,) or self.coef.shape!=(49,23) or self.intercept.shape!=(23,) or self.caps.shape!=(23,):
            raise ValueError('Invalid calibration layout')
        if not all(np.isfinite(a).all() for a in (self.center,self.scale,self.coef,self.intercept,self.caps)):
            raise ValueError('Nonfinite calibration')
        if not (self.scale>0).all() or not (self.caps>0).all():raise ValueError('Invalid scales/caps')

    def correction(self,body25,height,roll,pitch):
        z=np.clip((features(body25,height,roll,pitch)-self.center)/self.scale,-3,3)
        return np.clip(np.einsum('i,ij->j',z,self.coef,optimize=False)+self.intercept,-self.caps,self.caps)
