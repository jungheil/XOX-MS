import sys
sys.path.append('../src')
import nibabel as nib
import os
import glob
from tqdm import tqdm
import numpy as np
import mindspore.dataset as de
import mindspore.dataset.vision as vs
from model import get_model
from utils.medicine import GetBodyArea, ReSp
import mindspore as ms
from mindspore import context
from mindspore import ops as P
from mindspore import load_checkpoint, load_param_into_net, nn
import cv2

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
# context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
context.set_context(enable_graph_kernel=False)

nii_path = '/home/cat/lrx/datasets/kits19/test'
output_dir = '/home/cat/lrx/test'
ckpt_file = '/home/cat/lrx/git/xox_ms/weight/final.ckpt'


class SingleCT():
    def __init__(self,nii ,channels = 3,win_centre = 62,win_middle = 482):
        super().__init__()
        self.ct = nii

        self.len = self.ct.shape[0]-(channels+1)//2
        self.channels = channels
        
        self.x, self.y, self.w, self.h = GetBodyArea(self.ct)
        self.ct = ReSp(self.ct,self.x, self.y, self.w, self.h )
        win_min = win_centre-win_middle/2
        win_max = win_centre+win_middle/2
        self.ct[self.ct<win_min]=win_min
        self.ct[self.ct>win_max]=win_max
        ct_norm = (self.ct-self.ct.mean())/self.ct.std()
        self.ct = ct_norm - ct_norm.min()
        self.ct = np.float32(self.ct).transpose((1,2,0))
    
    def get_area(self):
        return self.x, self.y, self.w, self.h

    def __getitem__(self, index):
        return self.ct[:,:,index:index+self.channels]

    def __len__(self):
        return self.len

class OutAdt(nn.Cell):
    def __init__(self):
        super().__init__(auto_prefix=False)
        self.lq = P.LessEqual()
        self.ln = P.LogicalNot()
        self.lgand = P.LogicalAnd()
        
    def construct(self, x):
        x = P.sigmoid(x)
        bg = 1-x[:,0:1,...]
        sz = x[:,0:1,...] * (1-x[:,1:2,...])
        zl = x[:,0:1,...] * x[:,1:2,...]
        
        out = P.concat((bg,sz,zl), axis=1)

        return out
class Pred:
    def __init__(self,ckpt,channels=3) -> None:
        self.num_parallel_workers=1
        self.channels = channels
        self.net = get_model({'type': 'UNET3PLUS','n_classes':2})
        load_param_into_net(self.net, load_checkpoint(ckpt))
        
        self.model = ms.Model(
            network=self.net
        )
        self.argmax = P.Argmax(axis=1)
        self.adt = OutAdt()
    
    def get(self, nii):
        dataset = SingleCT(nii,self.channels)
        area = dataset.get_area()
        ds = de.GeneratorDataset(
            dataset,
            column_names=["img"],
            num_parallel_workers=self.num_parallel_workers,
            shuffle=False
        )
        ds = ds.map(
            operations=vs.Resize([512,512]),
            input_columns=['img'],
            num_parallel_workers=self.num_parallel_workers,
        )
        ds = ds.map(
            operations=[vs.HWC2CHW()],
            input_columns=['img'],
            num_parallel_workers=self.num_parallel_workers,
        )
        ds = ds.batch(1)
        
        out = []
        
        for ct in ds:
            ct = ct[0]
            seg = self.model.predict(ct)
            if isinstance(seg, tuple):
                seg = seg[0]
                seg = self.adt(seg)
            seg = self.argmax(seg).asnumpy().squeeze()
            out.append(seg)
        pad_front = [np.zeros(out[0].shape) for _ in range(self.channels // 2)]
        pad_end = [np.zeros(out[0].shape) for _ in range(self.channels // 2)]
        
        out = np.array(pad_front+out+pad_end, dtype=np.float32)
        out = ReSp(out,*area,restore=True,interpolation=cv2.INTER_NEAREST)
        return out.astype(np.uint8)
        
        
os.makedirs(output_dir, exist_ok=True)

nii_file = glob.glob(os.path.join(nii_path,'*','imaging.nii.gz'))
p = Pred(ckpt_file,3)
for nii in tqdm(nii_file):
    ct = nib.load(nii)
    hd = ct.header
    ct_data = p.get(ct.get_fdata())
    if hd['sizeof_hdr'] == 348:
        new_image = nib.Nifti1Image(ct_data, ct.affine, header=hd)
    elif hd['sizeof_hdr'] == 540:
        new_image = nib.Nifti2Image(ct_data, ct.affine, header=hd)
    nib.save(new_image, os.path.join(output_dir,nii.split('/')[-2]+'.nii.gz'))

    
