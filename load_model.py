import torch
from networks.voxnet import VoxNet
from networks.lightnet import light
from networks.dgcnn import DGCNN
from networks.pct import PointTransformerCls

class opt:
    def __init__(self, dropout=0.5, emb_dims = 1024, k = 20):
        self.dropout = dropout
        self.emb_dims = emb_dims
        self.k = k

dgcnn_parameters = opt()

voxnet = VoxNet(19)
voxnet.load_state_dict(torch.load('pretrain/voxnet.pth'))

lightnet = light(19)
lightnet.load_state_dict(torch.load('pretrain/light.pth'))

dgcnn = DGCNN(dgcnn_parameters,output_channels=19)
dgcnn.load_state_dict(torch.load('pretrain/dgcnn.pth'))

pct = PointTransformerCls(opt(),output_channels=19)
pct.load_state_dict(torch.load('pretrain/pct.pth'))

'''
# Pretrained feature encoder, until penultimate layer. I think that is what you need.
model.features()

#entire classifier
model.predict()
'''