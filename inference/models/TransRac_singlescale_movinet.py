"""TransRAC network"""
from mmcv import Config
from mmcv.runner import load_checkpoint
import torch
import torch.nn as nn
import math
from torch.cuda.amp import autocast
import numpy as np
import torch.nn.functional as F
from icecream import ic
try:
    from .movinets.movinet import MoViNet
    from .movinets.config import _C
except:
    from movinets.movinet import MoViNet
    from movinets.config import _C
from einops import rearrange


class attention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, scale=64, att_dropout=None):
        super().__init__()
        # self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(att_dropout)
        self.scale = scale

    def forward(self, q, k, v, attn_mask=None):
        # q: [B, head, F, model_dim]
        scores = torch.matmul(q, k.transpose(-2, -1)) / \
            math.sqrt(self.scale)  # [B,Head, F, F]
        if attn_mask:
            scores = scores.masked_fill_(attn_mask, -np.inf)
        scores = self.softmax(scores)
        scores = self.dropout(scores)  # [B,head, F, F]
        # context = torch.matmul(scores, v)  # output
        return scores  # [B,head,F, F]


class Similarity_matrix(nn.Module):
    ''' buliding similarity matrix by self-attention mechanism '''

    def __init__(self, num_heads=4, model_dim=512):
        super().__init__()

        # self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.input_size = 512
        self.linear_q = nn.Linear(self.input_size, model_dim)
        self.linear_k = nn.Linear(self.input_size, model_dim)
        self.linear_v = nn.Linear(self.input_size, model_dim)

        self.attention = attention(att_dropout=0)
        # self.out = nn.Linear(model_dim, model_dim)
        # self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        # dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        # linear projection
        query = self.linear_q(query)  # [B,F,model_dim]
        key = self.linear_k(key)
        value = self.linear_v(value)
        # split by heads
        # [B,F,model_dim] ->  [B,F,num_heads,per_head]->[B,num_heads,F,per_head]
        query = query.reshape(batch_size, -1, num_heads,
                              self.model_dim // self.num_heads).transpose(1, 2)
        key = key.reshape(batch_size, -1, num_heads,
                          self.model_dim // self.num_heads).transpose(1, 2)
        value = value.reshape(batch_size, -1, num_heads,
                              self.model_dim // self.num_heads).transpose(1, 2)
        # similar_matrix :[B,H,F,F ]
        matrix = self.attention(query, key, value, attn_mask)

        return matrix


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x


class TransEncoder(nn.Module):
    '''standard transformer encoder'''

    def __init__(self, d_model, n_head, dim_ff, dropout=0.0, num_layers=1, num_frames=64):
        super(TransEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, 0.1, num_frames)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=n_head,
                                                   dim_feedforward=dim_ff,
                                                   dropout=dropout,
                                                   activation='relu')
        encoder_norm = nn.LayerNorm(d_model)
        self.trans_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers, encoder_norm)

    def forward(self, src):
        src = self.pos_encoder(src)
        e_op = self.trans_encoder(src)
        return e_op


class Prediction(nn.Module):
    ''' predict the density map with densenet '''

    def __init__(self, input_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Prediction, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, n_hidden_1),
            nn.LayerNorm(n_hidden_1),
            nn.Dropout(p=0.25, inplace=False),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.Dropout(p=0.25, inplace=False),
            nn.Linear(n_hidden_2, out_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class TransferModelMovinet_SS(nn.Module):
    def __init__(self, config, checkpoint, num_frames, scales, OPEN=False, num_classes=None):
        super(TransferModelMovinet_SS, self).__init__()
        self.num_frames = num_frames
        self.config = config
        self.checkpoint = checkpoint
        self.scales = scales
        self.OPEN = OPEN

        self.backbone = MoViNet(
            _C.MODEL.MoViNetPose, causal=False, pretrained=False)

        self.Replication_padding1 = nn.ConstantPad3d((0, 0, 0, 0, 1, 1), 0)
        self.Replication_padding2 = nn.ConstantPad3d((0, 0, 0, 0, 2, 2), 0)
        self.Replication_padding4 = nn.ConstantPad3d((0, 0, 0, 0, 4, 4), 0)
        self.Replication_padding8 = nn.ConstantPad3d((0, 0, 0, 0, 8, 8), 0)
        self.Replication_padding16 = nn.ConstantPad3d((0, 0, 0, 0, 16, 16), 0)

        self.conv3D = nn.Conv3d(in_channels=480,
                                out_channels=512,
                                kernel_size=3,
                                padding=(3, 1, 1),
                                dilation=(3, 1, 1))

        self.bn1 = nn.BatchNorm3d(512)
        self.SpatialPooling = nn.AdaptiveMaxPool3d(
            output_size=(num_frames, 1, 1))
        # self.SpatialPooling = nn.MaxPool3d(kernel_size=(1, 7, 7))
        self.sims = Similarity_matrix()
        self.conv3x3 = nn.Conv2d(in_channels=4 * len(self.scales),  # num_head*scale_num
                                 out_channels=32,
                                 kernel_size=3,
                                 padding=1)

        self.bn2 = nn.BatchNorm2d(32)

        self.dropout1 = nn.Dropout(0.25)
        self.input_projection = nn.Linear(self.num_frames * 32, 512)  # 线性投射层
        self.ln1 = nn.LayerNorm(512)

        self.transEncoder = TransEncoder(d_model=512, n_head=4, dropout=0.2, dim_ff=512, num_layers=1,
                                         num_frames=self.num_frames)
        self.FC = Prediction(512, 512, 256, 1)  #
        if num_classes:
            self.classification = nn.Linear(512, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        if self.OPEN:
            feat = self.backbone(x)
        else:
            with torch.no_grad():
                feat = self.backbone(x)
        x_scale = F.relu(self.bn1(self.conv3D(feat)))
        x_scale = self.SpatialPooling(x_scale).squeeze(
            3).squeeze(3).transpose(1, 2)
        x = F.relu(self.sims(x_scale, x_scale, x_scale))
        x_features = x_scale

        x_matrix = x
        x = F.relu(self.bn2(self.conv3x3(x)))  # [b,32,f,f]

        x = self.dropout1(x)

        x = x.permute(0, 2, 3, 1)  # [b,f,f,32]
        # --------- transformer encoder ------
        x = x.flatten(start_dim=2)  # ->[b,f,32*f]
        x = F.relu(self.input_projection(x))  # ->[b,f, 512]
        x = self.ln1(x)

        x = x.transpose(0, 1)  # [f,b,512]
        x = self.transEncoder(x)  #
        x = x.transpose(0, 1)  # ->[b,f, 512]
        x = self.FC(x)  # ->[b,f,1]

        if self.num_classes:
            # features_classification = (
            #     torch.sum(x*x_features, axis=1) / torch.sum(x, axis=1))
            features_classification = torch.mean(x_features, axis=1)
            class_prediction = self.classification(features_classification)
            x = x.squeeze(2)
            return x, x_matrix, class_prediction
        x = x.squeeze(2)
        return x, x_matrix, torch.ones((x.shape[0], 10)).cuda()


if __name__ == "__main__":
    checkpoint = './pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth'
    config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'

    # TransRAC trained model checkpoint, we will upload soon.
    lastckpt = None  # "./checkpoint/ours/25_0.6321.pt"

    NUM_FRAME = 64
    # multi scales(list). we currently support 1,4,8 scale.
    SCALES = [64]

    my_model = TransferModelMovinet_SS(config=config, checkpoint=checkpoint,
                                       num_frames=NUM_FRAME, scales=SCALES, OPEN=True, num_classes=None).cuda()
    import time
    st = time.time()
    my_model(torch.ones((32, 17, NUM_FRAME, 56, 56)).cuda())
    print(time.time() - st)
