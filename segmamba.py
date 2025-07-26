# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch.nn as nn
import torch 
from functools import partial
import random

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F 
from einops import rearrange

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x
def frequency_domain_processing(x_fft, low_freq_radius=10, high_freq_radius=30):
    B, C, H, W = x_fft.shape[:,:,:,:]
    cy, cx = H // 2, W // 2  # 中心点

    # 1. 创建一个高通滤波器
    high_pass_mask = torch.ones((B, C, H, W), device=x_fft.device)
    high_pass_mask[:, :, cy - high_freq_radius:cy + high_freq_radius, cx - high_freq_radius:cx + high_freq_radius] = 0

    # 2. 将高通滤波器应用于频域
    x_fft_high = x_fft * high_pass_mask

    return x_fft_high

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices=num_slices,
        )
    
    def forward(self, x):
        #print(x.shape)
        #[2, 48, 64, 64, 64]-[2, 48, 64, 64, 64]-[2, 96, 32, 32, 32]-[2, 96, 32, 32, 32]-[2, 192, 16, 16, 16]-[2, 384, 8, 8, 8]
        B, C = x.shape[:2]
        #print("B: ",B, "C: ", C)
        #b:2, c:48-48-96-96-192-384-384
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        # x = x.to(torch.float32)
        # x_fft = torch.fft.fft2(x)  # 2D FFT for complex tensors

        # # Optional: Perform processing in the frequency domain
        # # For example, you can modify `x_fft` here if needed
        # #x_fft= frequency_domain_processing(x_fft)
        # # Step 2: Inverse Fourier Transform to bring it back to the time domain
        # x_ifft = torch.fft.ifft2(x_fft).to(torch.float16)
        # x_flat = x_ifft.reshape(B, C, n_tokens).transpose(-1, -2)
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip
        #print("out size ", out.shape)
        #(2,192,16,16,16)
        
        return out
    
class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):

        x_residual = x 

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        
        return x + x_residual

class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
              nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
              )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        self.hims = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])
            him = HIMBlock(style_channels=64, feat_channels=dims[i])


            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            self.hims.append(him)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x, x_style):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            #print("x shape ", x.shape)
            # x shape  torch.Size([2, 48, 64, 64, 64])
            # x shape  torch.Size([2, 96, 32, 32, 32])
            # x shape  torch.Size([2, 192, 16, 16, 16])
            # x shape  torch.Size([2, 384, 8, 8, 8])

            # style[0]: torch.Size([2, 64, 32, 32, 32])
            # style[1]: torch.Size([2, 32, 64, 64, 64])
            #print("x_style ", x_style[0].size())
            #x= self.him(x, x_style[0])

            x = self.hims[i](x, x_style[0])
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x, x_style):
        x = self.forward_features(x, x_style)
        return x

class HIMBlock(nn.Module):
    def __init__(self, style_channels, feat_channels):
        super(HIMBlock, self).__init__()
        self.proj = nn.Conv3d(style_channels, feat_channels, kernel_size=1)
        self.gate_conv = nn.Sequential(
            nn.Conv3d(feat_channels * 2, feat_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(feat_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.out_conv = nn.Conv3d(feat_channels, feat_channels, kernel_size=1)

    def forward(self, feat, style):
        if style.shape[2:] != feat.shape[2:]:
            style = F.interpolate(style, size=feat.shape[2:], mode='trilinear', align_corners=False)
        style = self.proj(style)
        gate = self.gate_conv(torch.cat([feat, style], dim=1))
        fused = gate * feat + (1 - gate) * style
        feat_out = self.out_conv(fused)
        return feat+ feat_out
    

# class HIMBlock(nn.Module):
#     def __init__(self, style_channels, feat_channels):
#         super(HIMBlock, self).__init__()

#         # 将style的通道数映射到feat_channels，减少卷积计算量
#         self.proj = nn.Conv3d(style_channels, feat_channels, kernel_size=1)

#         # 更高效的门控机制：使用深度可分离卷积替代普通卷积
#         self.gate_conv = nn.Sequential(
#             nn.Conv3d(feat_channels * 2, feat_channels, kernel_size=1),  # 减少卷积的计算量
#             nn.ReLU(),
#             nn.Conv3d(feat_channels, 1, kernel_size=1),  # 输出一个单通道的门控系数
#             nn.Sigmoid()  # 输出范围[0, 1]的门控系数
#         )

#         # 输出卷积，保持feat_channels通道数不变
#         self.out_conv = nn.Conv3d(feat_channels, feat_channels, kernel_size=1)

#     def forward(self, feat, style):
#         # 如果style和feat的尺寸不同，则调整style的尺寸以匹配feat
#         if style.shape[2:] != feat.shape[2:]:
#             style = F.interpolate(style, size=feat.shape[2:], mode='trilinear', align_corners=False)

#         # 使用1x1x1卷积将style的通道数映射到feat的通道数
#         style = self.proj(style)

#         # 拼接feat和style，生成门控系数
#         gate = self.gate_conv(torch.cat([feat, style], dim=1))

#         # 根据门控系数加权融合feat和style
#         fused = gate * feat + (1 - gate) * style

#         # 通过输出卷积获取最终的融合结果
#         fused_output = self.out_conv(fused)

#         # 这里加入原始feat作为残差连接（Residual Connection）
#         output = fused_output + feat  # 加上原始输入feat

#         return output
    
class SegMamba(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.spatial_dims = spatial_dims
        #self.him = HIM(feat_dim=64, prior_dim=128, hidden_dim=128)

        self.vit = MambaEncoder(in_chans, 
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                              )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in, x_style, cmask_modal):
        #如果让x_style的尺寸是(2,1,128,128,128),并且要传入是哪个模态，这样就可以用x_style替换为该模态的数据来送入到mamba了
        #(2,4,128,128,128)
        # style[0]: torch.Size([2, 64, 32, 32, 32])
        # style[1]: torch.Size([2, 32, 64, 64, 64])
        # style[2]: torch.Size([2, 16, 128, 128, 128])

        # x_style[0] shape: torch.Size([2, 64, 32, 32, 32])
        # x_style[1] shape: torch.Size([2, 128, 16, 16, 16])
        # x_style[2] shape: torch.Size([2, 128, 16, 16, 16])
        # style_feat = x_style[1]  #选择第二个特征融入到x中，之后送入到Mamba中

        # clone x 为结果容器

        # fused_x = x_in.clone()
        # for b in range(x_in.shape[0]):
        #     style_feat = x_style[1][b].unsqueeze(0)  # [1, 128, 16, 16, 16]

        #     i = cmask_modal[b]  # 当前样本被遮掩的模态索引（单个 int）
        #     xi = x_in[b, i].unsqueeze(0).unsqueeze(0)  # [1, 1, 128, 128, 128]

        #     # ✅ Step 1: 下采样 xi，降低 attention 开销
        #     xi_down = F.avg_pool3d(xi, kernel_size=4)  # [1, 1, 32, 32, 32]

        #     # ✅ Step 2: 送入 cross attention 模块
        #     xi_fused_down = self.him(xi_down, style_feat)  # 注意：style_feat是[1, 128, 16, 16, 16]

        #     # ✅ Step 3: 上采样回原分辨率
        #     xi_fused = F.interpolate(xi_fused_down, size=(128, 128, 128), mode='trilinear', align_corners=False)

        #     # ✅ Step 4: 可选残差（融合风格）
        #     xi_fused = xi + xi_fused  # 也可以不加残差，按你需求定

        #     # ✅ Step 5: 写入结果
        #     fused_x[b, i] = xi_fused.squeeze(0).squeeze(0)  # [128, 128, 128]

        # #print("fused_x ", fused_x.shape)
        # x_in = fused_x.clone()
        #print("cmask_modal ",cmask_modal)

        outs = self.vit(x_in, x_style)
        enc1 = self.encoder1(x_in)
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        enc_hidden = self.encoder5(outs[3])
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)
                
        return self.out(out)
    
