import numpy as np
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.losses.dice import DiceLoss
set_determinism(123)
import os
from torch.nn import functional as F
from torch.autograd import Variable
from model_segmamba.pretrain.model.mask_utils import MaskEmbeeding1
import random

data_dir = "./data/fullres/train"
logdir = f"./logs/segmamba"

model_save_path = os.path.join(logdir, "model")
# augmentation = "nomirror"
augmentation = True

env = "pytorch"
max_epoch = 1000
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:0"
roi_size = [128, 128, 128]

def func(m, epochs):
    return np.exp(-10*(1- m / epochs)**2)

class EDiceLoss(nn.Module):
    """Dice loss tailored to Brats need."""

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["ET", "TC", "WT"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1., device=self.device)
                else:
                    return torch.tensor(0., device=self.device)

        intersection = EDiceLoss.compute_intersection(inputs, targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)

        return 1 - dice if not metric_mode else dice

    @staticmethod
    def compute_intersection(inputs, targets):
        return torch.sum(inputs * targets)

    def forward(self, inputs, target):
        dice = 0
        ce = 0
        CE_L = torch.nn.BCEWithLogitsLoss()  # 安全替代

        for i in range(target.size(1)):
            dice += self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)
            ce += CE_L(inputs[:, i, ...], target[:, i, ...])  # 注意：不要加 sigmoid！

        final_loss = (0.7 * dice + 0.3 * ce) / target.size(1)
        return final_loss

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices

def proj(image, patch_size = 16):

    B, C, D, H, W = image.shape
    image_ = image.reshape(B, C, D // patch_size, patch_size, H // patch_size, patch_size, W // patch_size, patch_size)
    
    image_ = image_.permute(0, 1, 2, 4, 6, 3, 5, 7).reshape(B, C * D // patch_size * H // patch_size * W // patch_size,patch_size, patch_size, patch_size)
    
    return image_

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=roi_size,
                                        sw_batch_size=1,
                                        overlap=0.5)
        self.augmentation = augmentation
        from model_segmamba.segmamba import SegMamba
        from model_segmamba.pretrain.model.Unet import Unet_missing

        self.model_mae = Unet_missing(
            #input_shape = [128,128,128], pre_train = False, mask_ratio = 0.4, mdp = 4
            input_shape = [128,128,128], pre_train = False, mask_ratio = 0.875, mdp = 1, batch_size= self.batch_size
        )
        self.raw_input = proj(torch.ones((1, 4, 128, 128, 128)))
        self.limage = nn.Parameter(torch.randn((1, 4, 128, 128, 128)), requires_grad=False)
        self.model_mae = self.model_mae.to(self.device)

        self.model = SegMamba(in_chans=4,
                        out_chans=4,
                        depths=[2,2,2,2],
                        feat_size=[48, 96, 192, 384])

        self.patch_size = roi_size
        self.best_mean_dice = 0.0
        self.ce = nn.CrossEntropyLoss() 
        self.mse = nn.MSELoss()
        self.train_process = 18
        # self.optimizer = torch.optim.SGD(list(self.model.parameters()) + list(self.model_mae.parameters()), lr=1e-2, weight_decay=3e-5,
        #                           momentum=0.99, nesterov=True) #      1e-4  lr=3e-4, weight_decay=0.00001 mamba:lr=1e-2, weight_decay=3e-5
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=3e-5,   
        #                            momentum=0.99, nesterov=True)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=3e-5,
                                    momentum=0.99, nesterov=True)
        self.optimizer_mae = torch.optim.SGD(self.model_mae.parameters(), lr=3e-4, weight_decay=0.00001,
                                    momentum=0.99, nesterov=True)
        
        self.scheduler_type = "poly"
        self.cross = nn.CrossEntropyLoss()
        self.criterion = EDiceLoss(do_sigmoid=False)

    def training_step(self, batch, batch_size):
        image, label = self.get_input(batch)
        #print(batch['data'][0])

        patch_locations = batch["crop_indexes"]

        # inputs_S1 = image.clone()

        # for ci in range(batch_size):
        #     mask_np = MaskEmbeeding1(1, raw_input=self.raw_input, mdp=1, mask=False, patch_shape=128)
            
        #     mask = torch.from_numpy(mask_np).to(inputs_S1.device).float()
        #     pl = patch_locations[ci]
        #     zmin, ymin, xmin = pl[0]
        #     zmax, ymax, xmax = pl[1]


        #     limage_crop = self.limage[:, :, zmin:zmax, ymin:ymax, xmin:xmax].squeeze(0).cpu().numpy()
        #     limage_crop, crop_indexes = pad_or_crop_image(limage_crop, target_size=(128,128,128))
        #     limage_crop = torch.from_numpy(limage_crop).to(inputs_S1.device).float()
        #     inputs_S1[ci] = inputs_S1[ci] * mask + limage_crop * (1 - mask)
                    
        # print("3", inputs_S1.size())
        # inputs_S1= inputs_S1.cuda()

        with torch.amp.autocast(device_type="cuda", enabled=False):
            pred_mae, _, style, content, cmask_modal = self.model_mae(image, location = patch_locations, fmdp= [1, 1], batch_size = 2)#, fmdp= [1])

        #print(len(style))
        #print(batch["crop_indexes"][0])
        #print(cmask_modal)

        loss_recon = self.cross(pred_mae, label)

        # if torch.isnan(loss_recon) or torch.isinf(loss_recon) or loss_recon > 1e3:
        #     print("Warning: skipping batch due to unstable loss")
        #     return torch.tensor(0.0, requires_grad=True).to(image.device)
        
        pred = self.model(image, style, cmask_modal)

        loss = self.cross(pred, label)

        self.log("training_loss", loss, step=self.global_step)

        #print("loss_recon is: ", loss_recon, "loss is: ", loss)
        return loss + loss_recon*0.5
    
    def convert_labels(self, labels):
        ## TC, WT and ET
        result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]
        
        return torch.cat(result, dim=1).float()

    
    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
    
        label = label[:, 0].long()
        return image, label

    def cal_metric(self, gt, pred, voxel_spacing=[1.0, 1.0, 1.0]):
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            return np.array([d, 50])
        
        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])
        
        else:
            return np.array([0.0, 50])
    
    def validation_step(self, batch):
        image, label = self.get_input(batch)
        patch_locations = batch["crop_indexes"]

        with torch.amp.autocast(device_type="cuda", enabled=False):
            pred_mae, _, style, content, cmask_modal = self.model_mae(image, location = patch_locations, fmdp= [1, 1], batch_size = 1)
       
        output = self.model(image, style, cmask_modal)

        output = output.argmax(dim=1)

        output = output[:, None]
        output = self.convert_labels(output)

        label = label[:, None]
        label = self.convert_labels(label)

        output = output.cpu().numpy()
        target = label.cpu().numpy()
        
        dices = []

        c = 3
        for i in range(0, c):
            pred_c = output[:, i]
            target_c = target[:, i]

            cal_dice, _ = self.cal_metric(target_c, pred_c)
            dices.append(cal_dice)
        
        return dices
    
    def validation_end(self, val_outputs):
        dices = val_outputs

        tc, wt, et = dices[0].mean(), dices[1].mean(), dices[2].mean()

        print(f"dices is {tc, wt, et}")

        mean_dice = (tc + wt + et) / 3 
        
        self.log("tc", tc, step=self.epoch)
        self.log("wt", wt, step=self.epoch)
        self.log("et", et, step=self.epoch)

        self.log("mean_dice", mean_dice, step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{mean_dice:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{mean_dice:.4f}.pt"), 
                                        delete_symbol="final_model")


        if (self.epoch + 1) % 100 == 0:
            torch.save(self.model.state_dict(), os.path.join(model_save_path, f"tmp_model_ep{self.epoch}_{mean_dice:.4f}.pt"))

        print(f"mean_dice is {mean_dice}")

if __name__ == "__main__":

    trainer = BraTSTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17759,
                            training_script=__file__)

    train_ds, val_ds, test_ds = get_train_val_test_loader_from_train(data_dir)
    print("train_len: ",len(train_ds))

    #for idx, sample in enumerate(train_ds):
        #Sample 789 keys: dict_keys(['data', 'seg', 'properties'])   sample的格式，分别是数据，分割，元数据
        #Sample 789 size: (4, 142, 181, 147)    data的尺寸，感觉是4个模态，每个模态的尺寸

        #print(f"Sample {idx} size: {sample['data'].shape}", "and seg shape is: ", sample['seg'].shape)
        #seg shape is:  (1, 142, 182, 130)分割的尺寸

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
