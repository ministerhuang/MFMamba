
from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor 
import numpy as np 
import pickle 
import json 

data_filename = ["t2w.nii.gz",
                 "t2f.nii.gz",
                 "t1n.nii.gz",
                 "t1c.nii.gz"]
seg_filename = "seg.nii.gz"

base_dir = "./data/raw_data/BraTS2023/"
image_dir = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

def process_train():
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=seg_filename
                                   )

    out_spacing = [1.0, 1.0, 1.0]
    output_dir = "./data/fullres/train/"
    
    preprocessor.run(output_spacing=out_spacing, 
                     output_dir=output_dir, 
                     all_labels=[1, 2, 3],
    )

def plan():
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=seg_filename
                                   )
    
    preprocessor.run_plan()


if __name__ == "__main__":

    plan()
    process_train()

#将多模态数据，合并成一个多通道的numpy数组，每个通道对应一个模态
#返回的数据包括：data（多模态图像）、seg_arr（分割标签，如果存在）以及一些属性（如 spacing、原始大小等）。
#所以生成的.npy文件中，已经包含