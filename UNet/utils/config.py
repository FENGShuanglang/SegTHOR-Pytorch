# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:03:52 2019

@author: Administrator
"""
class DefaultConfig(object):
    num_epochs=40
    train_val_epochs=5#无用
    epoch_start_i=0
    checkpoint_step=5
    validation_step=1
    crop_height=256#无用
    crop_width=256#无用
    batch_size=6   
    
    
    data='/home/FENGsl/JBHI/dataset'#数据存放路径
    dataset="SegTHOR"#数据集名字
    log_dirs='/home/FENGsl/JBHI/Log/SegTHOR'#存放log的文件夹名字

    lr=0.01    
    lr_mode= 'poly'
    net_work= 'UNet'
    momentum = 0.9#�Ż�������ѡ��
    weight_decay = 1e-4#�Ż���˥��ϵ��ѡ��


    mode='train_test'
  
    k_fold=4
    test_fold=4
    num_workers=4
    num_classes=5
    cuda='3'
    use_gpu=True
    pretrained_model_path=None
    save_model_path='./checkpoints'
    


