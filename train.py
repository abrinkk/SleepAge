import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import copy
import argparse

from config import Config
from psg_dataset import PSG_Dataset, PSG_epoch_Dataset, PSG_pretrain_Dataset, PSG_feature_Dataset
from m_psg2label import M_PSG2FEAT, M_FEAT2LABEL
from utils import save_model_F, save_model_L, debug_model, plot_regression_pred, get_age_label_distribution, save_train_features, plot_train_epoch, plot_huber_loss

# Parser
parser = argparse.ArgumentParser(description='Train model based on setup in config.')
parser.add_argument('--train', type=bool, default=True,
                    help='To train model.')
parser.add_argument('--load_best_train', type=bool, default=False,
                    help='Load the trained model that was best on validation set.')
parser.add_argument('--train_continue', type=bool, default=False,
                    help='To continue preivously trained model')
parser.add_argument('--train_freeze_feat', type=bool, default=False,
                    help='To freeze feature model')
parser.add_argument('--test', type=bool, default=True,
                    help='To continue preivously trained model')
parser.add_argument('--pre_train', type=bool, default=False,
                    help='To pretrain model.')
parser.add_argument('--load_best_pre_train', type=bool, default=True,
                    help='Load the pretrained model that was best on validation set.')
parser.add_argument('--pre_train_continue', type=bool, default=False,
                    help='To continue preivously trained model')
parser.add_argument('--test_pre', type=bool, default=False,
                    help='To continue preivously trained model')
parser.add_argument('--save_feat', type=bool, default=False,
                    help='Overwrite model F features')
parser.add_argument('--post_train', type=bool, default=False,
                    help='To posttrain model.')
parser.add_argument('--load_best_post_train', type=bool, default=True,
                    help='Load the posttrained model that was best on validation set.')
parser.add_argument('--post_train_continue', type=bool, default=False,
                    help='To continue preivously trained model')
parser.add_argument('--test_post', type=bool, default=False,
                    help='To continue preivously trained model')

args = parser.parse_args()

def HuberLoss(y, t, d, s, device):
    #d = torch.Tensor(d).type(torch.DoubleTensor).to(device)
    #s = torch.Tensor(s).type(torch.DoubleTensor).to(device)
    dif = (t.type(torch.DoubleTensor) - y.type(torch.DoubleTensor)).abs_()
    mask_l2 = (dif < d)
    L = (d * dif - 1/2 * d**2) / s
    L[mask_l2] = (1/2 * dif[mask_l2]**2) / s
    return L.mean()
    

def pretrain(args, config, model_F, device, writer):
    
    # Parameters
    test_params = {'batch_size': 1,
                   'shuffle': True,
                   'num_workers': config.pre_n_workers}
    pretrain_params = {'batch_size': config.pre_batch_size,
                       'shuffle': True,
                       'num_workers': config.pre_n_workers}
    test_epoch_params = {'batch_size': config.pre_batch_size,
                       'shuffle': False,
                       'num_workers': config.pre_n_workers}
    
    # Generators
    train_set = PSG_pretrain_Dataset(config, 'train')
    train_gen = DataLoader(train_set, **pretrain_params)
    val_set = PSG_Dataset(config, 'val')
    val_gen = DataLoader(val_set, **test_params)
    
    # Model
    example_epoch, example_label = iter(train_gen).next()
    writer.add_graph(model_F, example_epoch.to(device))
    
    # Optimization
    optimizer_F = torch.optim.Adam(model_F.parameters(), lr=config.pre_lr, weight_decay = 1e-5)
    loss_fn_F = lambda y, t: HuberLoss(y, t, 5, 20, device)
    L1Loss = nn.L1Loss()
    
    # Continue previous pretraining
    if not args.pre_train:
        if args.load_best_pre_train:
            checkpoint = torch.load(config.model_F_path + '.tar')
            model_F = checkpoint['best_model']
        return model_F
    elif args.pre_train_continue:
        checkpoint = torch.load(config.model_F_path + '.tar')
        start_epoch = checkpoint['epoch']
        model_F.load_state_dict(checkpoint['model_state_dict'])
        optimizer_F.load_state_dict(checkpoint['optimizer_state_dict'])
        train_iter = start_epoch * len(train_set) // config.pre_batch_size
        best_loss = checkpoint['best_loss']
        
    else:
        train_iter = 0
        checkpoint = 0
        start_epoch = 0
    
    # Loop over epochs
    train_loss_all = []
    train_l1_loss_all = []
    val_loss_all = []
    val_l1_loss_all = []
    for epoch in range(start_epoch, config.pre_max_epochs):
        # Training
        train_loss = []
        train_l1_loss = []
        model_F.train()
        for psg, lab in train_gen:
            # Zero Gradients
            optimizer_F.zero_grad()
            # Make predictions
            psg, lab = psg.to(device), lab.to(device)
            out_F, _ = model_F(psg)
            # Optimize network
            loss_F = loss_fn_F(out_F, lab)
            loss_F.backward()
            optimizer_F.step()
            # Track the loss
            train_loss.append(loss_F.item())
            train_iter += 1
            loss_l1 = L1Loss(out_F.type(torch.float), lab.type(torch.float))
            train_l1_loss.append(loss_l1.item())
            writer.add_scalar('PreTraining/L1Loss', 
                              np.array(loss_l1.item()), 
                              train_iter)
            writer.add_scalar('PreTraining/HuberLoss', 
                              np.array(loss_F.item()), 
                              train_iter)
        
        # Validation
        model_F.eval()
        val_loss = 0.0
        val_l1_loss = 0.0
        with torch.no_grad():
            for X, y in val_gen:
                X, y = torch.squeeze(X, 0), torch.squeeze(y, 0)
                val_epoch_set = PSG_epoch_Dataset(config, X, y)
                val_epoch_gen = DataLoader(val_epoch_set, **test_epoch_params)
                for psg, lab in val_epoch_gen:
                    psg, lab = psg.to(device), lab.to(device)
                    out_F, _ = model_F(psg)
                    val_loss += loss_fn_F(out_F, lab).item() * lab.size(0) / float(len(val_epoch_set))
                    val_l1_loss += L1Loss(out_F.type(torch.float), lab.type(torch.float)).item() * lab.size(0) / float(len(val_epoch_set))
                    
                    
        val_loss = val_loss / float(len(val_set))
        val_l1_loss = val_l1_loss / float(len(val_set))
        train_loss_all.extend(train_loss)
        train_l1_loss_all.extend(train_l1_loss)
        val_loss_all.append(val_loss)
        val_l1_loss_all.append(val_l1_loss)
        
        # Track loss
        writer.add_scalar('PreValidation/L1Loss', 
                          val_l1_loss, 
                          (epoch + 1) * len(train_set) / config.pre_batch_size)
        writer.add_scalar('PreValidation/HuberLoss', 
                          val_loss, 
                          (epoch + 1) * len(train_set) / config.pre_batch_size)
    
        print('Train Epoch: {} of {}, Training loss: {:.6f}, Val loss: {:.6f}'.format(
                    epoch+1, config.pre_max_epochs, np.mean(train_loss), val_loss))
    
        # Save Model
        if epoch == 0:
            best_loss = val_loss
            best_epoch = epoch
            best_net = copy.deepcopy(model_F)
        elif val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_net = copy.deepcopy(model_F)
        save_model_F(config.model_F_path, epoch, model_F, optimizer_F, val_loss, best_net, best_epoch, best_loss)
    
    return best_net

def posttrain(args, config, model_L, device, writer):
    
    # Parameters
    posttrain_params = {'batch_size': config.post_batch_size,
                       'shuffle': True,
                       'num_workers': config.post_n_workers}
    
    # Generators
    train_set = PSG_feature_Dataset(config, 'train_F')
    train_gen = DataLoader(train_set, **posttrain_params)
    val_set = PSG_feature_Dataset(config, 'val_F')
    val_gen = DataLoader(val_set, **posttrain_params)
    
    # Optimization
    optimizer_L = torch.optim.Adam(model_L.parameters(), lr=config.post_lr, weight_decay = 1e-5)
    loss_fn_L = lambda y, t: HuberLoss(y, t, 5, 20, device)
    L1Loss = nn.L1Loss()
    
    # Continue previous pretraining
    if not args.post_train:
        if args.load_best_post_train:
            checkpoint = torch.load(config.model_L_path + '.tar')
            model_L = checkpoint['best_model']
        return model_L
    elif args.post_train_continue:
        checkpoint = torch.load(config.model_L_path + '.tar')
        start_epoch = checkpoint['epoch']
        model_L.load_state_dict(checkpoint['model_state_dict'])
        optimizer_L.load_state_dict(checkpoint['optimizer_state_dict'])
        train_iter = start_epoch * len(train_set) // config.post_batch_size
        best_loss = checkpoint['best_loss']
        
    else:
        train_iter = 0
        checkpoint = 0
        start_epoch = 0
    
    # Loop over epochs
    train_loss_all = []
    train_l1_loss_all = []
    val_loss_all = []
    val_l1_loss_all = []
    
    for epoch in range(start_epoch, config.post_max_epochs):
        # Training
        train_loss = []
        train_l1_loss = []
        model_L.train()
        for feat, lab in train_gen:
            # Zero Gradients
            optimizer_L.zero_grad()
            # Make predictions
            feat, lab = feat.to(device), lab.to(device)
            out_L = model_L(feat)
            # Optimize network
            loss_L = loss_fn_L(out_L, lab)
            loss_L.backward()
            optimizer_L.step()
            # Track the loss
            train_loss.append(loss_L.item())
            train_iter += 1
            loss_l1 = L1Loss(out_L.type(torch.float), lab.type(torch.float))
            train_l1_loss.append(loss_l1.item())
            writer.add_scalar('PostTraining/L1Loss', 
                              np.array(loss_l1.item()), 
                              train_iter)
            writer.add_scalar('PostTraining/HuberLoss', 
                              np.array(loss_L.item()), 
                              train_iter)
        
        # Validation
        model_L.eval()
        val_loss = 0.0
        val_l1_loss = 0.0
        with torch.no_grad():
            for feat, lab in val_gen:
                feat, lab = feat.to(device), lab.to(device)
                out_L = model_L(feat)
                val_loss += loss_fn_L(out_L, lab).item() * lab.size(0) / float(len(val_set))
                val_l1_loss += L1Loss(out_L.type(torch.float), lab.type(torch.float)).item() * lab.size(0) / float(len(val_set))
                    
        train_loss_all.extend(train_loss)
        train_l1_loss_all.extend(train_l1_loss)
        val_loss_all.append(val_loss)
        val_l1_loss_all.append(val_l1_loss)
        
        # Track loss
        writer.add_scalar('PostValidation/L1Loss', 
                          val_l1_loss, 
                          (epoch + 1) * len(train_set) / config.post_batch_size)
        writer.add_scalar('PostValidation/HuberLoss', 
                          val_loss, 
                          (epoch + 1) * len(train_set) / config.post_batch_size)
    
        print('Train Epoch: {} of {}, Training loss: {:.6f}, Val loss: {:.6f}'.format(
                    epoch+1, config.post_max_epochs, np.mean(train_loss), val_loss))
    
        # Save Model
        if epoch == 0:
            best_loss = val_loss
            best_epoch = epoch
            best_net = copy.deepcopy(model_L)
        elif val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_net = copy.deepcopy(model_L)
        save_model_F(config.model_L_path, epoch, model_L, optimizer_L, val_loss, best_net, best_epoch, best_loss)
    
    return model_L

def train(args, config, model_F, model_L, device, writer):
    # Training configurations
    
    # Parameters
    test_params = {'batch_size': 1,
                   'shuffle': False,
                   'num_workers': config.pre_n_workers}
    pretrain_params = {'batch_size': config.pre_batch_size,
                       'shuffle': False,
                       'num_workers': config.pre_n_workers}
    
    # Generators
    train_set = PSG_Dataset(config, 'train')
    train_gen = DataLoader(train_set, **test_params)
    val_set = PSG_Dataset(config, 'val')
    val_gen = DataLoader(val_set, **test_params)
    
    # Optimization
    optimizer_F = torch.optim.Adam(model_F.parameters(), lr=config.lr, weight_decay = config.l2)
    optimizer_L = torch.optim.Adam(model_L.parameters(), lr=config.lr, weight_decay = config.l2)
    loss_fn_F = lambda y, t: HuberLoss(y, t, 5, 20, device)
    L1Loss = nn.L1Loss()
    L2Loss = nn.MSELoss()
    # Continue previous pretraining
    if not args.train:
        if args.load_best_train:
            checkpoint = torch.load(config.model_J_path + '.tar')
            model_L = checkpoint['best_model_L']
            model_F = checkpoint['best_model_F']
        return model_L, model_F
    elif args.train_continue:
        checkpoint = torch.load(config.model_J_path + '.tar')
        start_epoch = checkpoint['epoch']
        model_L.load_state_dict(checkpoint['model_L_state_dict'])
        optimizer_L.load_state_dict(checkpoint['optimizer_L_state_dict'])
        model_F.load_state_dict(checkpoint['model_F_state_dict'])
        optimizer_F.load_state_dict(checkpoint['optimizer_F_state_dict'])        
        train_iter = start_epoch * len(train_set) // config.batch_size
        best_loss = checkpoint['loss']
    else:
        train_iter = 0
        checkpoint = 0
        start_epoch = 0
    
    # Loop over epochs
    train_loss_all = []
    train_l1_loss_all = []
    val_loss_all = []
    val_l1_loss_all = []
    for epoch in range(start_epoch, config.max_epochs):
        # Training
        train_loss = []
        train_l1_loss = []
        model_F.train()
        model_L.train()
        for X, y in train_gen:
            X = torch.squeeze(X, 0)
            train_epoch_set = PSG_epoch_Dataset(config, X, torch.squeeze(y,0))
            train_epoch_gen = DataLoader(train_epoch_set, **pretrain_params)
            optimizer_F.zero_grad()
            # Collect all features
            # Set feature model to not compute gradients
            for p in model_F.parameters():
                p.requires_grad = False
            output_F = []
            for psg, lab in train_epoch_gen:
                psg, lab = psg.to(device), lab.to(device)
                _, output_F_b = model_F(psg)
                output_F.append(output_F_b)
            output_F_psg = torch.cat(output_F, 0).detach()
            output_F_psg = torch.unsqueeze(output_F_psg, 0)
            output_F_psg.requires_grad = True
            # Update label model
            output_L = model_L(output_F_psg)
            loss_L = loss_fn_F(output_L, y.to(device))
            loss_L_l1 = L1Loss(output_L.type(torch.float), y.type(torch.float).to(device))
            loss_L.backward()
            in_grad = output_F_psg.grad.detach()
            y_feat = -0.5 * in_grad + torch.unsqueeze(torch.cat(output_F, 0),0).detach()
            y_feat = torch.squeeze(y_feat, 0).detach()
            
            # Update feature model
            if not args.train_freeze_feat:
                for p in model_F.parameters():
                    p.requires_grad = True
                for batch_idx, (psg, lab) in enumerate(train_epoch_gen):
                    psg, _ = psg.to(device), lab.to(device)
                    _, output_F_b = model_F(psg)
                    y_feat_b = y_feat[(batch_idx * config.pre_batch_size):(batch_idx * config.pre_batch_size + output_F_b.size(0))]
                    loss_F = L2Loss(output_F_b.type(torch.float), y_feat_b.type(torch.float))
                    loss_F.backward()
                
            # Optimize models
            if (train_iter + 1) % config.update_freq == 0:
                for p in model_L.parameters():
                    if p.grad is not None:
                        p.grad /= config.update_freq  
                optimizer_L.step()
                optimizer_L.zero_grad()
                if not args.train_freeze_feat:
                    for p in model_F.parameters():
                        if p.grad is not None:
                            p.grad /= config.update_freq  
                    optimizer_F.step()
                    optimizer_F.zero_grad()
                
            # record loss
            train_loss.append(loss_L.item())
            train_l1_loss.append(loss_L_l1.item())
            train_iter += 1
            writer.add_scalar('Training/L1Loss', 
                              np.array(loss_L_l1.item()), 
                              train_iter)
            writer.add_scalar('Training/HuberLoss', 
                              np.array(loss_L.item()), 
                              train_iter)
        # Validation
        model_F.eval()
        model_L.eval()
        val_loss = 0.0
        val_l1_loss = 0.0
        with torch.no_grad():
            for X, y in val_gen:
                X, y = torch.squeeze(X, 0), torch.squeeze(y, 0)
                val_epoch_set = PSG_epoch_Dataset(config, X, y)
                val_epoch_gen = DataLoader(val_epoch_set, **pretrain_params)
                # Collect faetures
                output_F = []
                for psg, lab in val_epoch_gen:
                    psg, lab = psg.to(device), lab.to(device)
                    _, output_F_b = model_F(psg)
                    output_F.append(output_F_b)
                output_F_psg = torch.cat(output_F, 0).detach()
                output_F_psg = torch.unsqueeze(output_F_psg, 0)
                # Run label model
                output_L = model_L(output_F_psg)
                val_loss += loss_fn_F(output_L, y.to(device)).item()
                val_l1_loss += L1Loss(output_L.type(torch.float), y.type(torch.float).to(device)).item()
                
        val_loss = val_loss / float(len(val_set))
        val_l1_loss = val_l1_loss / float(len(val_set))
        train_loss_all.extend(train_loss)
        train_l1_loss_all.extend(train_l1_loss)
        val_loss_all.append(val_loss)
        val_l1_loss_all.append(val_l1_loss)
        
        # Track loss
        writer.add_scalar('Validation/L1Loss', 
                          val_l1_loss, 
                          (epoch + 1) * len(train_set))
        writer.add_scalar('Validation/HuberLoss', 
                          val_loss, 
                          (epoch + 1) * len(train_set))
    
        print('Train Epoch: {} of {}, Training loss: {:.6f}, Val loss: {:.6f}'.format(
                    epoch+1, config.pre_max_epochs, np.mean(train_loss), val_loss))
    
        # Save Model
        if epoch == 0:
            best_loss = val_loss
            best_epoch = epoch
            best_net_F = copy.deepcopy(model_F)
            best_net_L = copy.deepcopy(model_L)
        elif val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_net_F = copy.deepcopy(model_F)
            best_net_L = copy.deepcopy(model_L)
        save_model_L(config.model_J_path, epoch, model_F, optimizer_F, model_L, optimizer_L, val_loss, best_net_F, best_net_L, best_epoch, best_loss)
                
    return best_net_L, best_net_F

def test(args, config, model_F, model_L, device, writer):
    
    test_params = {'batch_size': 1,
                   'shuffle': False,
                   'num_workers': config.pre_n_workers}
    pretrain_params = {'batch_size': config.pre_batch_size,
                       'shuffle': False,
                       'num_workers': config.pre_n_workers}
    
    loss_fn_F = lambda y, t: HuberLoss(y, t, 5, 20, device)
    L1Loss = nn.L1Loss()
    
    train_set = PSG_pretrain_Dataset(config, 'train')
    test_set = PSG_Dataset(config, 'test')
    test_gen = DataLoader(test_set, **test_params)
    
    # Testing
    test_loss = 0.0
    test_l1_loss = 0.0
    target_full = []
    pred_full = []
    model_F.eval()
    model_L.eval()
    with torch.no_grad():
        for X, y in test_gen:
            X, y = torch.squeeze(X, 0), torch.squeeze(y, 0)
            test_epoch_set = PSG_epoch_Dataset(config, X, y)
            test_epoch_gen = DataLoader(test_epoch_set, **pretrain_params)
            # Collect faetures
            output_F = []
            for psg, lab in test_epoch_gen:
                psg, lab = psg.to(device), lab.to(device)
                _, output_F_b = model_F(psg)
                output_F.append(output_F_b)
            output_F_psg = torch.cat(output_F, 0).detach()
            output_F_psg = torch.unsqueeze(output_F_psg, 0)
            # Run label model
            output_L = model_L(output_F_psg)
            test_loss += loss_fn_F(output_L, y.to(device)).item()
            test_l1_loss += L1Loss(output_L.type(torch.float), y.type(torch.float).to(device)).item()
            pred_full.append(output_L.cpu().numpy())
            target_full.append(y.cpu().numpy())
            
    test_loss = test_loss / float(len(test_set))
    test_l1_loss = test_l1_loss / float(len(test_set))
    
    # Track loss
    writer.add_scalar('Test/L1Loss', 
                      test_l1_loss, 
                      (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    writer.add_scalar('Test/HuberLoss', 
                      test_loss, 
                      (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    writer.add_figure('age scatter plot', 
                      plot_regression_pred(np.squeeze(pred_full), np.squeeze(target_full)), 
                      global_step = (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    
    
    print('Test loss: {:.6f}.'.format(test_loss))
    return

def pre_test(args, config, model_F, device, writer):
    
    test_params = {'batch_size': 1,
                   'shuffle': False,
                   'num_workers': config.pre_n_workers}
    pretrain_params = {'batch_size': config.pre_batch_size,
                       'shuffle': False,
                       'num_workers': config.pre_n_workers}
    
    loss_fn_F = lambda y, t: HuberLoss(y, t, 5, 20, device)
    L1Loss = nn.L1Loss()
    
    train_set = PSG_pretrain_Dataset(config, 'train')
    test_set = PSG_Dataset(config, 'test')
    test_gen = DataLoader(test_set, **test_params)
    
    # Testing
    test_loss = 0.0
    test_l1_loss = 0.0
    test_loss_avg = 0.0
    test_l1_loss_avg = 0.0
    target_epoch = []
    pred_epoch = []
    target_avg = []
    pred_avg = []
    model_F.eval()
    with torch.no_grad():
        for X, y in test_gen:
            X, y = torch.squeeze(X, 0), torch.squeeze(y, 0)
            test_epoch_set = PSG_epoch_Dataset(config, X, y)
            test_epoch_gen = DataLoader(test_epoch_set, **pretrain_params)
            output_F_class = []
            for psg, lab in test_epoch_gen:
                psg, lab = psg.to(device), lab.to(device)
                out_F, _ = model_F(psg)
                test_loss += loss_fn_F(out_F, lab).item() * lab.size(0) / float(len(test_epoch_set))
                test_l1_loss += L1Loss(out_F.type(torch.float), lab.type(torch.float)).item() * lab.size(0) / float(len(test_epoch_set))
                output_F_class.append(out_F)
                
                target_epoch.append(lab.cpu().numpy())
                pred_epoch.append(out_F.cpu().numpy())
            
            target_avg.append(y.cpu().numpy())
            pred_avg.append(torch.mean(torch.cat(output_F_class)).cpu().numpy())
            
            test_loss_avg += loss_fn_F(torch.mean(torch.cat(output_F_class)), y.to(device)).item()
            test_l1_loss_avg += L1Loss(torch.mean(torch.cat(output_F_class)).type(torch.float), y.type(torch.float).to(device)).item()
            
    test_loss = test_loss / float(len(test_set))
    test_l1_loss = test_l1_loss / float(len(test_set))
    test_loss_avg = test_loss_avg / float(len(test_set))
    test_l1_loss_avg = test_l1_loss_avg / float(len(test_set))
    
    # Track loss
    writer.add_scalar('PreTest/L1Loss', 
                      test_l1_loss, 
                      (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    writer.add_scalar('PreTest/HuberLoss', 
                      test_loss, 
                      (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    writer.add_scalar('PreTest/L1LossAvg', 
                      test_l1_loss_avg, 
                      (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    writer.add_scalar('PreTest/HuberLossAvg', 
                      test_loss_avg, 
                      (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    writer.add_figure('age epoch scatter plot', 
                      plot_regression_pred(np.concatenate(pred_epoch),np.concatenate(target_epoch)), 
                      global_step = (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    writer.add_figure('age avg scatter plot', 
                      plot_regression_pred(pred_avg, target_avg), 
                      global_step = (config.pre_max_epochs + 1) * len(train_set) / config.pre_batch_size)
    
    print('Test loss: {:.6f}. Test loss avg: {:.6f}'.format(test_loss, test_loss_avg))
    
    return

def post_test(args, config, model_L, device, writer):
    
    # Parameters
    posttrain_params = {'batch_size': config.post_batch_size,
                       'shuffle': False,
                       'num_workers': config.post_n_workers}
    
    loss_fn_L = lambda y, t: HuberLoss(y, t, 5, 20, device)
    L1Loss = nn.L1Loss()
    
    train_set = PSG_feature_Dataset(config, 'train_F')
    test_set = PSG_feature_Dataset(config, 'test_F')
    test_gen = DataLoader(test_set, **posttrain_params)
    
    # Testing
    test_loss = 0.0
    test_l1_loss = 0.0
    target_full = []
    pred_full = []
    model_L.eval()
    with torch.no_grad():
        for feat, lab in test_gen:
            feat, lab = feat.to(device), lab.to(device)
            out_L = model_L(feat)
            test_loss += loss_fn_L(out_L, lab).item() * lab.size(0) / float(len(test_set))
            test_l1_loss += L1Loss(out_L.type(torch.float), lab.type(torch.float)).item() * lab.size(0) / float(len(test_set))
            pred_full.append(out_L.cpu().numpy())
            target_full.append(lab.cpu().numpy())
    
    # Track loss
    writer.add_scalar('PostTest/L1Loss', 
                      test_l1_loss, 
                      (config.post_max_epochs + 1) * len(train_set) / config.post_batch_size)
    writer.add_scalar('PostTest/HuberLoss', 
                      test_loss, 
                      (config.post_max_epochs + 1) * len(train_set) / config.post_batch_size)
    writer.add_figure('age post scatter plot', 
                      plot_regression_pred(np.concatenate(pred_full),np.concatenate(target_full)), 
                      global_step = (config.post_max_epochs + 1) * len(train_set) / config.post_batch_size)
    
    print('Test loss: {:.6f}.'.format(test_loss))
    
    return
def main(args):
    # Training configurations
    config = Config()
    # Setup training
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # Age distribution
#    get_age_label_distribution(config)
    # Example plot
#    plot_train_epoch(config)
    # Loss function plot
    #plot_huber_loss(r = [-20, 20], d = 5, s = 20)
    
    # Model initialization
    model_F = M_PSG2FEAT(config).to(device)
    model_L = M_FEAT2LABEL(config).to(device)
    # Model Debug
    debug_model(model_F, (32, 13, 128*5*60), 2, device)
    debug_model(model_L, (64, 120, 256), 1, device)
    # Writer
    writer = SummaryWriter('runs/nAge_do075_feattrain_do50')
    # Pre-Training
    model_F = pretrain(args, config, model_F, device, writer)
    # Pre-testing
    if args.test_pre:
        pre_test(args, config, model_F, device, writer)
    # Save output features
    if args.save_feat:
        save_train_features(config, model_F, device)
    # Post-Training
    model_L = posttrain(args, config, model_L, device, writer)
    # Pre-testing
    if args.test_post:
        post_test(args, config, model_L, device, writer)
    # Training
    model_L, model_F = train(args, config, model_F, model_L, device, writer)
    # Testing
    if args.test:
        test(args, config, model_F, model_L, device, writer)
    
    writer.close()

if __name__ == "__main__":
    main(args)
    