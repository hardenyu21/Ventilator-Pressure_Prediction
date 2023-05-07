import torch
from torch import nn
from utils.util import get_model
from data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from utils.loss import criterion_calculator
import pandas as pd
import tqdm
import os

def dl_model_train_epoch(model, data, optimizer, scheduler, loss, device, epoch, args):
    model.train()
    progress_bar = tqdm.tqdm(enumerate(data), total=len(data))
    total_loss = 0
    print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
    for i, (X, y) in progress_bar:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        l = loss(pred, y)
        total_loss += l.data
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
        s = ('%10s' + '%10.4g' + '%10s') % ('%g/%g' % (epoch + 1, args.num_epochs), total_loss / (i + 1), mem)
        progress_bar.set_description(s)
    scheduler.step()

    return total_loss / (i + 1)

def dl_model_eval(model, data, device):
    model.eval()
    progress_bar = tqdm.tqdm(enumerate(data), total=len(data))
    creterion = criterion_calculator()
    with torch.no_grad():
        for _, (X, y) in progress_bar:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            creterion.add_item(pred, y)
        eval_result = creterion.get_item()
        print("validation loss: ", eval_result['MSE'][0])
    return eval_result

def dl_model_train_eval(args, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args)
    model = model.to(device)
    train_data = DataLoader(Dataset(args, split = 'train'), batch_size = args.batch_size, 
                      shuffle = True, num_workers = args.num_workers)
    val_data = DataLoader(Dataset(args, split = 'val'), batch_size = args.batch_size, 
                          shuffle = False, num_workers = args.num_workers)
    optimizer = AdamW(model.parameters(), lr = args.learning_rate, 
                      betas = (0.9, 0.999), weight_decay = args.weight_decay) 
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = args.gamma)
    loss = nn.MSELoss(reduction = 'mean')
    Epochs = args.num_epochs
    best_mse = float('inf')
    loss_dict = {'training_loss': [], 'validation_loss': []}
    for epoch in range(Epochs):
        train_loss = dl_model_train_epoch(model, train_data, optimizer, scheduler,loss, device, epoch, args)
        loss_dict['training_loss'].append(train_loss.cpu().item())
        eval_result = dl_model_eval(model, val_data,device)
        loss_dict['validation_loss'].append(eval_result['MSE'][0])
        if eval_result['MSE'][0] < best_mse:
            best_mse = eval_result['MSE'][0]
            results = eval_result
            if not os.path.exists(os.path.join(args.output_root, args.model_name)):
                os.makedirs(os.path.join(args.output_root, args.model_name))
            torch.save(model.state_dict(), os.path.join(args.output_root, args.model_name, 'model.pt'))
    
    logger.info(f' Saving the best model to {os.path.join(args.output_root, args.model_name, "model.pt")}')
    logger.info(f' Saving the loss during training to {os.path.join(args.output_root, args.model_name, "train.csv")}')
    pd.DataFrame(loss_dict).to_csv(os.path.join(args.output_root, args.model_name, "train.csv"), index = False)
    logger.info(f' The Evaluation results of the best model')
    print('{\n')
    for key, value in results.items():
        print(f'    {key}: {value[0]}\n')
    print('}\n')

   