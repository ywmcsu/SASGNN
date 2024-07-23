
from datetime import datetime
from models.model import Model
import torch
import torch.nn as nn
import numpy as np
import time
import os
from utils.metrics import metric
from utils.tools import EarlyStopping



def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_sasgnn.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)


def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_sasgnn.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model


def inference(model, data_set, dataloader, device, node_cnt, window_size, horizon):
    forecast_set = []
    target_set = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataloader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -horizon:, :]).float()
            dec_inp = torch.cat([batch_y[:, :horizon, :], dec_inp], dim=1).float().to(device)
            step = 0

            forecast_steps = np.zeros([batch_x.size()[0], horizon, node_cnt], dtype=np.float)
            while step < horizon:
                forecast_result = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                len_model_output = forecast_result.size()[1]
                if len_model_output == 0:
                    raise Exception('Get blank inference result')

                if horizon - step <= window_size:
                    batch_x = forecast_result[:, -window_size:, :].clone()
                else:
                    if len_model_output < window_size:
                        batch_x[:, :window_size - len_model_output, :] = batch_x[:, len_model_output:window_size, :].clone()
                        batch_x[:, window_size - len_model_output:, :] = forecast_result.clone()
                    else:
                        batch_x = forecast_result[:, -window_size:, :].clone()

                forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                    forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()
                step += min(horizon - step, len_model_output)

            forecast_steps = forecast_steps[:, -horizon:, :]
            forecast_set.append(forecast_steps)
            batch_y = batch_y[:, -horizon:, :].to(device)
            target_set.append(batch_y.detach().cpu().numpy())

    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0)



def validate(model, data_set,dataloader, device, normalize_method,
             node_cnt, window_size, horizon,
             result_file=None):
    forecast_norm, target_norm = inference(model, data_set,dataloader, device,
                                           node_cnt, window_size, horizon)
    if result_file:
        if not os.path.exists(result_file):
            os.makedirs(result_file)
        step_to_print = horizon
        # step_to_print = 0
        forcasting_2d = forecast_norm[:, 0:step_to_print, :]
        forcasting_2d_target = target_norm[:, 0:step_to_print, :]
    mae, mse, rmse, mape, mspe = metric(forcasting_2d, forcasting_2d_target)
    print('metrics of vali set:  MAE: {:5.4f} | RMSE: {:5.4f}'.format(mae, rmse))
    return dict(mae=mae,mape=mape,rmse=rmse,mse= mse, mspe= mspe)


def train(args, train_set, train_loader,valid_set, valid_loader, result_file):
    early_stopping = EarlyStopping(patience=7, verbose=True)
    flag = 'train'
    node_cnt = args.node_cnt
    model = Model(args)
    model.to(args.device)
    if args.optimizer == 'RMSProp':
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.lr, eps=1e-08)
    elif args.optimizer == 'Adam':
        my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    elif args.optimizer == 'SGD':
        my_optim = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.9)
    else:
        my_optim = torch.optim.Adadelta(params=model.parameters(), lr=args.lr)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)
    forecast_loss = nn.MSELoss(reduction='mean').to(args.device)
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
    best_validate_mae = np.inf
    validate_score_non_decrease_count = 0
    performance_metrics = {}
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

            batch_x = batch_x.float().to(args.device)

            batch_y = batch_y.float().to(args.device)
            batch_x_mark = batch_x_mark.float().to(args.device)
            batch_y_mark = batch_y_mark.float().to(args.device)
            dec_inp = torch.zeros_like(batch_y[:, -args.horizon:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.labellen, :], dec_inp], dim=1).float().to(args.device)
            model.zero_grad()
            forecast= model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            batch_y = batch_y[:, -args.horizon:, :].to(args.device)
            loss = forecast_loss(forecast, batch_y)
            cnt += 1
            loss.backward()
            my_optim.step()
            loss_total += float(loss)
        print('epoch {:3d} | time: {:5.2f}s | total_loss {:5.4f}'.format(epoch, (
                time.time() - epoch_start_time), loss_total / cnt))
        if (epoch+1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            is_best_for_now = False
            performance_metrics = \
                validate(model, valid_set,valid_loader, args.device, args.norm_method,
                         node_cnt, args.window_size, args.horizon,
                         result_file=result_file)
            early_stopping(performance_metrics['mae'], model, result_file)
            if early_stopping.early_stop:
                save_model(model, result_file)
                print("Early stopping")
                break

            if best_validate_mae > performance_metrics['mae']:
                best_validate_mae = performance_metrics['mae']
                is_best_for_now = True
                validate_score_non_decrease_count = 0
            else:
                validate_score_non_decrease_count += 1
            # save model
            if is_best_for_now:
                save_model(model, result_file)
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break
    return performance_metrics


def test(args,test_set, test_loader, result_train_file, result_test_file):
    model = load_model(result_train_file)
    node_cnt = args.node_cnt
    performance_metrics = validate(model, test_set,test_loader, args.device, args.norm_method,
                      node_cnt, args.window_size, args.horizon,
                      result_file=result_test_file)
    mae, mape, rmse = performance_metrics['mae'], performance_metrics['mape'], performance_metrics['rmse']
    print('metrics of test set:  MAE: {:5.4f} | RMSE: {:5.4f}'.format(mae, rmse))
    return mae, rmse