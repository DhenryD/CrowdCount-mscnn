import os
import torch
import numpy as np
import sys
import shutil

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src.timer import Timer
from src import utils
from src.evaluate_model import evaluate_model

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import crayonclient
except ImportError:
    crayonclient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


method = 'mscnn'
dataset_name = 'shtechA'
output_dir = './saved_models/'
best_model_dir = './best_models/'

train_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train'
train_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train_den'
val_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val'
val_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/val_den'

# training configuration
start_step = 0
end_step = 600
lr = 0.0001
momentum = 0.9
disp_interval = 500
log_interval = 250

# tensorboard  config
use_tensorboard = False
save_exp_name = method + '_' + dataset_name + '_' + 'v1'
remove_all_log = False  # remove all historical experiments in tensorboard
exp_name = None  # the previous experiment name in tensorboard

# ------------
rand_seed = 64678
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

# load net
net = CrowdCounter()
network.weights_normal_init(net, std=0.01)
net.cuda()
net.train()  # sets the module in training mode

params = list(net.parameters())
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# tensorboad
use_tensorboard = use_tensorboard and crayonclient is not None
if use_tensorboard:
    cc = crayonclient(hostname='127.0.0.1')
    if remove_all_log:
        cc.remove_all_experiments()
    if exp_name is None:
        exp_name = save_exp_name
        exp = cc.create_experiment(exp_name)
    else:
        exp = cc.open_experiment(exp_name)

# training
train_loss = 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()

data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True, pre_load=True)
data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True, pre_load=True)
best_mae = sys.maxsize

for epoch in range(start_step, end_step + 1):
    step = -1
    train_loss = 0
    # update the params with one img every time
    for blob in data_loader:
        step = step + 1
        im_data = blob['data']
        gt_data = blob['gt_density']
        density_map = net(im_data, gt_data)
        loss = net.loss
        train_loss += loss.item()
        step_cnt += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % disp_interval == 0:
            duration = t.toc(average=False)
            fps = step_cnt / duration
            gt_count = np.sum(gt_data)
            density_map = density_map.data.cpu().numpy()
            et_count = np.sum(density_map)
            utils.save_results(im_data, gt_data, density_map, output_dir)
            # time indicates the time required to process a picture
            log_text = 'epoch: %4d, step %4d, time: %.4fs, gt_cnt: %4.1f, et_cnt: %4.1f' % (epoch,
                                                                                            step, 1. / fps, gt_count,
                                                                                            et_count)
            log_print(log_text, color='green', attrs=['bold'])
            re_cnt = True

        if re_cnt:
            t.tic()
            re_cnt = False

    if (epoch % 2 == 0):
        save_name = os.path.join(output_dir, '{}_{}_{}.h5'.format(method, dataset_name, epoch))
        network.save_net(save_name, net)
        # calculate error on the validation dataset
        mae, mse = evaluate_model(save_name, data_loader_val)
        if mae < best_mae:
            best_mae = mae
            best_mse = mse
            best_model = '{}_{}_{}.h5'.format(method, dataset_name, epoch)
        log_text = 'epoch: %d, mae: %.1f, mse: %0.1f' % (epoch, mae, mse)
        log_print(log_text, color='green', attrs=['bold'])
        log_text = 'best mae: %0.1f, best mse: %0.1f, best model: %s' % (best_mae, best_mse, best_model)
        log_print(log_text, color='green', attrs=['bold'])
        if use_tensorboard:
            exp.add_scalar_value('mae', mae, step=epoch)
            exp.add_scalar_value('mse', mse, step=epoch)
            exp.add_scalar_value('train_loss', train_loss / data_loader.get_num_samples(), step=epoch)

# save the best model
if not os.path.exists(best_model_dir):
    os.mkdir(best_model_dir)
oldfile = os.path.join(output_dir, best_model)
newfile = os.path.join(best_model_dir, best_model)
shutil.copyfile(oldfile, newfile)
log_text = 'best model is in the file of best_models'
log_print(log_text, color='green', attrs=['bold'])
