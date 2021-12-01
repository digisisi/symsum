import torch
from models.resnets import resnet50, resnet101
from torch.optim import SGD, lr_scheduler, RMSprop, Adam
from utils import *
from tqdm import tqdm
import argparse
from datetime import datetime
import os
from os.path import join
import __main__
import wandb
import pkg_resources
import json

this_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
# In the past I have had runs that started at the same time down to the microsecond. So I had a
# random string at the end to avoid collisions when that happens.
import secrets
this_time += '_' + secrets.token_urlsafe(nbytes=2)

parser = argparse.ArgumentParser(description='Trains!')
parser.add_argument('-wandb_proj', type=str, default='symsum', help='Weights and Biases project name.')
parser.add_argument('-job_id', type=int, default=0, help='SLURM job ID')
parser.add_argument('-log_per_batch', action='store_true', default=False, help='Logs training accuracy loss per batch')
parser.add_argument('-num_workers', type=int, default=4, help='# of dataloader workers. Use 0 when debugging.')
parser.add_argument('-np_seed', type=int, default=0, help='Initial seed for numpy in the main process')
parser.add_argument('-cifar', type=int, choices=[10, 100, 200], default=10, help='CIFAR10, CIFAR100, or Tiny ImageNet. Yes, lack of foresight in naming :)')
parser.add_argument('-grad_anomaly', action='store_true', default=False, help='Activates Autograds anomaly detection. Slows down training. Good for debugging.')
parser.add_argument('-resume', type=str, default='', help='loads a checkpoint at the specified path and resumes training')
parser.add_argument('-model', choices=['res50', 'res101'], default='res50', help='Default ResNet18')
parser.add_argument('-activation', choices=['swish', 'relu', 'tanhshrink', 'softplus', 'symsum', 'leakyrelu', 'abs', 'softshrink_hardtanh', 'softsymsum',
                                            'prelu', 'tanh', 'softsign'], required=True, help='Activation function to use')
parser.add_argument('-init_gain_param', type=float, default=0., help='The parameter used to determine the gain for Kaiming He initialization.')
parser.add_argument('-fixup_init', action='store_true', default=False, help='performs fixup rescaling from arXiv:1901.09321')
parser.add_argument('-symmetry_corr', type=float, default=0., help='Adjusts the complimentary weight correlation for symlinear initialization. range is [0, 1]')
parser.add_argument('-lin_reg', type=float, default=0., help='linearity regularizer coefficient')
parser.add_argument('-lin_reg_norm', type=float, default=1., help='linearity regularizer coefficient')
parser.add_argument('-lin_reg_bias', type=float, default=1e-2, help='This is to avoid singularities. The smaller `lin_reg_norm` the larger this should be.')
parser.add_argument('-epochs', type=int, default=200, help='# of epochs to train for')
parser.add_argument('-batch_size', type=int, default=128, help='')
parser.add_argument('-optim', choices=['sgd', 'rmsprop', 'adam'], default='sgd', help='Optimizer to use')
parser.add_argument('-lr', type=float, default=0.05, help='learning rate for the model')
parser.add_argument('-momentum', type=float, default=0.9, help='Optimizer\'s momentum')
parser.add_argument('-lr_gamma', type=float, default=0.3, help='coefficient of lr decay')
parser.add_argument('-lr_milestones', nargs='+', type=int, default=[50, 100, 150], help='Epochs at which lr decays')
parser.add_argument('-weight_decay', type=float, default=1e-4, help='weight decay regularization')
parser.add_argument('-bn_affine', action='store_true', default=False, help='make batchnorm layers affine')
parser.add_argument('-bn_rs', action='store_true', default=False, help='batchnorm layers to keep running stats (vs. using batch stats)')
parser.add_argument('-norm_layer', choices=['batch', 'instance', 'none'], default='batch', help='select the type of normalization layer.')
parser.add_argument('-exclude_classes', type=int, default=[], nargs='*', help='classes to exclude from training and test. range is 0-99')
parser.add_argument('-down_sample_data', type=int, default=1, help='Downsamples the datasets by the specified factor. Useful for debugging on local machines.')
parser.add_argument('-augs', nargs='+', default=[], help='List of augmentations to apply in training')
parser.add_argument('-parallel', action='store_true', default=False, help='Do DataParallel if there are multiple GPUs')
parser.add_argument('-sm_temp_ramp', type=float, default=(1, 1), nargs=2, help='Can ramp down the softmax temprature to 1. Takes two arguments '
                                                                    '(initial temp, ramp length in num of batches)')
args = parser.parse_args()


# Creating a subdirectory to save the artifacts
calling_script_name = os.path.basename(__main__.__file__)
script_log_path = join('artifacts/training_logs', calling_script_name)
if not os.path.isdir(script_log_path):
    os.makedirs(script_log_path)
run_logs_dir = join(script_log_path, this_time)
if not os.path.isdir(run_logs_dir):
    os.makedirs(run_logs_dir)
user_home = os.path.split(os.path.expanduser('~'))[-1]  # This makes it easier know whose runs it is on W&B

# Inferring number of class and image resolution from dataset.
num_classes = args.cifar
img_size = 64 if args.cifar==200 else 32  # Cifars are 32x32 and Tiny ImageNet is 64x64
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Creating the model
if args.model.startswith('res'):
    if args.model.endswith('50'):
        model = resnet50(num_classes=num_classes, **vars(args)).to(device)
    elif args.model.endswith('101'):
        model = resnet101(num_classes=num_classes, **vars(args)).to(device)
    pass
else:
    raise ValueError(f'{args.model} architecture not implemented')

# Loading the model from a pretrained checkpoint
if args.resume:
    model.load_state_dict(torch.load(args.resume))

if torch.cuda.device_count() > 1 and args.parallel:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)

# Setting up the optimizer
parameters_to_train = [{'params': p, 'weight_decay': args.weight_decay} for name, p in
                       model.named_parameters() if p.requires_grad]
if args.optim == 'sgd':
    optim = SGD(parameters_to_train, lr=args.lr, momentum=args.momentum)
elif args.optim == 'rmsprop':
    optim = RMSprop(parameters_to_train, lr=args.lr, alpha=0.9, momentum=args.momentum)
elif args.optim == 'adam':
    optim = Adam(parameters_to_train, lr=args.lr)
scheduler = lr_scheduler.MultiStepLR(optim, milestones=args.lr_milestones, gamma=args.lr_gamma)
criterion = torch.nn.CrossEntropyLoss(reduction='none')

# Creating the datasets
train_dataset = ImageDataset(train=True, **vars(args))
test_dataset = ImageDataset(train=False, **vars(args))

# Initializing Weights&Biases
wandb.init(project=args.wandb_proj, name=user_home + '_' + this_time, dir=run_logs_dir, settings=wandb.Settings(start_method='fork'))
wandb.config.update(args)
wandb.config.update(get_commit_info())
installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
wandb.config.update({'installed_packages': installed_packages})
wandb.watch(model, log='all', log_freq=len(train_dataset)//args.batch_size)  # logs parameters and gradients once every epoch.

# List of installed packages is saved for debugging purposes.
with open(join(run_logs_dir, 'installed_packages.json'), 'w') as outfile:
    json.dump(installed_packages, outfile)

if args.grad_anomaly:
    torch.autograd.set_detect_anomaly(True)


old_test_loss = float('inf')
batch_num = 0
softmax_temp_ramp = ScalarRamp(args.sm_temp_ramp[0], 1., args.sm_temp_ramp[1])
for epoch in range(args.epochs):
    toral_corr, layers_corr, mean_difference = compute_comp_weight_corr(model)
    wandb.log({'Total comp weight corr': toral_corr, 'Total comp weight mean difference': mean_difference}, step=epoch)

    np.random.seed(args.np_seed + epoch)
    total_train_samples = 0

    test_acc, test_loss = test(test_dataset, model, device, criterion)
    wandb.log({'Test Loss': test_loss, 'Test Acc.': test_acc}, step=epoch)
    # Saving a checkpoint if test loss has improved or every 10 epochs.
    if test_loss < old_test_loss or (epoch % 10 == 0 and epoch > 0):
        save_model(model, run_logs_dir, epoch)
        old_test_loss = test_loss
    elif epoch % 10 == 0 and epoch > 0:
        # Saves a checkpoint every 10 epochs
        save_model(model, run_logs_dir, epoch)

    pbar = tqdm(DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=worker_init_fn))
    num_batches = np.ceil(len(train_dataset) / args.batch_size)
    acc_avg = AverageMeter()
    loss_avg = AverageMeter()
    model.train()
    for batch, (image, label) in enumerate(pbar):
        image, label = image.to(device), label.to(device)
        total_loss = torch.tensor(0.).to(device)

        output = model(image)
        softmax_temp = softmax_temp_ramp.step()
        class_loss = criterion(output / softmax_temp, label).mean()
        total_loss += class_loss
        if args.lin_reg > 0. and args.activation in ('symsum', 'softsymsum') and batch > 0:
            # Todo: this currently does Lp norm, but I could do an exponential norm of the form:
            #  \frac{\ln\left(b\operatorname{abs}\left(x\right)+1\right)}{\ln\left(b\right)}
            #  Such norm could be more desirable, because its gradient decays faster.
            total_loss += args.lin_reg * compute_symmetry_loss(model, args.lin_reg_norm, args.lin_reg_bias, device)
        optim.zero_grad()
        total_loss.backward()
        optim.step()
        _, pred = output.max(1)

        loss_avg.update(class_loss.item(), weight=label.size(0))
        acc_avg.update(pred.eq(label).to(torch.float).mean().item(), weight=label.size(0))

        pbar.set_description('epoch: %d | Loss: %.3f | Acc: %.3f%% | Count %d'
                             % (epoch, loss_avg.avg, 100 * acc_avg.avg, acc_avg.count))

        if args.log_per_batch:
            # Logging per-batch metrics during the first epoch to compare SymSum vs. ReLU
            batch_num = epoch * num_batches + batch
            wandb.log({'Train Batch Loss': class_loss.item(),
                       'Train Batch Acc.': 100 * pred.eq(label).to(torch.float).mean().item(), 'batch': batch_num, })

    scheduler.step()
    batch_num = (epoch+1) * num_batches

    wandb.log({'Train Loss': loss_avg.avg, 'Train Acc.': 100 * acc_avg.avg, 'batch': batch_num}, step=epoch + 1)


