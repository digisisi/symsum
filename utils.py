from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import numpy as np
import torch
from os import makedirs
from os.path import join, exists
import git
from tiny_imnet_dataset import TinyImageNetDataset
from typing import Tuple, List, Dict, Sequence


class ImageDataset(Dataset):
    """
    An image dataset with additional features.
    """
    def __init__(self, cifar: int, root: str = './data', train: bool = True, exclude_classes: Sequence[int] = (),
                 down_sample_data: int = 1, augs: Sequence[str] = (), **kwargs):
        """
        :param cifar: Determines which dataset to use: Cifar10, Cifar100, or TinyImageNet
        :param root: path to the root directory of all the datasets
        :param train: Whether it is the train or test set
        :param exclude_classes: Can specify to exclude some classes from the dataset by their index.
        :param down_sample_data: Can downsample the data by the specified factor.
        :param augs: Can specify which of the implemented augmentations to use
        :param kwargs:
        """
        super(ImageDataset, self).__init__()
        self.image_size = 64 if cifar==200 else 32  # TinyImageNet is 64x64
        channel_mean = (0.5071, 0.4867, 0.4408)
        channel_std = (0.2675, 0.2565, 0.2761)
        # I was using CIFAR100 mean and std in my normalization in all datasets.
        # I don't think this is a big issue given the type of results I'm presenting.
        # tiny_imnet_mean = [0.4824, 0.4495, 0.3981]
        # tiny_imnet_std = [0.2770, 0.2693, 0.2829]

        empty_aug = transforms.Compose([])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(channel_mean, channel_std),
        ])

        # The channel_mean and channel_std are for CIFAR-100
        train_transform = transforms.Compose([
            transforms.RandomEqualize(p=0.5) if 'rand_eq' in augs else empty_aug,
            transforms.RandomAutocontrast(p=0.5) if 'rand_cont' in augs else empty_aug,
            transforms.RandomHorizontalFlip() if 'mir' in augs else empty_aug,
            # `RandomCrop` may give away attack vs normal images, because of the way it is filled (i.e. the padding mode).
            transforms.RandomCrop(64 if cifar==200 else 32, padding=8 if cifar==200 else 4, padding_mode='reflect') if 'randcrop' in augs else empty_aug,
            transforms.RandomResizedCrop(64 if cifar==200 else 32, scale=(0.8, 1.0), ratio=(1, 1)) if 'randresizecrop' in augs else empty_aug,
            transforms.RandomRotation(degrees=30) if 'randrot' in augs else empty_aug,
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(0.5, 0.5, 0.5, 0.1),]), p=0.3) if 'colorjit' in augs else empty_aug,
            transforms.ToTensor(),
            transforms.Normalize(channel_mean, channel_std),
            lambda x: x + 0.03 * torch.randn_like(x) if 'noise' in augs else x
        ])

        self.transforms = train_transform if train else test_transform
        if cifar==100:
            self.img_dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=True)
        elif cifar==10:
            self.img_dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        elif cifar==200:
            self.img_dataset = TinyImageNetDataset(root_dir=join(root, 'tiny-imagenet-200/'), mode='train' if train else 'val')
        if exclude_classes:
            indices_to_exclude = list(np.concatenate([np.where(np.array(self.img_dataset.targets) == v)[0] for v in exclude_classes]))
            self.img_dataset.data = np.delete(self.img_dataset.data, indices_to_exclude, axis=0)
            self.img_dataset.targets = list(np.delete(self.img_dataset.targets, indices_to_exclude, axis=0))
        if down_sample_data > 1:
            # Down samples dataset. This is useful for debbugging on local machines etc
            self.img_dataset.data = self.img_dataset.data[::down_sample_data]
            self.img_dataset.targets = self.img_dataset.targets[::down_sample_data]

    def __len__(self) -> int:
        return len(self.img_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.img_dataset[idx]
        image = self.transforms(image)
        return image, label


def test(testset: torch.utils.data.Dataset, model: torch.nn.Module, device, criterion: torch.nn.Module = None,
         batch_size: int = 128) -> Tuple[float, float]:
    """
    Runs the test dataset over the model and return accuracy and loss.
    :param testset: Test dataset
    :param model: The model
    :param device: The device the model is on (i.e. cpu or gpu)
    :param criterion: The loss function to compute loss.
    :param batch_size: duh
    :return:
    """
    # Shuffling is enabled to make batchnorm compute better statistic in case `running stats` is disabled
    num_batches = len(testset) / batch_size
    dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    accumulated_loss = 0
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for batch, (image, label) in enumerate(dataloader):
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = criterion(output, label).mean() if criterion is not None else torch.tensor(-1)
            accumulated_loss += loss.detach().cpu()
            _, pred = output.max(1)
            correct += pred.eq(label).sum()
            total += label.size(0)

    return 100.*correct/total, accumulated_loss/num_batches


def save_model(model: torch.nn.Module, path: str, name):
    """Saves a model to the given directory.
    It creates a subdirectory called "models" in `dir`"""
    save_dir = join(path, "model_checkpoints")
    if not exists(save_dir):
        makedirs(save_dir)
    torch.save(model.state_dict(), join(save_dir, "net"+str(name)+".pth"))


def worker_init_fn(worker_id: int):
    """Used to set different random seeds for different DataLoader workers."""
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_commit_info() -> dict:
    """returns commit info for the head."""
    try:
        repo = git.Repo(search_parent_directories=True)
        commit_hash = repo.head.object.hexsha
        datetime = repo.head.commit.committed_datetime()
        message = repo.head.commit.message
        return {'git hash': commit_hash, 'git commit datetime': datetime, 'commit message': message}
    except:
        return {}


def compute_comp_weight_corr(model: torch.nn.Module) -> Tuple[np.float, List, np.float]:
    model_layers_corr = []
    all_layers_complimentary_weights = []
    with torch.no_grad():
        for name, m in model.named_modules():
            # correlation only makes sense for weights that have symmetry
            if hasattr(m, 'has_symmetry'):
                if m.weight.shape[1] % 2 == 0:
                    half_dim = m.weight.shape[1] // 2
                    all_layers_complimentary_weights.append(np.stack([m.weight[:, :half_dim, :, :].flatten().cpu().numpy(),
                                                                      m.weight[:, half_dim:, :, :].flatten().cpu().numpy()], axis=0))
                    model_layers_corr.append(np.corrcoef(m.weight[:, :half_dim, :, :].flatten().cpu().numpy(), m.weight[:, half_dim:, :, :].flatten().cpu().numpy())[0, 1])

    all_layers_complimentary_weights = np.concatenate(all_layers_complimentary_weights, axis=1)
    model_total_corr = np.corrcoef(all_layers_complimentary_weights)[0, 1]
    mean_difference = np.abs(all_layers_complimentary_weights[0] - all_layers_complimentary_weights[1]).mean()
    return model_total_corr, model_layers_corr, mean_difference


def compute_symmetry_loss(model: torch.nn.Module, norm: float, bias: float, device) -> torch.tensor:
    """Computes the Symmetry regularization loss"""
    linearity_reg_loss = torch.tensor(0., device=device)
    for m in model.modules():
        if hasattr(m, 'has_symmetry') and m.weight.shape[1] % 2 == 0:
            # `m.weight.shape[1] % 2 == 0` makes sure we skip the first convolution layer which has 3 in channels.
            half_dim = m.weight.shape[1] // 2
            diff_abs = torch.abs(m.weight[:, :half_dim, :, :] - m.weight[:, half_dim:, :, :])
            # Note: `.add(args.lin_reg_bias)` makes sure gradient in the backward pass doesn't become infinity when lin_reg_norm < 1.
            #  The smaller `lin_reg_norm` is the larger `lin_reg_bias` should be.
            linearity_reg_loss += diff_abs.add(bias).pow(norm).sum().pow(1 / norm)
    return linearity_reg_loss


class AverageMeter(object):
    """
    Simple class to compute and keep track of a metrics average.
    It could be smarter and more efficient, but it will do.
    """
    def __init__(self, window_size: int = None):
        self._avg = 0
        self._sum = 0
        self._count = 0
        self._data = []
        self.window_size = window_size

    def reset(self):
        self._avg = 0
        self._sum = 0
        self._count = 0

    def update(self, val: float, weight: float = 1):
        if self.window_size:
            self._data.append(val*weight)
            if self.window_size < len(self._data):
                self._data.pop(0)
            self._sum = sum(self._data)
            self._count = len(self._data)
            self._avg = self._sum / self._count
        else:
            self._sum += val * weight
            self._count += weight
            self._avg = self._sum / self._count

    @property
    def avg(self) -> float:
        return self._avg

    @property
    def count(self) -> float:
        return self._count


class ScalarRamp:
    """Simple utility class useful for ramping quantities up or down during training."""
    def __init__(self, start: float, end: float, num_steps: int):
        """
        :param start: start value
        :param end: end value
        :param num_steps: number of steps in the ramp
        """
        self.end = end
        self._step_size = (end - start) / num_steps
        self._current_val = start - self._step_size

    def step(self) -> float:
        """Takes one step and returns the updated value.
        Once the ramp ends, it keeps returning the end value.
        """
        if self._step_size > 0 and self._current_val < self.end:
            self._current_val += self._step_size
        elif self._step_size < 0 and self._current_val > self.end:
            self._current_val += self._step_size
        return self._current_val

    @property
    def current_val(self) -> float:
        return self._current_val
