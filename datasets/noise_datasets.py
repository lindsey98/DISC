# -*- coding: utf-8 -*-
# @Author : Jack (thanks for Cheng Tan. Most of the codes afapted from his codebase "Co-training-based_noisy-label-learning-master")
# @Email  : liyifan20g@ict.ac.cn
# @File   : noise_datasets.py

import numpy as np
import torchvision
from PIL import Image
import torch
from .cifar import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .utils import noisify
from .randaugment import TransformFixMatch_CIFAR10, TransformFixMatch_CIFAR100, Transform_2strong_CIFAR100, TransformGJS, Transform2Weak_CIFAR10, Transform2Weak_CIFAR100
from typing import List

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class NoiseDataset(torchvision.datasets.VisionDataset):

    def __init__(
        self,
        noise_type: str = 'none',
        asym_trans: dict = None,
        percent: float = 0.0,
    ) -> None:

        assert percent <= 1.0 and percent >= 0.0
        assert noise_type in ['sym', 'asym', 'ins', 'none']

        self.percent = percent
        self.noise_type = noise_type
        self.asym_trans = asym_trans

        # dataset info
        self.min_target = min(self.targets)
        self.max_target = max(self.targets)
        self.num_classes = len(np.unique(self.targets))
        assert self.num_classes == self.max_target - self.min_target + 1
        self.num_samples = len(self.targets)

        if self.noise_type == 'sym':
            self.symmetric_noise()
        elif self.noise_type == 'asym':
            self.asymmetric_noise()
        elif self.noise_type == 'ins':
            self.instance_noise(tau=self.percent)

    def symmetric_noise(self):
        type = 1

        if type == 1:
            indices = np.random.permutation(len(self.data))
            for i, idx in enumerate(indices):
                if i < self.percent * len(self.data):
                    self.targets[idx] = np.random.randint(
                        low=self.min_target,
                        high=self.max_target + 1,
                        dtype=np.int32)
        else:
            random_state = 0
            if self.num_classes == 10:
                dataset = 'cifar10'
            elif self.num_classes == 100:
                dataset = 'cifar100'
            train_noisy_labels, actual_noise_rate = noisify(
                dataset=dataset,
                train_labels=np.array(self.targets).reshape([50000, 1]),
                noise_type=self.noise_type,
                noise_rate=self.percent,
                random_state=random_state,
                nb_classes=self.num_classes)
            self.targets = [int(label) for label in train_noisy_labels]
            print("Actual noise rate is %.4f" % actual_noise_rate)

    def asymmetric_noise(self):
        type = 2
        if type == 1:
            target_copy = self.targets.copy()
            if self.asym_trans == None:
                indices = np.arange(self.num_samples)
                np.random.shuffle(indices)
                idx = indices[:int(self.percent * self.num_samples)]
                target_copy = np.array(target_copy)
                target_copy[idx] = (target_copy[idx] + 1) % (
                    self.max_target + 1) + self.min_target
                self.targets = target_copy
            else:
                for i in self.asym_trans.keys():
                    indices = list(np.where(np.array(target_copy) == i)[0])
                    np.random.shuffle(indices)
                    for j, idx in enumerate(indices):
                        if j <= self.percent * len(indices):
                            self.targets[idx] = (self.asym_trans[i] if i
                                                 in self.asym_trans.keys() else
                                                 i)
            del target_copy
        else:
            random_state = 0
            # import ipdb; ipdb.set_trace()
            if self.num_classes == 10:
                dataset = 'cifar10'
            elif self.num_classes == 100:
                dataset = 'cifar100'
                self.noise_type = 'pairflip'
            train_noisy_labels, actual_noise_rate = noisify(
                dataset=dataset,
                train_labels=np.array(self.targets).reshape([50000, 1]),
                noise_type=self.noise_type,
                noise_rate=self.percent,
                random_state=random_state,
                nb_classes=self.num_classes)
            self.targets = [int(label) for label in train_noisy_labels]
            print("Actual noise rate is %.4f" % actual_noise_rate)

    def instance_noise(
        self,
        tau: float = 0.2,
        std: float = 0.1,
        feature_size: int = 3 * 32 * 32,
        # seed: int = 1
    ):
        '''
        Thanks the code from https://github.com/SML-Group/Label-Noise-Learning wrote by SML-Group.
        LabNoise referred much about the generation of instance-dependent label noise from this repo.
        '''
        from scipy import stats
        from math import inf
        import torch.nn.functional as F

        # np.random.seed(int(seed))
        # torch.manual_seed(int(seed))
        # torch.cuda.manual_seed(int(seed))

        # common-used parameters
        num_samples = self.num_samples
        num_classes = self.num_classes

        P = []
        # sample instance flip rates q from the truncated normal distribution N(\tau, {0.1}^2, [0, 1])
        flip_distribution = stats.truncnorm((0 - tau) / std, (1 - tau) / std,
                                            loc=tau,
                                            scale=std)
        '''
        The standard form of this distribution is a standard normal truncated to the range [a, b]
        notice that a and b are defined over the domain of the standard normal. 
        To convert clip values for a specific mean and standard deviation, use:

        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        truncnorm takes  and  as shape parameters.

        so the above `flip_distribution' give a truncated standard normal distribution with mean = `tau`,
        range = [0, 1], std = `std`
        '''
        # import ipdb; ipdb.set_trace()
        # how many random variates you need to get
        q = flip_distribution.rvs(num_samples)
        # sample W \in \mathcal{R}^{S \times K} from the standard normal distribution N(0, 1^2)
        W = torch.tensor(
            np.random.randn(num_classes, feature_size,
                            num_classes)).float().to(
                                device)  #K*dim*K, dim=3072
        for i in range(num_samples):
            x, y = self.transform(Image.fromarray(self.data[i])), torch.tensor(
                self.targets[i])
            x = x.to(device)
            # step (4). generate instance-dependent flip rates
            # 1 x feature_size  *  feature_size x 10 = 1 x 10, p is a 1 x 10 vector
            p = x.reshape(1, -1).mm(W[y]).squeeze(0)  #classes
            # step (5). control the diagonal entry of the instance-dependent transition matrix
            # As exp^{-inf} = 0, p_{y} will be 0 after softmax function.
            p[y] = -inf
            # step (6). make the sum of the off-diagonal entries of the y_i-th row to be q_i
            p = q[i] * F.softmax(p, dim=0)
            p[y] += 1 - q[i]
            P.append(p)
        P = torch.stack(P, 0).cpu().numpy()
        l = [i for i in range(self.min_target, self.max_target + 1)]
        new_label = [np.random.choice(l, p=P[i]) for i in range(num_samples)]

        print('noise rate = ', (new_label != np.array(self.targets)).mean())
        self.targets = new_label


class NoiseCIFAR10(CIFAR10):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform=None,
        noisy_targets: np.ndarray=None,
        filtered_index: List[int]=None,
        filter_out: bool=False,
        download=True,
        mode: str = None,
        percent: float = 0.0,
    ) -> None:

        self.transform_train_weak = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        self.transform_train_strong = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        if mode == 'train_index':
            self.transform_train = TransformFixMatch_CIFAR10(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        elif mode == 'train_index_2strong':
            self.transform_train = TransformFixMatch_CIFAR10(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        self.root = root
        self.mode = mode
        self.transform = self.transform_test

        CIFAR10.__init__(self,
                         root=root,
                         train=train,
                         transform=transform,
                         download=download)

        ## fixme
        if noisy_targets is not None:
            clean_train_labels = np.array(self.targets)
            all_noisy_train_labels = np.asarray(noisy_targets)
            # Calculate the number of labels to replace
            num_to_replace = int(len(clean_train_labels) * percent)

            # Randomly select indices to replace
            np.random.seed(1)
            indices_to_replace = np.random.choice(len(clean_train_labels), num_to_replace, replace=False)

            # Create a copy of clean_train_labels to modify
            modified_labels = clean_train_labels.copy()

            # Replace the selected indices with noisy labels
            modified_labels[indices_to_replace] = all_noisy_train_labels[indices_to_replace]

            self.targets = [int(label) for label in modified_labels]

        if filtered_index is not None:
            if filter_out:
                self.data = [x for i, x in enumerate(self.data) if i not in filtered_index]
                self.targets = [x for i, x in enumerate(self.targets) if i not in filtered_index]
            else:
                self.data = [x for i, x in enumerate(self.data) if i in filtered_index]
                self.targets = [x for i, x in enumerate(self.targets) if i in filtered_index]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]
        image = Image.fromarray(image)

        if self.mode == 'train_single':
            img = self.transform_train_weak(image)
            return img, target
        elif self.mode == 'train':
            raw = self.transform_train_weak(image)
            img1 = self.transform_train_strong(image)
            img2 = self.transform_train_strong(image)
            return raw, img1, img2, target
        elif self.mode == 'train_index': # this mode is selected
            img = self.transform_train(image)
            return img, target, index
        elif self.mode == 'test':
            img = self.transform_test(image)
            return img, target


class NoiseCIFAR100(CIFAR100, NoiseDataset):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform=None,
        download=True,
        mode: str = None,
        noise_type: str = 'none',
        percent: float = 0.0,
    ) -> None:

        self.transform_train_weak = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409),
                                 (0.2673, 0.2564, 0.2762)),
        ])
        self.transform_train_strong = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomGrayscale(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409),
                                 (0.2673, 0.2564, 0.2762)),
        ])
        # import ipdb; ipdb.set_trace()
        if mode == 'train_index':
            self.transform_train = TransformFixMatch_CIFAR100(
                (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        if mode == 'train_index_2strong':
            self.transform_train = Transform_2strong_CIFAR100(
                (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409),
                                 (0.2673, 0.2564, 0.2762)),
        ])

        self.root = root
        self.mode = mode
        self.transform = self.transform_test
        self.target_transform = None

        CIFAR100.__init__(self,
                          root=root,
                          train=train,
                          transform=transform,
                          download=download)
        NoiseDataset.__init__(self,
                              noise_type=noise_type,
                              asym_trans=None,
                              percent=percent)

    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]
        image = Image.fromarray(image)

        if self.mode == 'train_single':
            img = self.transform_train_weak(image)
            return img, target
        elif self.mode == 'train':
            raw = self.transform_train_weak(image)
            img1 = self.transform_train_strong(image)
            img2 = self.transform_train_strong(image)
            return raw, img1, img2, target
        elif self.mode == 'train_index' or self.mode == 'train_index_2strong':
            img = self.transform_train(image)
            return img, target, index
        elif self.mode == 'test':
            img = self.transform_test(image)
            return img, target


class cifar_dataloader():

    def __init__(self, cifar_type, root, batch_size, num_workers, noise_type,
                 percent):
        self.cifar_type = cifar_type
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.noise_type = noise_type
        self.percent = percent

    def run(self, mode, noisy_targets, filtered_index):
        if mode != 'test':
            train_dataset = NoiseCIFAR10(root=self.root, train=True, transform=None,
                                         noisy_targets=noisy_targets,
                                         filtered_index=filtered_index,
                                        percent=self.percent, mode=mode)
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=False,
                                      num_workers=self.num_workers)
            return train_loader
        else:
            hard_test_dataset = NoiseCIFAR10(self.root, train=False, transform=None,
                                        noisy_targets=noisy_targets,
                                        filtered_index=filtered_index,
                                         filter_out=False,
                                        percent=0.0, mode=mode)
            easy_test_dataset = NoiseCIFAR10(self.root, train=False, transform=None,
                                             noisy_targets=noisy_targets,
                                             filtered_index=filtered_index,
                                             filter_out=True,
                                             percent=0.0, mode=mode)
            hard_test_loader = DataLoader(dataset=hard_test_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     drop_last=False,
                                     num_workers=self.num_workers)
            easy_test_loader = DataLoader(dataset=easy_test_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          drop_last=False,
                                          num_workers=self.num_workers)
            return easy_test_loader, hard_test_loader


