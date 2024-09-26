import argparse
from utils import load_config, get_log_name, set_seed, save_results, \
                  get_test_acc, print_config
from datasets import cifar_dataloader, clothing_dataloader, webvision_dataloader, food101N_dataloader, animal10N_dataloader, tiny_imagenet_dataloader
import algorithms
import numpy as np
import nni
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    '-c',
                    type=str,
                    default='./configs/DISC_CIFAR.py',
                    help='The path of config file.')
parser.add_argument('--model_name', type=str, default='DISC')
parser.add_argument('--dataset', type=str, default='cifar-10')
parser.add_argument('--noisy_label_list', type=str, default='./datasets/pairflip_labels.pt')
parser.add_argument('--root', type=str, default='./datasets/CIFAR10')
parser.add_argument('--save_path', type=str, default='./log/')
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--noise_type', type=str, default='sym')
parser.add_argument('--percent', type=float, default=0.4)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--momentum', type=float, default=0.99)
args = parser.parse_args()


def main():
    tuner_params = nni.get_next_parameter()
    config = load_config(args.config, _print=False)
    config.update(tuner_params)
    config['dataset'] = args.dataset
    config['root'] = args.root
    config['gpu'] = args.gpu
    config['noise_type'] = args.noise_type
    config['percent'] = args.percent
    config['seed'] = args.seed
    config['num_classes'] = args.num_classes
    config['momentum'] = args.momentum
    noisy_label_list = torch.load(args.noisy_label_list)
    print_config(config)
    set_seed(config['seed'])

    if config['algorithm'] == 'DISC':
        model = algorithms.DISC(config,
                                input_channel=config['input_channel'],
                                num_classes=config['num_classes'])
        train_mode = 'train_index'
        
    elif config['algorithm'] == 'colearning':
        model = algorithms.Colearning(config,
                                      input_channel=config['input_channel'],
                                      num_classes=config['num_classes'])
        train_mode = 'train'
        
    elif config['algorithm'] == 'JointOptimization':
        model = algorithms.JointOptimization(
            config,
            input_channel=config['input_channel'],
            num_classes=config['num_classes'])
        train_mode = 'train_index'
        
    elif config['algorithm'] == 'GJS':
        model = algorithms.GJS(config,
                               input_channel=config['input_channel'],
                               num_classes=config['num_classes'])
        train_mode = 'train_index'
        
    elif config['algorithm'] == 'ELR':
        model = algorithms.ELR(config,
                               input_channel=config['input_channel'],
                               num_classes=config['num_classes'])
        train_mode = 'train_index'
        
    elif config['algorithm'] == 'PENCIL':
        model = algorithms.PENCIL(config,
                                  input_channel=config['input_channel'],
                                  num_classes=config['num_classes'])
        train_mode = 'train_index'
        
    else:
        model = algorithms.__dict__[config['algorithm']](
            config,
            input_channel=config['input_channel'],
            num_classes=config['num_classes'])
        train_mode = 'train_single'
        if config['algorithm'] == 'StandardCETest':
            train_mode = 'train_index'

    dataloaders = cifar_dataloader(cifar_type=config['dataset'],
                                  root=config['root'],
                                  batch_size=config['batch_size'],
                                  num_workers=config['num_workers'],
                                  noise_type=config['noise_type'],
                                  percent=config['percent'])
    trainloader = dataloaders.run(mode=train_mode, noisy_targets=noisy_label_list, filtered_index=None)
    hard_indices = np.load('./datasets/test_hard_ids_lt_train02aum.npy')
    easy_test_loader, hard_test_loader = dataloaders.run(mode='test', noisy_targets=None, filtered_index=hard_indices)

    num_easy_test_images = len(easy_test_loader.dataset)
    num_hard_test_images = len(hard_test_loader.dataset)

    start_epoch = 0
    epoch = 0

    # evaluate models with random weights
    easy_test_acc = get_test_acc(model.evaluate(easy_test_loader))
    hard_test_acc = get_test_acc(model.evaluate(hard_test_loader))
    print('Epoch [%d/%d] Test Accuracy on the %s EASY test images: %.4f' %
          (epoch, config['epochs'], num_easy_test_images, easy_test_acc))
    print('Epoch [%d/%d] Test Accuracy on the %s HARD test images: %.4f' %
          (epoch, config['epochs'], num_hard_test_images, hard_test_acc))

    acc_list, acc_all_list = [], []
    best_acc, best_epoch = 0.0, 0
    
    # loading training labels
    if config['algorithm'] == 'DISC' or config['algorithm'] == 'StandardCETest':
        model.get_labels(trainloader)
        model.weak_labels = model.labels.detach().clone()
        print('The labels are loaded!!!')

    since = time.time()
    for epoch in range(start_epoch, config['epochs']):
        # train
        model.train(trainloader, epoch)

        # evaluate
        easy_test_acc = get_test_acc(model.evaluate(easy_test_loader))
        hard_test_acc = get_test_acc(model.evaluate(hard_test_loader))

        if best_acc < easy_test_acc:
            best_acc, best_epoch = easy_test_acc, epoch

        print('Epoch [%d/%d] Test Accuracy on the %s EASY test images: %.4f' %
              (epoch, config['epochs'], num_easy_test_images, easy_test_acc))
        print('Epoch [%d/%d] Test Accuracy on the %s HARD test images: %.4f' %
              (epoch, config['epochs'], num_hard_test_images, hard_test_acc))

        if epoch >= config['epochs'] - 10:
            acc_list.extend([easy_test_acc])

        acc_all_list.extend([easy_test_acc])

    time_elapsed = time.time() - since
    total_min = time_elapsed // 60
    hour = total_min // 60
    min = total_min % 60
    sec = time_elapsed % 60

    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        hour, min, sec))

    if config['save_result']:
        config['algorithm'] = config['algorithm'] + args.model_name
        acc_np = np.array(acc_list)
        nni.report_final_result(acc_np.mean())
        jsonfile = get_log_name(config, path=args.save_path)
        np.save(jsonfile.replace('.json', '.npy'), np.array(acc_all_list))
        save_results(config=config,
                     last_ten=acc_np,
                     best_acc=best_acc,
                     best_epoch=best_epoch,
                     jsonfile=jsonfile)


if __name__ == '__main__':
    main()
