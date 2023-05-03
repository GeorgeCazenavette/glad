import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, \
    TensorDataset, epoch, load_generator, latent_to_im, DiffAugment, ParamDiffAug, do_fft, fft_project, fft_loss, load_biggan
import copy

import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):

    outer_loop_default, inner_loop_default = get_loops(args.ipc)

    args.lr_net = args.lr_teacher

    if args.outer_loop is None:
        args.outer_loop = outer_loop_default
    if args.inner_loop is None:
        args.inner_loop = inner_loop_default

    if args.g_train_ipc is None:
        args.g_train_ipc = args.ipc
    if args.g_eval_ipc is None:
        args.g_eval_ipc = args.ipc
    if args.g_grad_ipc is None:
        args.g_grad_ipc = args.ipc

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa_param.blur_perc = args.blur_perc
    args.dsa_param.blur_min = args.blur_min
    args.dsa_param.blur_max = args.blur_max

    if args.lr_img is None:
        if args.space == 'p':
            args.lr_img = 0.1
        else:
            args.lr_img = 0.01

    if args.batch_syn is None:
        args.batch_syn = args.ipc

    args.num_batches = args.ipc // args.batch_syn

    run_name = ""

    if args.clip:
        run_name += "clip_"

    run_name += "{}_".format(args.model)

    run_name += "{}_".format(args.dataset)

    if args.dataset == "ImageNet":
        run_name += "{}_".format(args.subset)
        run_name += "{}_".format(args.res)


    run_name += "space_{}_".format(args.space)
    if args.space != "p":
        run_name += "tanh_{}_".format(args.tanh)
        run_name += "proj_{}_".format(args.proj_ball)
        run_name += "trunc_{}_".format(args.trunc)
        run_name += "layer_{}_".format(args.layer)

    if args.space == "p":
        run_name += "init_{}_".format(args.pix_init)
    elif args.space == "z":
        run_name += "init_{}_".format(args.gan_init)
        run_name += "RandCond_{}_".format(args.rand_cond)
        run_name += "RandLat_{}_".format(args.rand_lat)

    elif args.patch:
        run_name += "patch_"

    elif args.rand_gen:
        run_name += "rand-gen_"

    if args.spec_proj:
        run_name += "spec-proj_"

    if args.spec_reg is not None:
        run_name += "spec-reg_{}_".format(args.spec_reg)


    run_name += "aug_{}_".format(args.dsa)

    run_name += "ipc_{}_".format(args.ipc)

    run_name += "batch_{}_".format(args.batch_syn)

    run_name += "ol_{}_il_{}_".format(args.outer_loop, args.inner_loop)

    run_name += "im-opt_{}_".format(args.im_opt)

    run_name += "eval_{}_".format(args.eval_mode)

    args.save_path = os.path.join(args.save_path, run_name)

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # eval_it_pool = np.arange(0, args.syn_batches*args.Iteration+1, 100).tolist() if args.eval_mode == 'S' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args.res, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    im_res = im_size[0]

    # print('\n================== Exp %d ==================\n '%exp)
    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    save_dir = os.path.join(args.buffer_path, args.dataset)
    if args.pm1:
        save_dir = os.path.join(args.buffer_path, "tanh", args.dataset)
    else:
        save_dir = os.path.join(args.buffer_path, args.dataset)
    # if args.dataset == "ImageNet":
    #     save_dir = os.path.join(save_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and args.zca:
        save_dir += "_ZCA"
    save_dir = os.path.join(save_dir, args.model)

    save_dir = os.path.join(save_dir, "depth-{}".format(args.depth), "width-{}".format(args.width))

    save_dir = os.path.join(save_dir, args.norm_train)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    if args.dataset != "ImageNet" or True:
        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]
        print(len(dst_train))
        print("BUILDING DATASET")
        for i in tqdm(range(len(dst_train))):
            sample = dst_train[i]
            images_all.append(torch.unsqueeze(sample[0], dim=0))
            labels_all.append(class_map[torch.tensor(sample[1]).item()])
        # images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in tqdm(range(len(dst_train)))]
        # labels_all = [class_map[dst_train[i][1]] for i in tqdm(range(len(dst_train)))]
        for i, lab in tqdm(enumerate(labels_all)):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to("cpu")
        labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    criterion = nn.CrossEntropyLoss().to(args.device)

    trajectories = []

    dst_train = TensorDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach()))
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    ''' set augmentation for whole-dataset training '''
    # args.dsa = False
    args.dc_aug_param = get_daparam(args.dataset, args.model, args.model, None)
    args.dc_aug_param['strategy'] = 'crop_scale_rotate'  # for whole-dataset training
    print('DC augmentation parameters: \n', args.dc_aug_param)

    for it in range(0, args.Iteration):

        ''' Train synthetic data '''
        teacher_net = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width, norm=args.norm_train).to(args.device) # get a random model
        teacher_net.train()
        # if torch.cuda.device_count() > 1:
        #     teacher_net = torch.nn.DataParallel(teacher_net)
        lr = args.lr_teacher
        # teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)  # optimizer_img for synthetic data
        teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)  # optimizer_img for synthetic data
        teacher_optim.zero_grad()

        timestamps = []

        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

        lr_schedule = [args.train_epochs // 2 + 1]

        for e in range(args.train_epochs):


            train_loss, train_acc = epoch("train", dataloader=trainloader, net=teacher_net, optimizer=teacher_optim,
                                        criterion=criterion, args=args, aug=True)

            test_loss, test_acc = epoch("test", dataloader=testloader, net=teacher_net, optimizer=None,
                                        criterion=criterion, args=args, aug=False)

            print("Itr: {}\tEpoch: {}\tReal Acc: {}\tTest Acc: {}".format(it, e, train_acc, test_acc))

            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

            if e in lr_schedule and args.decay:
                lr *= 0.1
                teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
                teacher_optim.zero_grad()

        trajectories.append(timestamps)

        if len(trajectories) == args.save_interval:
            n = 0
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []

    print(trajectories)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='subset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode')  # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    # parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=None, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=None, help='learning rate learning rate')
    parser.add_argument('--mom_img', type=float, default=0.5, help='momentum for updating synthetic images')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_syn', type=int, default=None, help='batch size for syn data')
    parser.add_argument('--syn_batches', type=int, default=1, help='number of synthetic batches')
    parser.add_argument('--pix_init', type=str, default='noise', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--gan_init', type=str, default='class', choices=["class", "rand"])
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--blur', action='store_true')
    parser.add_argument('--patch', action='store_true')
    parser.add_argument('--tanh', action='store_true')
    parser.add_argument('--proj_ball', action='store_true')
    parser.add_argument('--rand_gen', action='store_true')
    parser.add_argument('--spec_proj', action='store_true')
    parser.add_argument('--spec_reg', type=float, default=None)
    parser.add_argument('--clip', action='store_true')
    parser.add_argument('--im_opt', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--outer_loop', type=int, default=None)
    parser.add_argument('--inner_loop', type=int, default=None)
    parser.add_argument('--skip_epochs', type=int, default=0)
    parser.add_argument('--train_epochs', type=int, default=200)
    parser.add_argument('--trunc', type=float, default=1, help='truncation_trick')
    # parser.add_argument('--res', type=int, default=128, choices=[128, 256, 512], help='resolution')
    parser.add_argument('--res', type=int, default=128, help='resolution')
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--blur_perc', type=float, default=0.0)
    parser.add_argument('--blur_min', type=float, default=0.0)
    parser.add_argument('--blur_max', type=float, default=3.0)
    parser.add_argument('--rand_cond', action='store_true')
    parser.add_argument('--rand_lat', action='store_true')
    parser.add_argument('--coarse2fine', action='store_true')
    parser.add_argument('--teacher', type=str, default='fake', choices=['real', 'fake'])
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--texture', action='store_true')
    group.add_argument('--tex_avg', action='store_true')
    parser.add_argument('--tex_it', type=int, default=500)
    parser.add_argument('--tex_batch', type=int, default=10)
    parser.add_argument('--lr_decay', type=str, default='none', choices=['none', 'cosine', 'linear', 'step'])
    # parser.add_argument('--g_grad_ipc', type=int, default=10)
    parser.add_argument('--g_train_ipc', type=int, default=None)
    parser.add_argument('--g_eval_ipc', type=int, default=None)
    parser.add_argument('--g_grad_ipc', type=int, default=None)
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--learn_labels', action='store_true')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--mom', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--decay', type=bool, default=False)
    parser.add_argument('--cl_subset', default=None)
    parser.add_argument('--kip_zca', action='store_true')
    parser.add_argument('--pm1', action='store_true')

    parser.add_argument('--canvas_size', type=int, default=None)
    parser.add_argument('--canvas_samples', type=int, default=1)
    parser.add_argument('--canvas_stride', type=int, default=1)

    parser.add_argument('--space', type=str, default='p', choices=['p', 'z', 'w', 'w+', 'wp', 'g'],
                        help='[ p | z | w | w+ ]')

    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--depth', type=int, default=3)

    parser.add_argument('--norm_train', type=str, default="batchnorm")

    parser.add_argument('--norm_eval', type=str, default="none")

    # For speeding up, we can decrease the Iteration and epoch_eval_train, which will not cause significant performance decrease.

    args = parser.parse_args()
    main(args)


