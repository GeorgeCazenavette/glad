import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, match_loss, get_time, \
    TensorDataset, epoch, DiffAugment, ParamDiffAug
from tqdm import tqdm
import torchvision
import random
import gc
import wandb

from glad_utils import *

def main(args):
    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    run = wandb.init(
        project="GLaD",
        job_type="DC",
        config=args,
    )

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    run_dir = "{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), run.name)

    args.save_path = os.path.join(args.save_path, "dc", run_dir)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(
        args.dataset, args.data_path, args.batch_real, args.res, args=args)

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


    accs_all_exps = dict()  # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []
    args.distributed = torch.cuda.device_count() > 1
    if args.space == 'p':
        G, zdim = None, None
    elif args.space == 'wp':
        G, zdim, w_dim, num_ws = load_sgxl(args.res, args)

    images_all, labels_all, indices_class = build_dataset(dst_train, class_map, num_classes)

    real_train_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True,
                                                    num_workers=16)

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle].to(args.device)

    latents, f_latents, label_syn = prepare_latents(channel=channel, num_classes=num_classes, im_size=im_size, zdim=zdim, G=G, class_map_inv=class_map_inv, get_images=get_images, args=args)

    optimizer_img = get_optimizer_img(latents=latents, f_latents=f_latents, G=G, args=args)

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins' % get_time())

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    print('%s training begins' % get_time())

    best_acc = {"{}".format(m): 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}

    eval_pool_dict = get_eval_lrs(args)

    save_this_it = False
    for it in range(args.Iteration+1):

        if it in eval_it_pool and it > 0:
            save_this_it = eval_loop(latents=latents, f_latents=f_latents, label_syn=label_syn, G=G, best_acc=best_acc,
                                     best_std=best_std, testloader=testloader,
                                     model_eval_pool=model_eval_pool, channel=channel, num_classes=num_classes,
                                     im_size=im_size, it=it, args=args)


        if it > 0 and ((it in eval_it_pool and (save_this_it or it % 1000 == 0)) or (
                args.save_it is not None and it % args.save_it == 0)):
            image_logging(latents=latents, f_latents=f_latents, label_syn=label_syn, G=G, it=it, save_this_it=save_this_it, args=args)

        ''' Train synthetic data '''
        net = get_network(args.model, channel, num_classes, im_size, depth=args.depth, width=args.width).to(args.device) # get a random model
        net.train()
        net_parameters = list(net.parameters())
        optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
        optimizer_net.zero_grad()
        loss_avg = 0
        args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.


        for ol in range(args.outer_loop):

            ''' freeze the running mu and sigma for BatchNorm layers '''
            # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
            # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
            # This would make the training with BatchNorm layers easier.

            BN_flag = False
            BNSizePC = 16  # for batch normalization
            for module in net.modules():
                if 'BatchNorm' in module._get_name(): #BatchNorm
                    BN_flag = True
            if BN_flag:
                img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                net.train() # for updating the mu, sigma of BatchNorm
                output_real = net(img_real) # get running mu, sigma
                for module in net.modules():
                    if 'BatchNorm' in module._get_name():  #BatchNorm
                        module.eval() # fix mu and sigma of every BatchNorm layer

            if args.space == "wp":
                with torch.no_grad():
                    image_syn_w_grad = torch.cat([latent_to_im(G, (syn_image_split, f_latents_split), args) for
                                       syn_image_split, f_latents_split, label_syn_split in
                                       zip(torch.split(latents, args.sg_batch),
                                           torch.split(f_latents, args.sg_batch),
                                           torch.split(label_syn, args.sg_batch))])
            else:
                image_syn_w_grad = latents

            if args.space == "wp":
                image_syn = image_syn_w_grad.detach()
                image_syn.requires_grad_(True)
            else:
                image_syn = image_syn_w_grad
            ''' update synthetic data '''

            optimizer_img.zero_grad()
            for c in range(num_classes):
                loss = torch.tensor(0.0).to(args.device)
                img_real = get_images(c, args.batch_real)
                lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                if args.dsa:
                    seed = int(time.time() * 1000) % 100000
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                output_real = net(img_real)
                loss_real = criterion(output_real, lab_real)
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))

                output_syn = net(img_syn)
                loss_syn = criterion(output_syn, lab_syn)
                gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                loss = match_loss(gw_syn, gw_real, args)



                loss.backward()
                loss_avg += loss.item()
                del img_real, output_real, loss_real, gw_real, output_syn, loss_syn, gw_syn, loss



            if args.space == "wp":
                # this method works in-line and back-props gradients to latents and f_latents
                gan_backward(latents=latents, f_latents=f_latents, image_syn=image_syn, G=G, args=args)

            else:
                latents.grad = image_syn.grad.detach().clone()


            optimizer_img.step()
            optimizer_img.zero_grad()


            if ol == args.outer_loop - 1:
                break


            ''' update network '''
            image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
            dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
            trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
            for il in range(args.inner_loop):
                epoch('train', trainloader, net, optimizer_net, criterion, args, aug = True if args.dsa else False)


        loss_avg /= (num_classes*args.outer_loop)

        wandb.log({
            "Loss": loss_avg
        }, step=it)

        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

        if it == args.Iteration: # only record the final results
            data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
            torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))


if __name__ == '__main__':
    import shared_args

    parser = shared_args.add_shared_args()

    parser.add_argument('--lr_img', type=float, default=1, help='learning rate for pixels or f_latents')
    parser.add_argument('--lr_w', type=float, default=0.001, help='learning rate for updating synthetic latent w')
    parser.add_argument('--lr_g', type=float, default=0.0001, help='learning rate for gan weights')

    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--inner_loop', type=int, default=1, help='inner loop')
    parser.add_argument('--outer_loop', type=int, default=1, help='outer loop')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    args = parser.parse_args()

    main(args)


