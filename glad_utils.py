import torch
import numpy as np
import copy
import utils
import wandb
import os
import torchvision
import gc
from tqdm import tqdm

from utils import get_network, config, evaluate_synset

def build_dataset(ds, class_map, num_classes):
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(ds))):
        sample = ds[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])
    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    return images_all, labels_all, indices_class


def prepare_latents(channel=3, num_classes=10, im_size=(32, 32), zdim=512, G=None, class_map_inv={}, get_images=None, args=None):
    with torch.no_grad():
        ''' initialize the synthetic data '''
        label_syn = torch.tensor([i*np.ones(args.ipc, dtype=np.int64) for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.space == 'p':
            latents = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=False, device=args.device)
            f_latents = None

        else:
            zs = torch.randn(num_classes * args.ipc, zdim, device=args.device, requires_grad=False)

            if "imagenet" in args.dataset:
                one_hot_dim = 1000
            elif args.dataset == "CIFAR10":
                one_hot_dim = 10
            elif args.dataset == "CIFAR100":
                one_hot_dim = 100
            if args.avg_w:
                G_labels = torch.zeros([label_syn.nelement(), one_hot_dim], device=args.device)
                G_labels[
                    torch.arange(0, label_syn.nelement(), dtype=torch.long), [class_map_inv[x.item()] for x in
                                                                         label_syn]] = 1
                new_latents = []
                for label in G_labels:
                    zs = torch.randn(1000, zdim).to(args.device)
                    ws = G.mapping(zs, torch.stack([label] * 1000))
                    w = torch.mean(ws, dim=0)
                    new_latents.append(w)
                latents = torch.stack(new_latents)
                del zs
                for _ in new_latents:
                    del _
                del new_latents


            else:
                G_labels = torch.zeros([label_syn.nelement(), one_hot_dim], device=args.device)
                G_labels[
                    torch.arange(0, label_syn.nelement(), dtype=torch.long), [class_map_inv[x.item()] for x in
                                                                              label_syn]] = 1
                if args.distributed and False:
                    latents = G.mapping(zs.to("cuda:1"), G_labels.to("cuda:1")).to("cuda:0")
                else:
                    latents = G.mapping(zs, G_labels)
                del zs

            del G_labels

            ws = latents
            if args.layer is not None:
                f_latents = torch.cat(
                    [G.forward(split_ws, f_layer=args.layer, mode="to_f").detach() for split_ws in
                     torch.split(ws, args.sg_batch)])
                f_type = f_latents.dtype
                f_latents = f_latents.to(torch.float32).cpu()
                f_latents = torch.nan_to_num(f_latents, posinf=5.0, neginf=-5.0)
                f_latents = torch.clip(f_latents, min=-10, max=10)
                f_latents = f_latents.to(f_type).cuda()

                print(torch.mean(f_latents), torch.std(f_latents))

                if args.rand_f:
                    f_latents = (torch.randn(f_latents.shape).to(args.device) * torch.std(
                        f_latents, dim=(1,2,3), keepdim=True) + torch.mean(f_latents, dim=(1,2,3), keepdim=True))

                f_latents = f_latents.to(f_type)
                print(torch.mean(f_latents), torch.std(f_latents))
                f_latents.requires_grad_(True)
            else:
                f_latents = None

        if args.pix_init == 'real' and args.space == "p":
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                latents.data[c*args.ipc:(c+1)*args.ipc] = torch.cat([get_images(c, 1).detach().data for s in range(args.ipc)])
        else:
            print('initialize synthetic data from random noise')

        latents = latents.detach().to(args.device).requires_grad_(True)

        return latents, f_latents, label_syn


def get_optimizer_img(latents=None, f_latents=None, G=None, args=None):
    if args.space == "wp" and (args.layer is not None and args.layer != -1):
        optimizer_img = torch.optim.SGD([latents], lr=args.lr_w, momentum=0.5)
        optimizer_img.add_param_group({'params': f_latents, 'lr': args.lr_img, 'momentum': 0.5})
    else:
        optimizer_img = torch.optim.SGD([latents], lr=args.lr_img, momentum=0.5)

    if args.learn_g:
        G.requires_grad_(True)
        optimizer_img.add_param_group({'params': G.parameters(), 'lr': args.lr_g, 'momentum': 0.5})

    optimizer_img.zero_grad()

    return optimizer_img

def get_eval_lrs(args):
    eval_pool_dict = {
        args.model: 0.001,
        "ResNet18": 0.001,
        "VGG11": 0.0001,
        "AlexNet": 0.001,
        "ViT": 0.001,

        "AlexNetCIFAR": 0.001,
        "ResNet18CIFAR": 0.001,
        "VGG11CIFAR": 0.0001,
        "ViTCIFAR": 0.001,
    }

    return eval_pool_dict


def eval_loop(latents=None, f_latents=None, label_syn=None, G=None, best_acc={}, best_std={}, testloader=None, model_eval_pool=[], it=0, channel=3, num_classes=10, im_size=(32, 32), args=None):
    curr_acc_dict = {}
    max_acc_dict = {}

    curr_std_dict = {}
    max_std_dict = {}

    eval_pool_dict = get_eval_lrs(args)

    save_this_it = False
    
    for model_eval in model_eval_pool:

        if model_eval != args.model and args.wait_eval and it != args.Iteration:
            continue
        print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (
        args.model, model_eval, it))

        accs_test = []
        accs_train = []

        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size, width=args.width, depth=args.depth,
                                   dist=False).to(args.device)  # get a random model
            eval_lats = latents
            eval_labs = label_syn
            image_syn = latents
            image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                eval_labs.detach())  # avoid any unaware modification

            if args.space == "wp":
                with torch.no_grad():
                    image_syn_eval = torch.cat(
                        [latent_to_im(G, (image_syn_eval_split, f_latents_split), args=args).detach() for
                         image_syn_eval_split, f_latents_split, label_syn_split in
                         zip(torch.split(image_syn_eval, args.sg_batch), torch.split(f_latents, args.sg_batch),
                             torch.split(label_syn, args.sg_batch))])

            args.lr_net = eval_pool_dict[model_eval]
            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader,
                                                     args=args, aug=True)
            del _
            del net_eval
            accs_test.append(acc_test)
            accs_train.append(acc_train)

        print(accs_test)
        accs_test = np.array(accs_test)
        accs_train = np.array(accs_train)
        acc_test_mean = np.mean(np.max(accs_test, axis=1))
        acc_test_std = np.std(np.max(accs_test, axis=1))
        best_dict_str = "{}".format(model_eval)
        if acc_test_mean > best_acc[best_dict_str]:
            best_acc[best_dict_str] = acc_test_mean
            best_std[best_dict_str] = acc_test_std
            save_this_it = True

        curr_acc_dict[best_dict_str] = acc_test_mean
        curr_std_dict[best_dict_str] = acc_test_std

        max_acc_dict[best_dict_str] = best_acc[best_dict_str]
        max_std_dict[best_dict_str] = best_std[best_dict_str]

        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
        len(accs_test[:, -1]), model_eval, acc_test_mean, np.std(np.max(accs_test, axis=1))))
        wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
        wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[best_dict_str]}, step=it)
        wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
        wandb.log({'Max_Std/{}'.format(model_eval): best_std[best_dict_str]}, step=it)

    wandb.log({
        'Accuracy/Avg_All'.format(model_eval): np.mean(np.array(list(curr_acc_dict.values()))),
        'Std/Avg_All'.format(model_eval): np.mean(np.array(list(curr_std_dict.values()))),
        'Max_Accuracy/Avg_All'.format(model_eval): np.mean(np.array(list(max_acc_dict.values()))),
        'Max_Std/Avg_All'.format(model_eval): np.mean(np.array(list(max_std_dict.values()))),
    }, step=it)

    curr_acc_dict.pop("{}".format(args.model))
    curr_std_dict.pop("{}".format(args.model))
    max_acc_dict.pop("{}".format(args.model))
    max_std_dict.pop("{}".format(args.model))

    wandb.log({
        'Accuracy/Avg_Cross'.format(model_eval): np.mean(np.array(list(curr_acc_dict.values()))),
        'Std/Avg_Cross'.format(model_eval): np.mean(np.array(list(curr_std_dict.values()))),
        'Max_Accuracy/Avg_Cross'.format(model_eval): np.mean(np.array(list(max_acc_dict.values()))),
        'Max_Std/Avg_Cross'.format(model_eval): np.mean(np.array(list(max_std_dict.values()))),
    }, step=it)

    return save_this_it


def load_sgxl(res, args=None):
    import sys
    import os
    p = os.path.join("stylegan_xl")
    if p not in sys.path:
        sys.path.append(p)
    import dnnlib
    import legacy
    from sg_forward import StyleGAN_Wrapper
    device = torch.device('cuda')
    if args.special_gan is not None:
        if args.special_gan == "ffhq":
            # network_pkl = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/ffhq{}.pkl".format(res)
            network_pkl = "../stylegan_xl/ffhq{}.pkl".format(res)
            key = "G_ema"
        elif args.special_gan == "pokemon":
            # network_pkl = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/pokemon{}.pkl".format(res)
            network_pkl = "../stylegan_xl/pokemon{}.pkl".format(
                res)
            key = "G_ema"

    elif "imagenet" in args.dataset:
        if args.rand_gan_con:
            network_pkl = "../stylegan_xl/random_conditional_{}.pkl".format(res)
            key = "G"
        elif args.rand_gan_un:
            network_pkl = "../stylegan_xl/random_unconditional_{}.pkl".format(res)
            key = "G"
        else:
            network_pkl = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet{}.pkl".format(res)
            key = "G_ema"
    elif args.dataset == "CIFAR10":
        if args.rand_gan_un:
            network_pkl = "../stylegan_xl/random_unconditional_32.pkl"
            key = "G"
        else:
            network_pkl = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/cifar10.pkl"
            key = "G_ema"
    elif args.dataset == "CIFAR100":
        if args.rand_gan_con:
            network_pkl = "../stylegan_xl/random_conditional_32.pkl"
            key = "G"
        elif args.rand_gan_un:
            network_pkl = "../stylegan_xl/random_unconditional_32.pkl"
            key = "G"
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)[key]
        G = G.eval().requires_grad_(False).to(device)

    z_dim = G.z_dim
    w_dim = G.w_dim
    num_ws = G.num_ws

    G.eval()
    mapping = G.mapping
    G = StyleGAN_Wrapper(G)
    gpu_num = torch.cuda.device_count()
    if gpu_num > 1:

        G = nn.DataParallel(G)
        mapping = nn.DataParallel(mapping)

    G.mapping = mapping

    return G, z_dim, w_dim, num_ws


def latent_to_im(G, latents, args=None):

    if args.space == "p":
        return latents

    mean, std = config.mean, config.std

    if "imagenet" in args.dataset:
        class_map = {i: x for i, x in enumerate(config.img_net_classes)}

        if args.space == "p":
            im = latents

        elif args.space == "wp":
            if args.layer is None or args.layer==-1:
                im = G(latents[0], mode="wp")
            else:
                im = G(latents[0], latents[1], args.layer, mode="from_f")

        im = (im + 1) / 2
        im = (im - mean) / std

    elif args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
        if args.space == "p":
            im = latents
        elif args.space == "wp":
            if args.layer is None or args.layer == -1:
                im = G(latents[0], mode="wp")
            else:
                im = G(latents[0], latents[1], args.layer, mode="from_f")

            if args.distributed and False:
                mean, std = config.mean_1, config.std_1

        im = (im + 1) / 2
        im = (im - mean) / std

    return im


def image_logging(latents=None, f_latents=None, label_syn=None, G=None, it=None, save_this_it=None, args=None):
    with torch.no_grad():
        image_syn = latents.cuda()

        if args.space == "wp":
            with torch.no_grad():
                if args.layer is None or args.layer == -1:
                    image_syn = latent_to_im(G, (image_syn.detach(), None), args=args)
                else:
                    image_syn = torch.cat(
                        [latent_to_im(G, (image_syn_split.detach(), f_latents_split.detach()), args=args).detach() for
                         image_syn_split, f_latents_split, label_syn_split in
                         zip(torch.split(image_syn, args.sg_batch),
                             torch.split(f_latents, args.sg_batch),
                             torch.split(label_syn, args.sg_batch))])

        save_dir = os.path.join(args.logdir, args.dataset, wandb.run.name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(image_syn.cpu(), os.path.join(save_dir, "images_{0:05d}.pt".format(it)))
        torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_{0:05d}.pt".format(it)))

        if save_this_it:
            torch.save(image_syn.cpu(), os.path.join(save_dir, "images_best.pt".format(it)))
            torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_best.pt".format(it)))

        wandb.log({"Latent_Codes": wandb.Histogram(torch.nan_to_num(latents.detach().cpu()))}, step=it)

        if args.ipc < 50 or args.force_save:

            upsampled = image_syn
            if "imagenet" not in args.dataset:
                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
            wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
            wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)

            for clip_val in []:
                upsampled = torch.clip(image_syn, min=-clip_val, max=clip_val)
                if "imagenet" not in args.dataset:
                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                wandb.log({"Clipped_Synthetic_Images/raw_{}".format(clip_val): wandb.Image(
                    torch.nan_to_num(grid.detach().cpu()))}, step=it)

            for clip_val in [2.5]:
                std = torch.std(image_syn)
                mean = torch.mean(image_syn)
                upsampled = torch.clip(image_syn, min=mean - clip_val * std, max=mean + clip_val * std)
                if "imagenet" not in args.dataset:
                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(
                    torch.nan_to_num(grid.detach().cpu()))}, step=it)

    del upsampled, grid


def gan_backward(latents=None, f_latents=None, image_syn=None, G=None, args=None):
    f_latents.grad = None
    latents_grad_list = []
    f_latents_grad_list = []
    for latents_split, f_latents_split, dLdx_split in zip(torch.split(latents, args.sg_batch),
                                                          torch.split(f_latents, args.sg_batch),
                                                          torch.split(image_syn.grad, args.sg_batch)):
        latents_detached = latents_split.detach().clone().requires_grad_(True)
        f_latents_detached = f_latents_split.detach().clone().requires_grad_(True)

        syn_images = latent_to_im(G=G, latents=(latents_detached, f_latents_detached), args=args)

        syn_images.backward((dLdx_split,))

        latents_grad_list.append(latents_detached.grad)
        f_latents_grad_list.append(f_latents_detached.grad)

        del syn_images
        del latents_split
        del f_latents_split
        del dLdx_split
        del f_latents_detached
        del latents_detached

        gc.collect()

    latents.grad = torch.cat(latents_grad_list)
    del latents_grad_list
    if args.layer != -1:
        f_latents.grad = torch.cat(f_latents_grad_list)
        del f_latents_grad_list


