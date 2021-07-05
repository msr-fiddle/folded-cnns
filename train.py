import argparse
import json
import os
from shutil import copyfile
import torchvision.transforms as transforms

import models
import datasets
from fold_trainer import FoldTrainer
from util.util import construct, game_models

noscope_models = [
        "noscope-coral",
        "noscope-night",
        "noscope-taipei",
        "noscope-roundabout"
        ]

def get_config(num_epoch, fold, loss, model, train_dataset, val_dataset,
               test_dataset, save_dir, do_distill, do_fold, do_efficientnet,
               optim, nclasses, do_curriculum, mb_size, do_train_val_split):

    if do_fold:
        # When folding, we put images into groups of size `fold`, so
        # we need `mb_size` to be divisible by `fold`.
        mb_size -= (mb_size % fold)

    cfg = {
        "save_dir": save_dir,
        "final_epoch": num_epoch,
        "batch_size": mb_size,
        "fold": fold,

        "Loss": loss,

        "Optimizer": optim,
        "Model": model,
        "TrainDataset": train_dataset,
        "ValDataset": val_dataset,
        "TestDataset": test_dataset,
        "distill": do_distill,
        "curriculum": do_curriculum,
        "train_val_split": do_train_val_split,
        "do_fold": do_fold,
        "do_efficientnet": do_efficientnet,
        "nclasses": nclasses
    }

    return cfg


def get_optimizer(dataset_name):
    """
    Arguments
    ---------
        dataset_name (str): name of dataset being trained

    Returns
    -------
        optimizer (dict): parameters defining the optimizer to use during
                          training.
    """
    if "noscope" in dataset_name:
        # We use the RMSprop optimizer as found in the NoScope source:
        #    https://github.com/stanford-futuredata/noscope/blob/ffc53d415a6075258a766c01621abcc65ff71200/noscope/Models.py#L33
        # with nb_layers = 1 (that means 2 layers in the NoScope implementation) and lr_mult = 1
        return {
            "class": "torch.optim.RMSprop",
            "args": {
                "lr": (0.001 / 1.5)
            }
        }
    else:
        return {
            "class": "torch.optim.Adam",
            "args": {
                "lr": 1e-04,
                "weight_decay": 1e-05
            }
        }


def get_distill(do_fold, dataset_name):
    """
    Arguments
    ---------
        do_fold (bool): whether to perform folding

        dataset_name (str): name of the dataset used in training

    Returns
    -------
        do_distill (bool): whether to use distillation loss

        distill_params (dict): parameters used for distillation training
    """
    if not do_fold or dataset_name != "cifar10":
        return False, {}

    distill_params = {
            "distill_model": "resnet18",
            "distill_model_file": "model_files/cifar10/resnet18/model.t7"
        }

    return True, distill_params


def get_model_description(dataset_name):
    """
    Arguments
    ---------
        dataset_name (str): name of dataset used in training

    Returns
    -------
        model_desc (disct): parameters of the dataset/model, as found in
                            the corresponding JSON file in the "config"
                            directory.
    """
    desc_file = os.path.join("config", dataset_name, "model_description.json")
    with open(desc_file, 'r') as infile:
        model_desc = json.load(infile)
    return model_desc


def get_loss(do_fold, dataset_name):
    """
    Arguments
    ---------
        do_fold (bool): whether to perform folding

        dataset_name (str): name of the dataset used in training

    Returns
    -------
        loss (dict): parameters specifying loss function to use

        do_distill (bool): whether to use distillation loss

        distill_params (dict): parameters used for distillation training
    """
    do_distill, distill_params = get_distill(do_fold, dataset_name)
    if do_distill:
        loss = {"class": "torch.nn.MSELoss"}
    else:
        loss = {"class": "torch.nn.CrossEntropyLoss"}

    return loss, do_distill, distill_params


def get_curriculum(do_fold, dataset_name, fold, do_efficientnet):
    """
    Arguments
    ---------
        do_fold (bool): whether to perform folding

        dataset_name (str): name of the dataset used in training

        fold (int): number of images to fold into a single input

    Returns
    -------
        do_curriculum (bool): whether to use curriculum learning

        curric_params (dict): parameters used for curriculum learning

        num_epoch (int): number of epochs to run for
    """
    num_epoch = 1500 if "noscope" not in dataset_name else 50
    if do_efficientnet or not do_fold or "cifar" in dataset_name or "noscope" in dataset_name:
        return False, {}, num_epoch
    elif dataset_name == "lol/timer-minutes" and fold == 4:
        # For lol/timer-minutes with f = 4, we double the epoch gap and halve
        # the initial number of classes and delta parameters. This is described
        # in Footnote 1 of the Appendix.
        curric_params = {
            "init_num_classes": max(num_classes // 20, int(fold)),
            "steps": [(120, max(num_classes // 20, 1))],
            "verbose": True
        }

        # The number of epochs is increased to account for the fact that it
        # takes more epochs to reach the full number of classes when using these
        # parameters. Note, however, that earlier epochs are considerably
        # shorter because they do not use all classes.
        num_epoch = 3000
    else:
        curric_params = {
            "init_num_classes": max(num_classes // 10, int(fold)),
            "steps": [(60, max(num_classes // 10, 1))],
            "verbose": True
        }

    return True, curric_params, num_epoch


def get_num_classes(model_description):
    """
    Arguments
    ---------
        model_description (dict): dictionary containing model/dataset metadata.
                                  Examples of these can be found under the
                                  "config" directory.

    Returns
    -------
        num_classes (int): number of classes in the dataset
    """
    if "num_classes" in model_description:
        return model_description["num_classes"]
    elif "ClassNames" in model_description:
        return len(model_description["ClassNames"])
    else:
        raise Exception("Can't find number of classes in model description")


def get_model(dataset_name, model_desc, num_classes, fold, do_efficientnet=False,
              internal_multiplier=None, do_fold=False, depth_mult=1.2, width_mult=1.1, res_mult=1.15):
    """
    Arguments
    ---------
        dataset_name (str): name of dataset to use.

        model_desc (dict): dictionary containing model/dataset metadata.
                           Examples of these can be found under the "config"
                           directory.

        num_classes (int): number of classes in the dataset.

        fold (int): number of images to fold into a single input. Only relevant
                    if `do_fold` is `True`.

        internal_multiplier (float): factor by which to increase the number of
                                     intermediate channels when folding. In the
                                     paper, this defaults to `sqrt(fold)`.

        do_fold (bool): whether to perform folding

    Returns
    -------
        model (dict): parameters defining the model to be trained.
    """
    if "cifar" in dataset_name:
        model = {
            "class": "models.resnet.ResNet18",
            "args": {
                "size_for_cifar": True,
                "external_fold": fold,
                "internal_multiplier": internal_multiplier,
                "num_classes": num_classes,
                "do_fold": do_fold
            }
        }

    elif do_efficientnet:
        cfgs = {
            'out-h': model_desc['out-h'],
            'out-w': model_desc['out-w'],
            'chan-in': 3,
            'num-classes': num_classes,
            'model-architecture': model_desc['model-architecture']
        }

        model = {
            "class": "models.efficientnet_game.ConvN",
            "args": {
                "cfgs": cfgs,
                "fold": fold,
                "do_fold": do_fold,
                "depth_mult": depth_mult,
                "res_mult": res_mult,
                "width_mult": width_mult
            }
        }


    elif dataset_name in game_models:
        cfgs = {
            'out-h': model_desc['out-h'],
            'out-w': model_desc['out-w'],
            'chan-in': 3,
            'num-classes': num_classes,
            'model-architecture': model_desc['model-architecture']
        }

        model = {
            "class": "models.game.ConvN",
            "args": {
                "cfgs": cfgs,
                "fold": fold,
                "channel_mult": internal_multiplier,
                "do_fold": do_fold
            }
        }

    elif dataset_name in noscope_models:
        # Model name must now be of the form "noscope-<dataset_name>"
        # Example: "noscope-coral"
        name = '-'.join(dataset_name.split('-')[1:])
        model = {
            "class": "models.noscope.noscope_cnn",
            "args": {
                "model_name": name,
                "fold": fold,
                "internal_multiplier": internal_multiplier,
                "do_fold": do_fold
            }
        }

    else:
        raise Exception("Invalid model_type: {}".format(dataset_name))

    return model


def get_dataset(dataset_name, do_augment, do_curriculum=False,
                curric_params={}, extra_transforms=[]):
    """
    Arguments
    ---------
        dataset_name (str): name of the dataset

        do_augment (bool): whether to perform data augmentation on the
                           training set. This is only relevant when this
                           function is used to get the datasets used in
                           model distillation.

        do_curriculum (bool): whether to perform curriculum learning. This is
                              only relevant if `dataset_name` is one the
                              game models.

        curric_params (dict): parameters used in curriculum learning. Only
                              relevant if `do_curriculum` is `True`.

    Returns
    -------
        {train, val, test}_dataset (dict): parameters used for each of the
                                           train, validation, and test
                                           datasets.
    """

    if "cifar" in dataset_name:
        transforms_list = []
        test_transforms_list = []
        if do_augment:
            transforms_list.append(transforms.RandomCrop(32, padding=4))
            transforms_list.append(transforms.RandomHorizontalFlip())

        for l in [transforms_list, test_transforms_list]:
            l.append(transforms.ToTensor())
            l.append(transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)))

        if dataset_name == "cifar10":
            classname = "torchvision.datasets.CIFAR10"
            rootdir = "data/cifar10"
        elif dataset_name == "cifar100":
            classname = "torchvision.datasets.CIFAR100"
            rootdir = "data/cifar100"

        train_dataset = {
            "class": classname,
            "args": {
                "root": rootdir,
                "train": True,
                "transform": transforms.Compose(transforms_list),
                "download": True
            }
        }

        # We will split the train and val datasets in `fold_trainer`
        val_dataset = {
            "class": classname,
            "args": {
                "root": rootdir,
                "train": True,
                "transform": transforms.Compose(test_transforms_list),
                "download": True
            }
        }

        test_dataset = {
            "class": classname,
            "args": {
                "root": rootdir,
                "train": False,
                "transform": transforms.Compose(test_transforms_list),
                "download": True
            }
        }

    elif dataset_name in game_models:
        rootdir = "data_root"
        transforms_list = []
        for tf in extra_transforms:
            transforms_list.append(tf)
        transforms_list.append(transforms.ToTensor())
        transformations = transforms.Compose(transforms_list)

        if do_curriculum:
            classname = "datasets.subset_image_folder.SubsetImageFolder"
        else:
            classname = "torchvision.datasets.ImageFolder"

        train_dataset = {
            "class": classname,
            "args": {
                "root": os.path.join(rootdir, dataset_name, "train"),
                "transform": transformations,
                **curric_params
            }
        }

        val_dataset = {
            "class": classname,
            "args": {
                "root": os.path.join(rootdir, dataset_name, "val"),
                "transform": transformations,
                **curric_params
            }
        }

        test_dataset = {
            "class": classname,
            "args": {
                "root": os.path.join(rootdir, dataset_name, "test"),
                "transform": transformations,
                **curric_params
            }
        }

    elif dataset_name in noscope_models:
        rootdir = "data"
        subdir = dataset_name + "-subset"
        transforms_list = []
        transforms_list.append(transforms.ToTensor())
        transformations = transforms.Compose(transforms_list)

        if do_curriculum:
            classname = "datasets.subset_image_folder.SubsetImageFolder"
        else:
            classname = "torchvision.datasets.ImageFolder"

        train_dataset = {
            "class": classname,
            "args": {
                "root": os.path.join(rootdir, subdir, "train"),
                "transform": transformations,
                **curric_params
            }
        }

        val_dataset = {
            "class": classname,
            "args": {
                "root": os.path.join(rootdir, subdir, "val"),
                "transform": transformations,
                **curric_params
            }
        }

        test_dataset = {
            "class": classname,
            "args": {
                "root": os.path.join(rootdir, subdir, "test"),
                "transform": transformations,
                **curric_params
            }
        }

    else:
        raise Exception("Unrecognized datset '{}'".format(dataset_name))

    return train_dataset, val_dataset, test_dataset


def distill_dataset(dataset_name, model, labelgen_model_file,
                    train_base_dataset, val_base_dataset, test_base_dataset,
                    num_classes, model_desc, do_fold=True):
    """
    Arguments
    ---------
        dataset_name (str): name of dataset used in training

        model (dict): description of "teacher" model used

        labelgen_model_file (str): path to checkpoint file containing teaacher
                                   weights

        train_base_dataset (dict): description of training dataset to use

        val_base_dataset (dict): description of validation dataset to use

        test_base_dataset (dict): description of testing dataset to use

        num_classes (int): number of classes in the dataset

        do_fold (bool): whether to perform folding

        model_desc (dict): description of model and dataset parameters

    Returns
    -------
        {train, val, test}_distill_dataset (dict): description of distilled datasets
    """
    train_labelgen_dataset, val_labelgen_dataset, test_labelgen_dataset = get_dataset(dataset_name, do_augment=False)

    def construct_and_randomize(dset_in):
        dset = construct(dset_in)
        #if do_fold:
        #    randomize_image_folder(dset)
        return dset

    train_labelgen_dataset = construct(train_labelgen_dataset)
    val_labelgen_dataset = construct(val_labelgen_dataset)
    test_labelgen_dataset = construct(test_labelgen_dataset)

    train_base_dataset = construct(train_base_dataset)
    val_base_dataset = construct(val_base_dataset)
    test_base_dataset = construct(test_base_dataset)

    train_distill_dataset = {
        "class": "datasets.distilled.DistilledDataset",
        "args": {
            "labelgen_dataset": train_labelgen_dataset,
            "labelgen_model": get_model(dataset_name, fold=1, internal_multiplier=1.,
                                        num_classes=num_classes, model_desc=model_desc),
            "labelgen_model_file": labelgen_model_file,
            "base_dataset": train_base_dataset,
            "num_classes": num_classes
        }
    }

    val_distill_dataset = {
        "class": "datasets.distilled.DistilledDataset",
        "args": {
            "labelgen_dataset": val_labelgen_dataset,
            "labelgen_model": get_model(dataset_name, fold=1, internal_multiplier=1.,
                                        num_classes=num_classes, model_desc=model_desc),
            "labelgen_model_file": labelgen_model_file,
            "base_dataset": val_base_dataset,
            "num_classes": num_classes
        }
    }

    test_distill_dataset = {
        "class": "datasets.distilled.DistilledDataset",
        "args": {
            "labelgen_dataset": test_labelgen_dataset,
            "labelgen_model": get_model(dataset_name, fold=1, internal_multiplier=1.,
                                        num_classes=num_classes, model_desc=model_desc),
            "labelgen_model_file": labelgen_model_file,
            "base_dataset": test_base_dataset,
            "num_classes": num_classes
        }
    }

    return train_distill_dataset, val_distill_dataset, test_distill_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("overall_save_dir", type=str,
                        help="Directory to save logs and models to")
    parser.add_argument("--checkpoint_cycle", type=int, default=1,
                        help="Number of epochs between checkpoints")
    parser.add_argument("--continue_from_file",
                        help="Path to file containing previous training state.")
    args = parser.parse_args()

    if not os.path.isdir(args.overall_save_dir):
        os.makedirs(args.overall_save_dir)

    game_datasets = [
        "lol/goldnumber-fraction",
        "apex/squad_count_v2",
        "sot/coin_count-digits_3_4-v2",
        "sot/timer3",
        "lol/goldnumber-int",
        "lol/timer-minutes",
    ]

    noscope_datasets = ["noscope-coral", "noscope-night", "noscope-taipei", "noscope-roundabout"]
    general_datasets = ["cifar10", "cifar100"]
    datasets_to_run = game_datasets + noscope_datasets + general_datasets

    # Whether to scale datasets using EfficientNet-style compound scaling
    do_efficientnet = False

    depth_mult = 1.2
    width_mult = 1.1
    res_mult = 1.15
    folds = [2, 3, 4]
    mb_size = 32

    for dataset_name in datasets_to_run:
        model_desc = get_model_description(dataset_name)
        model_name = dataset_name
        num_classes = get_num_classes(model_desc)
        do_train_val_split = ("cifar" in dataset_name)

        for fold in folds:
            if do_efficientnet and fold > 2:
                continue
            internal_multiplier = (fold ** 0.5)
            if fold == 1:
                do_fold = False
                mode_string = "none"
            else:
                do_fold = True
                mode_string = "fold"

            print(dataset_name, model_name, mode_string,
                  fold, internal_multiplier, "EfficientNet=" + str(do_efficientnet))

            do_curriculum, curric_params, num_epoch = get_curriculum(
                                                        do_fold=do_fold,
                                                        dataset_name=dataset_name,
                                                        fold=fold,
                                                        do_efficientnet=do_efficientnet)

            loss_fn, do_distill, distill_params = get_loss(
                                                    do_fold=do_fold,
                                                    dataset_name=dataset_name)

            model = get_model(dataset_name,
                              fold=fold,
                              num_classes=num_classes,
                              model_desc=model_desc,
                              internal_multiplier=internal_multiplier,
                              do_fold=do_fold,
                              do_efficientnet=do_efficientnet,
                              depth_mult=depth_mult,
                              width_mult=width_mult,
                              res_mult=res_mult)

            extra_transforms = []
            if do_efficientnet:
                width = model_desc['out-w']
                height = model_desc['out-h']
                extra_transforms.append(transforms.Resize((int(width / res_mult), int(height / res_mult))))

            train_dataset, val_dataset, test_dataset = get_dataset(
                                                            dataset_name,
                                                            do_augment=True,
                                                            do_curriculum=do_curriculum,
                                                            curric_params=curric_params,
                                                            extra_transforms=extra_transforms)
            if do_distill:
                train_dataset, val_dataset, test_dataset = distill_dataset(
                        dataset_name,
                        distill_params["distill_model"],
                        distill_params["distill_model_file"],
                        train_dataset, val_dataset, test_dataset, num_classes,
                        do_fold,
                        model_desc)

            suffix_dir = os.path.join(dataset_name,
                                      "{}".format(
                                          model_name),
                                      "fold{}".format(fold),
                                      "multiplier{}".format(internal_multiplier),
                                      mode_string)

            save_dir = os.path.join(
                args.overall_save_dir, suffix_dir)

            optim = get_optimizer(dataset_name)
            config_map = get_config(num_epoch=num_epoch,
                                    mb_size=mb_size,
                                    model=model,
                                    optim=optim,
                                    loss=loss_fn,
                                    train_dataset=train_dataset,
                                    val_dataset=val_dataset,
                                    test_dataset=test_dataset,
                                    nclasses=num_classes,
                                    save_dir=save_dir,
                                    fold=fold,
                                    do_distill=do_distill,
                                    do_fold=do_fold,
                                    do_efficientnet=do_efficientnet,
                                    do_curriculum=do_curriculum,
                                    do_train_val_split=do_train_val_split)

            if args.continue_from_file:
                config_map["continue_from_file"] = args.continue_from_file

            try:
                trainer = FoldTrainer(config_map,
                                      checkpoint_cycle=args.checkpoint_cycle)
                trainer.train()
            except KeyboardInterrupt:
                print("INTERRUPTED")
