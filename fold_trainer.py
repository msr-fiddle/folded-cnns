import os
import psutil
import random
import scipy.stats
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm

from util.subset_sampler import SubsetSampler
import util.stats
import util.util
from util.util import construct, try_cuda


class FoldTrainer(object):
    """
    Top-level class which carries out the training of a folded model.
    """

    def __init__(self, config_map, checkpoint_cycle=1):
        """
        Arguments
        ----------
        config_map (dict): dictionary containing the full specification of
                           components to be used in training.

        checkpoint_cycle (int): number of epochs between model checkpoints
        """
        self.__init_from_config_map(config_map)
        self.checkpoint_cycle = checkpoint_cycle

    def train(self):
        """
        Trains folded model as specified via configuration.
        """
        no_grad_epoch = torch.no_grad()(self.__epoch)

        if self.do_curriculum:
            # The `begin` method sets up the initial samples that will be used
            # by the dataset. In some cases, the samples list maintained by
            # the underlying ImageFolder will be shuffled after construction.
            # We call `begin` here so as to construct the sub-list after
            # shuffling has been performed
            self.train_dataloader.dataset.begin()
            self.test_dataloader.dataset.begin()

            if self.val_dataloader:
                self.val_dataloader.dataset.begin()

        while self.cur_epoch < self.final_epoch:
            # Perform epoch on training set
            train_loss, train_acc = self.__epoch(self.train_dataloader, "train", do_step=True)

            # Perform epoch on validation set
            _, val_acc = no_grad_epoch(self.val_dataloader, "val", do_step=False)

            # Perform epoch on test dataset
            _, _ = no_grad_epoch(self.test_dataloader,
                                 "test",
                                 do_step=False,
                                 do_print=False)

            if self.do_schedule:
                self.lr_scheduler.step()

            # Update curriculum datasets, if necessary
            if self.do_curriculum:
                self.train_dataloader.dataset.epoch()
                self.test_dataloader.dataset.epoch()

                if self.val_dataloader:
                    self.val_dataloader.dataset.epoch()

            self.__save_current_state(val_acc)
            self.cur_epoch += 1

            # Place functions back on GPU, if necessary
            self.model = try_cuda(self.model)
            self.loss_fn = try_cuda(self.loss_fn)


    def __epoch(self, data_loader, label, do_step=False, do_print=True):
        """
        Performs a single epoch of either training or validation.

        Arguments
        ----------
        data_loader (DataLoader): data loader to use for this cycle

        label (str): name identifying the dataset being run (e.g., "train")

        do_step (bool): whether to make optimization steps. For example, one
                        might only set this to `True` when `data_loader` is
                        that for the training dataset.

        do_print (bool): whether to print loss and accuracy. One might set this
                         to `False` for the test dataset.
        """
        stats = util.stats.StatsTracker()

        if label == "train":
            self.model.train()
        else:
            self.model.eval()

        if do_print:
            data_loader = tqdm(data_loader, ascii=True,
                               desc="Epoch {}. {}".format(self.cur_epoch, label))

        for mb_vals in data_loader:
            mb_data = try_cuda(mb_vals[0])
            mb_true_labels = try_cuda(mb_vals[1])
            if len(mb_vals) == 3:
                mb_distilled_labels = try_cuda(mb_vals[2])
            else:
                mb_distilled_labels = None

            num_channels = mb_data.size(1)
            og_batch_size = mb_data.size(0)

            # Make sure that we have a number of samples that is divisible by our
            # fold.
            if self.do_fold and not self.do_efficientnet:
                if (mb_data.size(0) % self.fold) != 0:
                    crop = mb_data.size(0) // self.fold
                    if crop == 0:
                        continue
                    crop = crop * self.fold
                    mb_data = mb_data[:crop]
                    mb_true_labels = mb_true_labels[:crop]
                    if mb_distilled_labels is not None:
                        mb_distilled_labels = mb_distilled_labels[:crop]
                    og_batch_size = crop

                # Perform the stack
                mb_data = mb_data.view(-1, num_channels * self.fold, *(mb_data.size()[2:]))

            if do_step:
                self.opt.zero_grad()

            outputs = self.model(mb_data)

            if self.do_fold and not self.do_efficientnet:
                # Reshape output before calculating loss
                outputs = outputs.view(og_batch_size, -1)

            if mb_distilled_labels is not None:
                loss = self.loss_fn(outputs, mb_distilled_labels)
            else:
                loss = self.loss_fn(outputs, mb_true_labels)

            stats.update_loss(loss.cpu().item())
            stats.update_accuracies(outputs, mb_true_labels)

            if do_step:
                loss.backward()
                self.opt.step()

            if do_print:
                rloss, rtop1, rtop5 = stats.running_averages()
                data_loader.set_description(
                    "Epoch {}. {}. Top-1={:.4f}, Top-5={:.4f}, Loss={:.4f}".format(
                    self.cur_epoch, label, rtop1, rtop5, rloss))

        epoch_loss, epoch_acc_map = stats.averages()
        outfile_fmt = os.path.join(self.save_dir, label + "_{}.txt")
        epoch_map = epoch_acc_map
        epoch_map["loss"] = epoch_loss
        util.util.write_vals_dict(outfile_fmt, epoch_map)

        top1 = epoch_acc_map["top1"]
        return epoch_loss, top1

    def __save_current_state(self, validation_reconstruction_accuracy):
        """
        Serializes and saves the current state associated with training.
        """
        is_best = False
        if validation_reconstruction_accuracy > self.best_accuracy:
            self.best_accuracy = validation_reconstruction_accuracy
            is_best = True

        save_dict = {
            "epoch": self.cur_epoch,
            "best_val_acc": self.best_accuracy,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
        }

        util.util.save_checkpoint(
            save_dict, self.save_dir, "current.pth", is_best)

    def __init_from_config_map(self, config_map):
        """
        Initializes state for training based on the contents of `config_map`.
        """
        # If "continue_from_file" is set, we load previous state for training
        # from the associated value.
        prev_state = None
        if "continue_from_file" in config_map and config_map["continue_from_file"] is not None:
            prev_state = util.util.load_state(config_map["continue_from_file"])

        self.batch_size = config_map["batch_size"]
        self.fold = config_map["fold"]

        extra_kwargs = {}
        if not config_map["distill"]:
            print("Using extra workers")
            extra_kwargs["num_workers"] = psutil.cpu_count() - 1
            extra_kwargs["pin_memory"] = True

        self.do_fold = config_map["do_fold"]
        self.do_efficientnet = config_map["do_efficientnet"]
        tds = construct(config_map["TrainDataset"])
        vds = construct(config_map["ValDataset"])
        sds = construct(config_map["TestDataset"])
        if not config_map["distill"] and self.do_fold and "CIFAR" not in config_map["TrainDataset"]["class"]:
            util.util.randomize_image_folder(tds)
            util.util.randomize_image_folder(vds)
            util.util.randomize_image_folder(sds)

        if config_map["train_val_split"]:
            indices = list(range(len(tds)))

            # 19 is chosen because 2019 was a good year
            random.seed(19)
            random.shuffle(indices)
            num_val = 5000 + (self.fold - (5000 % self.fold))
            train_sampler = data.sampler.SubsetRandomSampler(indices[:-num_val])
            val_sampler = SubsetSampler(indices[-num_val:])
            self.train_dataloader = data.DataLoader(
                    tds,
                    batch_size=self.batch_size,
                    sampler=train_sampler, **extra_kwargs)

            self.val_dataloader = data.DataLoader(
                    vds,
                    batch_size=self.batch_size,
                    sampler=val_sampler, **extra_kwargs)
        else:
            self.train_dataloader = data.DataLoader(
                    tds,
                    batch_size=self.batch_size,
                    shuffle=True, **extra_kwargs)

            if "root" in config_map["TrainDataset"]["args"] and "noscope" in config_map["TrainDataset"]["args"]["root"]:
                v_indices = list(range(len(vds)))
                random.seed(19)
                random.shuffle(v_indices)
                v_sampler = SubsetSampler(v_indices)
                print("Shuffle val")
                self.val_dataloader = data.DataLoader(
                        vds,
                        batch_size=self.batch_size,
                        sampler=v_sampler, **extra_kwargs)

            else:
                self.val_dataloader = data.DataLoader(
                        vds,
                        batch_size=self.batch_size,
                        shuffle=False, **extra_kwargs)

        if "root" in config_map["TrainDataset"]["args"] and "noscope" in config_map["TrainDataset"]["args"]["root"]:
            s_indices = list(range(len(sds)))
            random.seed(19)
            random.shuffle(s_indices)
            print("Shuffle test")
            s_sampler = SubsetSampler(s_indices)
            self.test_dataloader = data.DataLoader(
                    sds,
                    batch_size=self.batch_size,
                    sampler=s_sampler, **extra_kwargs)

        else:
            self.test_dataloader = data.DataLoader(
                    sds,
                    batch_size=self.batch_size,
                    shuffle=False, **extra_kwargs)

        self.loss_fn = construct(config_map["Loss"])
        self.loss_fn = try_cuda(self.loss_fn)

        self.model = construct(config_map["Model"])
        util.util.init_weights(self.model)
        self.opt = construct(config_map["Optimizer"],
                             {"params": self.model.parameters()})
        self.model = try_cuda(self.model)

        self.cur_epoch = 0
        self.best_accuracy = 0.0
        self.final_epoch = config_map["final_epoch"]
        self.num_classes = config_map["nclasses"]
        self.do_distill = config_map["distill"]
        self.do_curriculum = config_map["curriculum"]

        self.do_schedule = ("CIFAR" in config_map["TrainDataset"]["class"])
        if self.do_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.opt, step_size=200, gamma=0.5)

        # If we are loading from a previous state, update our model,
        # optimizers, and current status of training so that we can continue.
        if prev_state is not None:
            self.model.load_state_dict(prev_state["model"])
            self.cur_epoch = prev_state["epoch"]
            self.best_accuracy = prev_state["best_val_acc"]
            self.opt.load_state_dict(prev_state["opt"])

        # Directory to save stats and checkpoints to
        self.save_dir = config_map["save_dir"]
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
