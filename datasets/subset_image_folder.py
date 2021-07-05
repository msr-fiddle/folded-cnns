from torchvision.datasets import ImageFolder


class SubsetImageFolder(ImageFolder):
    """
    A PyTorch dataset that gradually introduces classes of a dataset throughout
    training.

    Usage:
        dataset = SubsetImageFolder(...)
        dataset.begin()

        for epoch in range(num_epochs):
            for mb_data, mb_labels in dataset:
                train

            // Tell the dataset that an epoch has passed. The dataset will
            // add more classes if it is necessary.
            dataset.epoch()
    """
    def __init__(self, root, init_num_classes, steps,
                 transform=None, target_transform=None,
                 is_valid_file=None, verbose=False):
        """
        Arguments
        ---------
            init_num_classes: (int) Number of classes to begin training with

            steps: (list) Tuples of epoch gap and number of classes to add. For
                   example, [(60, 2), (30, 5)] indicates that one should wait
                   60 epochs then add 2 classes, then 30 more epochs at which
                   point 5 classes are added. Beyond this, 5 more classes are
                   added every 30 epochs until we have added all classes.

            verbose: (bool) Whether to print the number of classes and samples
                     used in the dataset at each call to `epoch()`.
        """
        super(SubsetImageFolder, self).__init__(
                root, transform=transform,
                target_transform=target_transform,
                is_valid_file=is_valid_file)

        # The parent class, `ImageFolder` maintains a member variable list
        # `samples`. This contains all samples in the dataset. We will create
        # a second list that contains the samples that are currently being
        # used in training, called `cur_samples`.

        self.init_num_classes = init_num_classes
        self.steps = steps
        self.verbose = verbose

        self.all_classes = sorted(list(set(x[1] for x in self.samples)))
        self.class_idx = min(self.init_num_classes, len(self.all_classes))

        self.cur_samples = [x for x in self.samples if x[1] in self.all_classes[:self.class_idx]]

    def _print_status(self):
        if self.verbose:
            print("NCLASSES:", self.class_idx, "/", len(self.all_classes))
            print("NSAMPLES:", len(self.cur_samples))

    def begin(self):
        """
        Initializes the `cur_samples` list based on `init_num_classes`.
        This method is necessary in case we shuffle the parent class's
        `samples` after construction.
        """
        self.cur_samples = [x for x in self.samples if x[1] in self.all_classes[:self.class_idx]]
        self.cur_epoch = 0
        self.next_step = 0
        self.next_step_epoch = self.cur_epoch + self.steps[self.next_step][0]
        self._print_status()

    def epoch(self):
        """
        This method should be called after each epoch of training. This method
        updates the current samples in our dataset, depending on the schedule.
        """
        self.cur_epoch += 1

        if self.cur_epoch == self.next_step_epoch:
            self.class_idx = min(self.class_idx + self.steps[self.next_step][1], len(self.all_classes))
            self.cur_samples = [x for x in self.samples if x[1] in self.all_classes[:self.class_idx]]

            self._print_status()

            # Stay on the last step for remaining epochs after final step
            if self.next_step < len(self.steps) - 1:
                self.next_step += 1

            self.next_step_epoch = self.cur_epoch + self.steps[self.next_step][0]

    def __getitem__(self, index):
        """
        This code is basically copied from ImageFolder, but samples from
        `cur_samples`

        Arguments
        ---------
            index (int): Index

        Returns
        -------
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.cur_samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.cur_samples)
