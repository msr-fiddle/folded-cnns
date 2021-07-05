import torch
import torch.utils.data as data


from util.util import try_cuda, construct


class DistilledDataset:
    """
    A dataset that returns logits from a teacher model as labels rather than
    the original class labels
    """
    def __init__(self, labelgen_dataset, labelgen_model, labelgen_model_file,
                 base_dataset, num_classes=10):
        """
        Arguments
        ---------
            labelgen_dataset (Dataset): dataset to draw from for generating
                                        labels from the teacher model

            labelgen_model (dict): description of teacher model

            labelgen_model_file (str): path to teacher model weights

            base_dataset (Dataset): dataset to be used when providing new
                                    samples for training. The difference
                                    between this and `labelgen_dataset`
                                    is that `labelgen_dataset` generally
                                    is not augmented, but `base_dataset`
                                    can be for training.

            num_classes (int): number of classes in the dataset
        """
        self.base_dataset = base_dataset
        self.labelgen_dataset = labelgen_dataset

        self.labelgen_model = construct(labelgen_model)
        self.labelgen_model.load_state_dict(torch.load(labelgen_model_file))

        self.labelgen_model = try_cuda(self.labelgen_model)
        self.labelgen_model.eval()

        self.labelgen_outputs = torch.zeros(len(labelgen_dataset), num_classes)
        self.labelgen_outputs = try_cuda(self.labelgen_outputs)

        # Dataset used for getting labels from the labelgen model to be used in
        # training under distillation.
        label_dataloader = data.DataLoader(self.labelgen_dataset, batch_size=32,
                                           shuffle=False)

        num_correct = 0
        num_tried = 0
        cur_idx = 0
        with torch.no_grad():
            for mb_data, mb_labels in label_dataloader:
                mb_data = try_cuda(mb_data)
                mb_labels = try_cuda(mb_labels)
                outs = self.labelgen_model(mb_data)
                self.labelgen_outputs[cur_idx:(cur_idx + mb_data.size(0))] = outs.cpu()
                cur_idx += mb_data.size(0)
                max_outputs = torch.max(outs, dim=1)[1]
                num_correct += torch.sum((max_outputs == mb_labels).float()).item()
                num_tried += mb_data.size(0)

        print("Accuracy of base model: {} / {} = {}".format(num_correct, num_tried, (num_correct / num_tried)))

        # Remove labelgen model from GPU. Not sure if this is necessary.
        self.labelgen_model = self.labelgen_model.cpu()

    def __getitem__(self, idx):
        """
        Arguments
        ---------
            idx (int): index to get

        Returns
        -------
            data (Tensor): the input sample

            true_label (int): the class to which this sample belongs

            distillation_label (Tensor): the output logits for this sample that
                                         result from the teacher model.
        """
        data, true_label = self.base_dataset[idx]
        distillation_label = self.labelgen_outputs[idx]
        return data, true_label, distillation_label

    def __len__(self):
        return self.labelgen_outputs.size(0)
