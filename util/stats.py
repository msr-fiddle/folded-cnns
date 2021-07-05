import torch


def get_top_k(k, outputs, compare):
    """
    Returns the number of elements in ``compare`` for which the corresponding
    entry in ``outputs`` has the value of ``compare`` in its top-k.
    """
    rep_compare = compare.unsqueeze(1).repeat(1, k)
    return torch.sum((outputs.topk(k)[1] == rep_compare)).item()


class StatsTracker(object):
    """
    Container for tracking the statistics associated with an epoch. For each of
    a training and validation pass, a new StatsTracker should be instantiated.
    The common use pattern of the class looks as follows:
        for e in range(num_epochs):
            stats = StatsTracker()
            # Add some loss stats
            stats.update_loss(loss)
            # Add accuracy metrics
            stats.update_accuracies(outputs_output, labels, true_labels)
            # Get current average stats
            a, b, c = stats.averages()
    """

    def __init__(self):
        self.loss = 0.
        self.num_overall_match = 0

        self.acc_keys = []
        self.top_k_vals = [1, 2, 5, 10]
        for val in self.top_k_vals:
            self.acc_keys.append("top{}".format(val))

        self.acc_map = {}
        for k in self.acc_keys:
            self.acc_map[k] = 0

        # Hold different counters for the number of loss and accuracy attempts.
        # Losses are added in the unit of the average for a minibatch, while
        # accuracy metrics are added for individual samples.
        self.num_loss_attempts = 0
        self.num_match_attempts = 0

        self.running_top1 = 0
        self.running_top5 = 0

    def averages(self):
        """
        Returns average loss and accuracy since this ``StatsTracker`` was
        instantiated.
        """
        avg_loss = self.loss / self.num_loss_attempts

        for k in self.acc_map:
            self.acc_map[k] /= self.num_match_attempts

        return avg_loss, self.acc_map

    def running_averages(self):
        """
        Returns running average loss, top-1 overall accuracy, and top-5
        overall accuracy.
        """
        avg_loss = self.loss / self.num_loss_attempts
        top1 = self.running_top1 / self.num_match_attempts
        top5 = self.running_top5 / self.num_match_attempts
        return avg_loss, top1, top5

    def update_accuracies(self, outputs, true_labels):
        """
        Calculates the number of outputs outputs that match (1) the outputs
        from the base model and (2) the true labels associated with the outputs
        sample. These results are maintained for later aggregate statistics.
        """
        self.num_match_attempts += outputs.size(0)
        max_outputs = torch.max(outputs, dim=1)[1]

        top1_correct = torch.sum((max_outputs == true_labels)).item()
        self.acc_map["top1"] += top1_correct
        self.running_top1 += top1_correct

        # Ignore first of top_k_vals because we already covered it above.
        for k in self.top_k_vals[1:]:
            if outputs.size(-1) < k:
                break

            overall_correct = get_top_k(k, outputs, true_labels)
            self.acc_map["top{}".format(k)] += overall_correct
            if k == 5:
                self.running_top5 += overall_correct

    def update_loss(self, loss):
        """
        Adds `loss` to the current aggregate loss for this epoch.
        """
        self.loss += loss
        self.num_loss_attempts += 1
