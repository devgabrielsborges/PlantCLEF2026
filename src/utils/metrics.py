import mlflow


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_dann_metrics(metrics, step, domain="train"):
    """
    Log DANN specific metrics to MLflow.
    Expected metrics: {'loss_class', 'loss_domain_s', 'loss_domain_t', 'acc_domain', 'acc_species'}
    """
    mlflow_metrics = {}
    for k, v in metrics.items():
        mlflow_metrics[f"{domain}/{k}"] = v
    mlflow.log_metrics(mlflow_metrics, step=step)
