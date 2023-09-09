from typing import Any, Tuple, List
import glob
import torch
import torch.nn as nn
import torch.nn.init as init
from lightning import LightningModule
from torchmetrics import MinMetric, MaxMetric, MeanMetric
from torchmetrics.regression.mse import MeanSquaredError
from torchmetrics.regression.mae import MeanAbsoluteError
from torchmetrics.classification.accuracy import BinaryAccuracy
from src.models.leak_detection_distance_module import LeakDetectionModule


class LeakDetectionTransferModule(LightningModule):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        checkpoint_file: str,
        gamma: float,
        from_scratch: bool = False,
        use_representation: bool = True,
        use_material: bool = True,
        freezed_layers: List = [1, 2]
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        if from_scratch:
            assert not use_representation

        print(glob.glob(
                f"logs/train/runs/{checkpoint_file}/checkpoints/epoch_*.ckpt")[0])

        self.net = LeakDetectionModule.load_from_checkpoint(glob.glob(
                f"logs/train/runs/{checkpoint_file}/checkpoints/epoch_*.ckpt")[0])

        if from_scratch:
            self.net.feature_extractor.apply(weight_init)
            self.net.outlayer.apply(weight_init)
        else:
            # self.net.outlayer.apply(weight_init)
            if len(freezed_layers) > 0:
                for name, layer in self.net.outlayer.named_children():
                    for freezed_layer in freezed_layers:
                        if str(freezed_layer) in name:
                            for param in layer.parameters():
                                param.requires_grad = False
                        else:
                            layer.apply(weight_init)

                # for name, param in self.net.outlayer.named_parameters():
                #     for layer in freezed_layers:
                #         if str(layer) in name and param.requires_grad:
                #             param.requires_grad = False

            else:
                self.net.outlayer.apply(weight_init)

        # loss function
        self.criterion_r = torch.nn.MSELoss()
        self.criterion_c = torch.nn.BCEWithLogitsLoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        # metric objects for calculating and averaging metric across batches
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

        # metric objects for calculating and averaging metric across batches
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.hparams.use_representation:
            self.net.feature_extractor.eval()
            with torch.no_grad():
                _, representations, _ = self.net.feature_extractor(x)
        else:
            _, representations, _ = self.net.feature_extractor(x)
        if self.hparams.use_material:
            representations = torch.concat((representations, y[:, -1].view(-1, 1)), dim=1)
        return self.net.outlayer(representations)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        assert x.dim() == 4 and y.dim() == 2
        cls_out, dis_out = self.forward(x, y)
        cls_loss = self.criterion_c(cls_out, y[:, 0].view(-1, 1))
        dis_loss = self.criterion_r(dis_out * y[:, 0].view(-1, 1), (y[:, 1] * y[:, 0]).view(-1, 1))
        loss = cls_loss * self.hparams.gamma + dis_loss
        leak_prediction = cls_out > 0

        return loss, leak_prediction, dis_out, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, leak_prediction, dis_out, y = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(leak_prediction, y[:, 0].view(-1, 1))
        self.train_mse(dis_out[y[:, 0]==1, :], y[y[:, 0]==1, 1].view(-1, 1))
        self.train_mae(dis_out[y[:, 0]==1, :], y[y[:, 0]==1, 1].view(-1, 1))
        self.log("train/loss", self.train_loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mse", self.train_mse,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mae", self.train_mae,
                 on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, leak_prediction, dis_out, y = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(leak_prediction, y[:, 0].view(-1, 1))
        self.val_mse(dis_out[y[:, 0]==1, :], y[y[:, 0]==1, 1].view(-1, 1))
        self.val_mae(dis_out[y[:, 0]==1, :], y[y[:, 0]==1, 1].view(-1, 1))
        self.log("val/loss", self.val_loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mse", self.val_mse,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae", self.val_mae,
                 on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_best(loss)  # update best so far val loss
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, leak_prediction, dis_out, y = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(leak_prediction, y[:, 0].view(-1, 1))
        self.test_mse(dis_out[y[:, 0]==1, :], y[y[:, 0]==1, 1].view(-1, 1))
        self.test_mae(dis_out[y[:, 0]==1, :], y[y[:, 0]==1, 1].view(-1, 1))
        self.log("test/loss", self.test_loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mse", self.test_mse,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mae", self.test_mae,
                 on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=filter(lambda p: p.requires_grad, self.parameters()))
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
 

if __name__ == "__main__":
    _ = LeakDetectionModule(None, None, None)
