import os
import time

import cv2
import numpy as np
from mindspore import Tensor, save_checkpoint
from mindspore.train.callback import Callback


class StepLossTimeMonitor(Callback):
    def __init__(self, logger, opt, pp_fun=None, save_max_ckpt=True):
        super(StepLossTimeMonitor, self).__init__()
        self._per_print_times = opt.get("print_freq")
        self.logger = logger
        self.step_time = 0
        self.opt = opt
        self.pp_fun = pp_fun
        self.save_max_ckpt = save_max_ckpt
        self.save_ckpt_metric = -float("inf") if save_max_ckpt else float("inf")
        self.best_epoch = 0
        self.val_num = 0

    def on_train_step_begin(self, run_context):
        if not self.step_time:
            self.step_time = time.time()
        self.step_losses = []

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()

        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(
                loss[0].asnumpy(), np.ndarray
            ):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError(
                "epoch: {} step: {}. Invalid loss, terminating training.".format(
                    cb_params.cur_epoch_num, cur_step_in_epoch
                )
            )
        self.step_losses.append(loss)
        self.losses.append(loss)

        if (
            self._per_print_times != 0
            and cb_params.cur_step_num % self._per_print_times == 0
        ):

            batch_size = cb_params.train_dataset.get_batch_size()

            cost_time = time.time() - self.step_time
            self.step_time = 0

            cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
            total_step = cb_params.batch_num

            epoch = cb_params.cur_epoch_num
            lr = cb_params.optimizer.get_lr()

            # TEST
            self.logger.info(
                "[epoch {}] [{}/{}] loss: {:.4f}, lr: {}, cost: {:.2f}s".format(
                    epoch,
                    cur_step_in_epoch,
                    total_step,
                    np.mean(self.step_losses),
                    lr,
                    cost_time,
                )
            )

    def on_train_epoch_begin(self, run_context):
        self.epoch_start = time.time()
        self.losses = []

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()

        if self.opt.get("save_ckpt_freq"):
            if cb_params.cur_epoch_num % self.opt["save_ckpt_freq"] == 0:
                save_checkpoint(
                    cb_params.train_network,
                    os.path.join(
                        self.logger.log_file_path,
                        "ckpt",
                        f'{self.opt["name"]}_{cb_params.cur_epoch_num}_{cb_params.cur_epoch_num*cb_params.batch_num}.ckpt',
                    ),
                )

        epoch_cost = time.time() - self.epoch_start
        self.logger.info(
            "[epoch {}] [end] avg loss:{:.4f}, total cost: {:.3f}s".format(
                cb_params.cur_epoch_num, np.mean(self.losses), epoch_cost
            )
        )
        self.logger.info("===============================================")

    def on_eval_begin(self, run_context):
        cb_params = run_context.original_args()

        self.save_img_id = 0
        if self.opt['phase'] == 'train':
            if self.opt['eval_freq']:
                os.makedirs(
                    os.path.join(
                        self.logger.log_file_path, "img", str(cb_params.cur_epoch_num)
                    )
                )
        else:
            os.makedirs(
                os.path.join(self.logger.log_file_path, "img", f'val_{self.val_num}')
            )

    def on_eval_step_end(self, run_context):
        cb_params = run_context.original_args()
        if self.opt['phase'] == 'train' and self.opt['eval_freq']:
            if self.opt.get("save_img") and self.save_img_id < self.opt["save_img"]:
                assert self.pp_fun
                out = cb_params.net_outputs[1]
                out = self.pp_fun(out)
                cv2.imwrite(
                    os.path.join(
                        self.logger.log_file_path,
                        "img",
                        str(cb_params.cur_epoch_num),
                        str(self.save_img_id) + ".jpg",
                    ),
                    out,
                )
        elif self.opt['phase'] == 'eval':
            if self.opt.get("save_img") and self.save_img_id < self.opt["save_img"]:
                assert self.pp_fun
                out = cb_params.net_outputs[0]
                out = self.pp_fun(out)
                cv2.imwrite(
                    os.path.join(
                        self.logger.log_file_path,
                        "img",
                        f'val_{self.val_num}',
                        str(self.save_img_id) + ".jpg",
                    ),
                    out,
                )
        self.save_img_id += 1

    def on_eval_end(self, run_context):
        self.val_num += 1
        cb_params = run_context.original_args()
        if self.opt['phase'] == 'train' and self.opt['eval_freq']:
            output = f"[epoch {cb_params.cur_epoch_num}] [val] "
            for m in cb_params.metrics:
                output += "{}: {}, ".format(m, cb_params.metrics[m])
            m = list(cb_params.metrics.keys())[0]
            if (
                self.save_ckpt_metric < cb_params.metrics[m]
                if self.save_max_ckpt
                else self.save_ckpt_metric > cb_params.metrics[m]
            ):
                self.best_epoch = cb_params.cur_epoch_num
                self.save_ckpt_metric = cb_params.metrics[m]
                if self.opt.get("save_best_ckpt"):

                    save_checkpoint(
                        cb_params.train_network,
                        os.path.join(
                            self.logger.log_file_path,
                            "ckpt",
                            f'{self.opt["name"]}_best.ckpt',
                        ),
                    )
                    self.logger.info(
                        f"[epoch {cb_params.cur_epoch_num}] [val] save the best ckpt."
                    )
            output += f'best epoch: {self.best_epoch}'
            self.logger.info(output)

        if self.opt['phase'] == 'eval':
            output = f"[eval] [val_{self.val_num}] "
            for m in cb_params.metrics:
                output += "{}: {}, ".format(m, cb_params.metrics[m])
            self.logger.info(output)
