import torch
import numpy as np
from tqdm import tqdm


class Trainer(object):

    @staticmethod
    def train(logger, metric, train_loader, model, criterion, optimizer, scheduler, epoch, max_epoch, device,
              best_pred):
        model.train()
        # metric.reset()

        # train_loss, train_miou, train_mpa, train_cpa = [], [], [], []
        train_loss = []
        with tqdm(train_loader) as train_bar:
            train_bar.set_description('Train Epoch[{:3d}/{:3d}]'.format(epoch, max_epoch))
            for i, sample in enumerate(train_bar):
                inputs, targets = sample['image'].to(device), sample['target'].to(device)
                scheduler(optimizer, i, epoch, best_pred)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                # 统计混淆矩阵
                # target = targets.cpu().numpy()
                # pred = np.argmax(outputs.data.cpu().numpy(), axis=1)

                # metric.add_batch(pred, target, [255])

                # 评估训练结果
                train_loss.append(loss.item())
                # train_mpa.append(metric.mean_pixel_accuracy())
                # train_cpa.append(metric.class_pixel_accuracy())
                # train_miou.append(metric.mean_intersection_over_union())
                train_bar.set_postfix({'batch_loss': '{0:1.5f}'.format(loss.item())})

            train_bar.update(len(train_loader))

            # 打印训练信息
            logger.info(f'Iter| [{epoch:3d}/{max_epoch}] Train Loss={np.nanmean(train_loss):.5f}')
            # logger.info(f'Iter| [{epoch:3d}/{max_epoch}] Train MIoU={np.nanmean(train_miou):.5f}')
            # logger.info(f'Iter| [{epoch:3d}/{max_epoch}] Train MPA={np.nanmean(train_mpa):.5f}')
            # logger.info(f'Iter| [{epoch:3d}/{max_epoch}] Train CPA={list(np.round(np.nanmean(train_cpa, 0), 5))}')

            # return np.nanmean(train_loss), np.nanmean(train_mpa), np.nanmean(train_miou), metric.confusion_matrix
            return np.nanmean(train_loss)

    torch.cuda.empty_cache()

    @staticmethod
    def valid(logger, metric, valid_loader, model, criterion, epoch_id, max_epoch, device):
        model.eval()
        metric.reset()

        eval_loss, eval_miou, eval_mpa, eval_cpa = [], [], [], []
        with tqdm(valid_loader) as eval_bar:
            eval_bar.set_description('Valid Epoch[{:3d}/{}]'.format(epoch_id, max_epoch))
            for i, sample in enumerate(eval_bar):
                inputs, targets = sample['image'].to(device), sample['target'].to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                loss = criterion(outputs, targets)
                eval_loss.append(loss.item())
                # 统计混淆矩阵
                target = targets.cpu().numpy()
                pred = np.argmax(outputs.data.cpu().numpy(), axis=1)

                # 评估训练结果
                # 由于dataloader是每次输出一个batch，因此需要等所有batch添加进来后，再进行计算
                metric.add_batch(pred, target, [0])

                # eval_mpa.append(metric.mean_pixel_accuracy())
                # eval_cpa.append(metric.class_pixel_accuracy())
                # eval_miou.append(metric.mean_intersection_over_union())

                eval_bar.set_postfix({'batch_loss': '{0:1.5f}'.format(loss.item())})

            cpa = metric.pixel_accuracy_class()
            mpa = metric.mean_pixel_accuracy()
            mIoU = metric.mean_intersection_over_union()

            eval_bar.update(len(valid_loader))

            # 打印训练信息
            # logger.info(f'Iter| [{epoch_id:3d}/{max_epoch}] Valid Loss={np.nanmean(eval_loss):.5f}')
            # logger.info(f'Iter| [{epoch_id:3d}/{max_epoch}] Valid MIoU={np.nanmean(eval_miou):.5f}')
            # logger.info(f'Iter| [{epoch_id:3d}/{max_epoch}] Valid MPA={np.nanmean(eval_mpa):.5f}')
            # logger.info(f'Iter| [{epoch_id:3d}/{max_epoch}] Valid CPA={list(np.round(np.nanmean(eval_cpa, 0), 5))}')

            logger.info(f'Iter| [{epoch_id:3d}/{max_epoch}] Valid Loss={np.nanmean(eval_loss):.5f}')
            logger.info(f'Iter| [{epoch_id:3d}/{max_epoch}] Valid MIoU={mIoU:.5f}')
            logger.info(f'Iter| [{epoch_id:3d}/{max_epoch}] Valid MPA={mpa:.5f}')
            logger.info(f'Iter| [{epoch_id:3d}/{max_epoch}] Valid CPA={list(np.round(cpa, 5))}')

            # return np.nanmean(eval_loss), np.nanmean(eval_mpa), np.nanmean(eval_miou), metric.confusion_matrix
            return np.nanmean(eval_loss), mpa, mIoU, metric.confusion_matrix

    torch.cuda.empty_cache()
