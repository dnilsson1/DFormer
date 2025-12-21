import argparse
import datetime
import os
import pprint
import random
import time
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from val_mm import evaluate, evaluate_msf

from models.builder import EncoderDecoder as segmodel
from utils.dataloader.dataloader import get_train_loader, get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.engine import Engine
from utils.engine.logger import get_logger
from utils.init_func import configure_optimizers, group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.pyt_utils import all_reduce_tensor

# from eval import evaluate_mid


parser = argparse.ArgumentParser()
parser.add_argument("--config", help="train config file path")
parser.add_argument("--gpus", default=2, type=int, help="used gpu number")
# parser.add_argument('-d', '--devices', default='0,1', type=str)
parser.add_argument("-v", "--verbose", default=False, action="store_true")
parser.add_argument("--epochs", default=0)
parser.add_argument("--show_image", "-s", default=False, action="store_true")
parser.add_argument("--save_path", default=None)
parser.add_argument("--checkpoint_dir")
parser.add_argument("--continue_fpath")
parser.add_argument("--sliding", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--compile", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--compile_mode", default="default")
parser.add_argument("--syncbn", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--mst", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--val_amp", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--pad_SUNRGBD", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--use_seed", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("-tb", "--tensorboard", default=True, action=argparse.BooleanOptionalAction, help="Enable TensorBoard logging")
parser.add_argument("--local-rank", default=0)
# parser.add_argument('--save_path', '-p', default=None)

# os.environ['MASTER_PORT'] = '169710'
torch.set_float32_matmul_precision("high")
import torch._dynamo

torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.automatic_dynamic_shapes = False


def is_eval(epoch, config):
    eval_iter = getattr(config, 'eval_iter', 10)  # Default to every 10 epochs
    return epoch == 1 or epoch % eval_iter == 0


class gpu_timer:
    def __init__(self, beta=0.6) -> None:
        self.start_time = None
        self.stop_time = None
        self.mean_time = None
        self.beta = beta
        self.first_call = True

    def start(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def stop(self):
        if self.start_time is None:
            print("Use start() before stop(). ")
        torch.cuda.synchronize()
        self.stop_time = time.perf_counter()
        elapsed = self.stop_time - self.start_time
        self.start_time = None
        if self.first_call:
            self.mean_time = elapsed
            self.first_call = False
        else:
            self.mean_time = self.beta * self.mean_time + (1 - self.beta) * elapsed


def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True  # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True, warn_only=True)


with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    config = getattr(import_module(args.config), "C")
    logger = get_logger(config.log_dir, config.log_file, rank=engine.local_rank)
    # check if pad_SUNRGBD is used correctly
    if args.pad_SUNRGBD and config.dataset_name != "SUNRGBD":
        args.pad_SUNRGBD = False
        logger.warning("pad_SUNRGBD is only used for SUNRGBD dataset")
    if (args.pad_SUNRGBD) and (not config.backbone.startswith("DFormerv2")):
        raise ValueError("DFormerv1 is not recommended with pad_SUNRGBD")
    if (not args.pad_SUNRGBD) and config.backbone.startswith("DFormerv2") and config.dataset_name == "SUNRGBD":
        raise ValueError("DFormerv2 is not recommended without pad_SUNRGBD")
    config.pad = args.pad_SUNRGBD
    if args.use_seed:
        set_seed(config.seed)
        logger.info(f"set seed {config.seed}")
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        logger.info("use random seed")

    # assert not (args.compile and args.syncbn), "syncbn is not supported in compile mode"
    if not args.compile and args.compile_mode != "default":
        logger.warning("compile_mode is only valid when compile is enabled, ignoring compile_mode")

    train_loader, train_sampler = get_train_loader(engine, RGBXDataset, config)

    if args.gpus == 2:
        if args.mst and args.compile and args.compile_mode == "reduce-overhead":
            val_dl_factor = 0.25
        elif args.mst and not args.val_amp:
            val_dl_factor = 1.5
        elif args.mst and args.val_amp:
            val_dl_factor = 1.3
        else:
            val_dl_factor = 2
    elif args.gpus == 4:
        if args.mst and args.compile and args.compile_mode == "reduce-overhead":
            val_dl_factor = 0.25
        elif args.mst and not args.val_amp:
            val_dl_factor = 1.5
        elif args.mst and args.val_amp:
            val_dl_factor = 0.6
        else:
            val_dl_factor = 2
    else:
        val_dl_factor = 1.5

    val_dl_factor = 1  # TODO: remove this line
    val_loader, val_sampler = get_val_loader(
        engine,
        RGBXDataset,
        config,
        val_batch_size=int(config.batch_size * val_dl_factor) if config.dataset_name != "SUNRGBD" else int(args.gpus),
    )
    logger.info(f"val dataset len:{len(val_loader) * int(args.gpus)}")

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        if args.tensorboard:
            tb_dir = config.tb_dir + "/{}".format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
            generate_tb_dir = config.tb_dir + "/tb"
            tb = SummaryWriter(log_dir=tb_dir)
            engine.link_tb(tb_dir, generate_tb_dir)
            logger.info(f"TensorBoard enabled: {tb_dir}")
        else:
            tb = None
            logger.info("TensorBoard disabled")
        pp = pprint.PrettyPrinter(indent=4)
        logger.info("config: \n" + pp.pformat(config))

    logger.info("args parsed:")
    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))

    # Configure loss function based on config settings
    # Supports both CrossEntropyLoss and FocalLoss for class imbalance handling
    use_focal_loss = getattr(config, 'use_focal_loss', False)
    
    if config.use_focal_loss:
        # Use Focal Loss for automatic hard example mining
        from models.losses.segmentation_focal_loss import SegmentationFocalLoss
        
        focal_gamma = getattr(config, 'focal_gamma', 2.0)
        focal_alpha = getattr(config, 'focal_alpha', 0.25)
        # Don't use class_weights with Focal Loss unless explicitly set and not None
        class_weights = getattr(config, 'class_weights', None)
        
        criterion = SegmentationFocalLoss(
            num_classes=config.num_classes,
            gamma=focal_gamma,
            alpha=focal_alpha,
            class_weights=class_weights,  # Will be None - focal loss handles imbalance
            ignore_index=config.background,
            reduction='none'  # Return per-pixel loss, builder.py will handle masking and mean
        )
        logger.info(f"Using Focal Loss: gamma={focal_gamma}, alpha={focal_alpha}, "
                   f"class_weights={'enabled' if class_weights is not None else 'disabled (focal loss handles imbalance)'}")
    else:
        # Use standard CrossEntropyLoss with optional class weights
        class_weights = None
        if hasattr(config, 'class_weights') and config.class_weights is not None:
            import torch
            class_weights = torch.FloatTensor(config.class_weights).cuda()
            logger.info(f"Using class weights for {len(class_weights)} classes")
        
        criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=config.background, weight=class_weights)

    if args.syncbn:
        BatchNorm2d = nn.SyncBatchNorm
        logger.info("using syncbn")
    else:
        BatchNorm2d = nn.BatchNorm2d
        logger.info("using regular bn")

    model = segmodel(
        cfg=config,
        criterion=criterion,
        norm_layer=BatchNorm2d,
        syncbn=args.syncbn,
    )
    # weight=torch.load('checkpoints/NYUv2_DFormer_Large.pth')['model']
    # w_list=list(weight.keys())
    # # for k in w_list:
    # #     weight[k[7:]] = weight[k]
    # print('load model')
    # model.load_state_dict(weight)

    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr

    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
    # params_list = configure_optimizers(model, base_lr, config.weight_decay)

    if config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params_list,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "SGDM":
        optimizer = torch.optim.SGD(
            params_list,
            lr=base_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        raise NotImplementedError

    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(
        base_lr,
        config.lr_power,
        total_iteration,
        config.niters_per_epoch * config.warm_up_epoch,
    )
    if engine.distributed:
        logger.info(".............distributed training.............")
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(
                model,
                device_ids=[engine.local_rank],
                output_device=engine.local_rank,
                find_unused_parameters=False,
            )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()

    logger.info("begin trainning:")
    data_setting = {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "x_single_channel": config.x_is_single_channel,
        "class_names": config.class_names,
        "train_source": config.train_source,
        "eval_source": config.eval_source,
    }
    # val_pre = ValPre()
    # val_dataset = RGBXDataset(data_setting, 'val', val_pre)
    # test_loader, test_sampler = get_test_loader(engine, RGBXDataset,config)
    all_dev = [0]
    # segmentor = SegEvaluator(val_dataset, config.num_classes, config.norm_mean,
    #                                 config.norm_std, None,
    #                                 config.eval_scale_array, config.eval_flip,
    #                                 all_dev, config,args.verbose, args.save_path,args.show_image)
    uncompiled_model = model
    if args.compile:
        compiled_model = torch.compile(model, backend="inductor", mode=args.compile_mode)
    else:
        compiled_model = model
    miou, best_miou = 0.0, 0.0
    train_timer = gpu_timer()
    eval_timer = gpu_timer()

    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    for epoch in range(engine.state.epoch, config.nepochs + 1):
        model = compiled_model
        model.train()
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, "refresh_epoch"):
            train_loader.dataset.refresh_epoch()
        # bar_format = "{desc}[{elapsed}<{remaining},{rate_fmt}]"
        # pbar = tqdm(
        #     range(config.niters_per_epoch),
        #     file=sys.stdout,
        #     bar_format=bar_format,
        #     # range(5),
        #     # file=sys.stdout,
        #     # bar_format=bar_format,
        # )
        dataloader = iter(train_loader)

        sum_loss = 0
        i = 0
        train_timer.start()
        for idx in range(config.niters_per_epoch):
            engine.update_iteration(epoch, idx)

            # minibatch = dataloader.next()
            minibatch = next(dataloader)
            imgs = minibatch["data"]
            gts = minibatch["label"]
            modal_xs = minibatch["modal_x"]

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)

            if args.amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(imgs, modal_xs, gts)
            else:
                loss = model(imgs, modal_xs, gts)
            
            # Check for NaN loss and log details
            if torch.isnan(loss):
                logger.error(f"NaN loss detected at epoch {epoch}, iter {idx}")
                logger.error(f"  Images: min={imgs.min():.4f}, max={imgs.max():.4f}, mean={imgs.mean():.4f}")
                logger.error(f"  Depth: min={modal_xs.min():.4f}, max={modal_xs.max():.4f}, mean={modal_xs.mean():.4f}")
                logger.error(f"  Labels: min={gts.min()}, max={gts.max()}, unique={torch.unique(gts).tolist()}")
                # Skip this batch
                continue

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)

            if args.amp:
                # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)
                # Updates the scale for next iteration.
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # TODO: check if set_to_none=True impact the performance
            else:
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if epoch == 1:
                    for name, param in model.named_parameters():
                        if param.grad is None:
                            logger.warning(f"{name} has no grad, please check")

            current_idx = (epoch - 1) * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]["lr"] = lr

            if engine.distributed:
                sum_loss += reduce_loss.item()
                print_str = (
                    "Epoch {}/{}".format(epoch, config.nepochs)
                    + " Iter {}/{}:".format(idx + 1, config.niters_per_epoch)
                    + " lr=%.4e" % lr
                    + " loss=%.4f total_loss=%.4f" % (reduce_loss.item(), (sum_loss / (idx + 1)))
                )

            else:
                sum_loss += loss.item()
                print_str = (
                    f"Epoch {epoch}/{config.nepochs} "
                    + f"Iter {idx + 1}/{config.niters_per_epoch}: "
                    + f"lr={lr:.4e} loss={loss:.4f} total_loss={(sum_loss / (idx + 1)):.4f}"
                )

            if ((idx + 1) % int((config.niters_per_epoch) * 0.025) == 0 or idx == 0) and (
                (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed)
            ):
                print(print_str)

            del loss
            # pbar.set_description(print_str, refresh=False)
        logger.info(print_str)
        train_timer.stop()

        if ((engine.distributed and (engine.local_rank == 0)) or (not engine.distributed)) and args.tensorboard and tb is not None:
            tb.add_scalar("train_loss", float(sum_loss / config.niters_per_epoch), epoch)
            tb.add_scalar("learning_rate", float(lr), epoch)

        if is_eval(epoch, config):
            eval_timer.start()
            torch.cuda.empty_cache()
            eval_success = False  # Track if evaluation completed successfully
            miou = 0.0  # Initialize miou in case eval fails
            try:
                # if args.compile and args.mst and (not args.sliding):
                #     model = uncompiled_model
                # TODO: FIX this
                if engine.distributed:
                    with torch.no_grad():
                        model.eval()
                        device = torch.device("cuda")
                        if args.val_amp:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                if args.mst:
                                    all_metrics = evaluate_msf(
                                        model,
                                        val_loader,
                                        config,
                                        device,
                                        [0.5, 0.75, 1.0, 1.25, 1.5],
                                        True,
                                        engine,
                                        sliding=args.sliding,
                                    )
                                else:
                                    all_metrics = evaluate(
                                        model,
                                        val_loader,
                                        config,
                                        device,
                                        engine,
                                        sliding=args.sliding,
                                    )
                        else:
                            if args.mst:
                                all_metrics = evaluate_msf(
                                    model,
                                    val_loader,
                                    config,
                                    device,
                                    [0.5, 0.75, 1.0, 1.25, 1.5],
                                    True,
                                    engine,
                                    sliding=args.sliding,
                                )
                            else:
                                all_metrics = evaluate(
                                    model,
                                    val_loader,
                                    config,
                                    device,
                                    engine,
                                    sliding=args.sliding,
                                )
                        if engine.local_rank == 0:
                            metric = all_metrics[0]
                            for other_metric in all_metrics[1:]:
                                metric.update_hist(other_metric.hist)
                            ious, miou = metric.compute_iou()
                            acc, macc = metric.compute_pixel_acc()
                            f1, mf1 = metric.compute_f1()
                            if miou > best_miou:
                                best_miou = miou
                                engine.save_and_link_checkpoint(
                                    config.log_dir,
                                    config.log_dir,
                                    config.log_dir_link,
                                    infor="_miou_" + str(miou),
                                    metric=miou,
                                )
                            print("miou", miou, "best", best_miou)
                            # Log validation metrics to TensorBoard
                            if args.tensorboard and tb is not None:
                                tb.add_scalar("val_miou", miou, epoch)
                                tb.add_scalar("val_best_miou", best_miou, epoch)
                                tb.add_scalar("val_pixel_acc", acc, epoch)
                                tb.add_scalar("val_mean_acc", macc, epoch)
                                tb.add_scalar("val_f1", f1, epoch)
                                tb.add_scalar("val_mean_f1", mf1, epoch)
                            eval_success = True  # Mark eval as successful (distributed)
                elif not engine.distributed:
                    with torch.no_grad():
                        model.eval()
                        device = torch.device("cuda")
                        if args.val_amp:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                if args.mst:
                                    metric = evaluate_msf(
                                        model,
                                        val_loader,
                                        config,
                                        device,
                                        [0.5, 0.75, 1.0, 1.25, 1.5],
                                        True,
                                        engine,
                                        sliding=args.sliding,
                                    )
                                else:
                                    metric = evaluate(
                                        model,
                                        val_loader,
                                        config,
                                        device,
                                        engine,
                                        sliding=args.sliding,
                                    )
                        else:
                            if args.mst:
                                metric = evaluate_msf(
                                    model,
                                    val_loader,
                                    config,
                                    device,
                                    [0.5, 0.75, 1.0, 1.25, 1.5],
                                    True,
                                    engine,
                                    sliding=args.sliding,
                                )
                            else:
                                metric = evaluate(
                                    model,
                                    val_loader,
                                    config,
                                    device,
                                    engine,
                                    sliding=args.sliding,
                                )
                        ious, miou = metric.compute_iou()
                        acc, macc = metric.compute_pixel_acc()
                        f1, mf1 = metric.compute_f1()
                        
                        # Log per-class IoU for detailed analysis
                        logger.info(f"\n{'='*80}")
                        logger.info(f"Epoch {epoch} - Per-Class IoU Breakdown:")
                        logger.info(f"{'='*80}")
                        for idx, (class_name, iou) in enumerate(zip(config.class_names, ious)):
                            logger.info(f"  Class {idx:2d} - {class_name:25s}: {iou:6.2f}%")
                        logger.info(f"{'='*80}")
                        logger.info(f"  Mean IoU: {miou:.2f}%")
                        logger.info(f"  Mean Acc: {macc:.2f}%")
                        logger.info(f"  Mean F1:  {mf1:.2f}%")
                        logger.info(f"{'='*80}\n")
                        
                    # Outside with block but still inside elif
                    if miou > best_miou:
                        best_miou = miou
                        engine.save_and_link_checkpoint(
                            config.log_dir,
                            config.log_dir,
                            config.log_dir_link,
                            infor="_miou_" + str(miou),
                            metric=miou,
                        )
                    print("miou", miou, "best", best_miou)
                    # Log validation metrics to TensorBoard (non-distributed)
                    if args.tensorboard and tb is not None:
                        tb.add_scalar("val_miou", float(miou), epoch)
                        tb.add_scalar("val_best_miou", float(best_miou), epoch)
                        tb.add_scalar("val_mean_acc", float(macc), epoch)
                        tb.add_scalar("val_mean_f1", float(mf1), epoch)
                        
                        # Log per-class IoU to TensorBoard
                        for idx, (class_name, iou) in enumerate(zip(config.class_names, ious)):
                            tb.add_scalar(f"val_iou/{class_name}", float(iou), epoch)
                    eval_success = True  # Mark eval as successful
            except Exception as e:
                logger.error(f"Epoch {epoch} evaluation failed: {str(e)}")
                logger.error("Continuing training without eval metrics for this epoch...")
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            if eval_success:
                logger.info(f"Epoch {epoch} validation result: mIoU {miou:.2f}%, best mIoU {best_miou:.2f}%")
            eval_timer.stop()

        eval_count = 0
        for i in range(engine.state.epoch + 1, config.nepochs + 1):
            if is_eval(i, config):
                eval_count += 1
        # Safely handle None mean_time values
        train_mean = train_timer.mean_time if train_timer.mean_time is not None else 0.0
        eval_mean = eval_timer.mean_time if eval_timer.mean_time is not None else 0.0
        left_time = train_mean * (config.nepochs - engine.state.epoch) + eval_mean * eval_count
        eta = (datetime.datetime.now() + datetime.timedelta(seconds=left_time)).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"Avg train time: {train_mean:.2f}s, avg eval time: {eval_mean:.2f}s, left eval count: {eval_count}, ETA: {eta}"
        )
        
        # Clear GPU cache and cleanup workers after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection to cleanup any hanging references
        import gc
        gc.collect()
        
        # Additional cleanup - delete optimizer state temporarily and recreate
        # This helps prevent memory accumulation between epochs
        if (epoch + 1) % 5 == 0:  # Every 5 epochs
            torch.cuda.empty_cache()
            gc.collect()
