import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import get_config
from datasets.dataset_oct_png import OCT_PNG_dataset, RandomGenerator
from networks.vision_transformer import SwinUnet as ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/OCTLayers',
                    help='root dir for PNG data')
parser.add_argument('--dataset', type=str,
                    default='OCTLayers', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/OCTLayers', help='list dir')
parser.add_argument('--output_dir', type=str, required=True, help='output dir')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum iteration number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save png predictions during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as png')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs.",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, full: cache all data, part: shard dataset and cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument("--n_class", default=9, type=int)
parser.add_argument("--split_name", default="test", help="list split name")
parser.add_argument("--num_workers", default=1, type=int)

args = parser.parse_args()
if args.n_class != args.num_classes:
    args.n_class = args.num_classes
else:
    args.num_classes = args.n_class
config = get_config(args)


def dice_for_class(pred, target, class_index, eps=1e-5):
    pred_c = (pred == class_index).float()
    target_c = (target == class_index).float()
    denominator = pred_c.sum() + target_c.sum()
    if denominator.item() == 0:
        return None
    intersection = (pred_c * target_c).sum()
    return ((2.0 * intersection + eps) / (denominator + eps)).item()


def load_snapshot(args, model):
    candidates = [
        os.path.join(args.output_dir, 'best_model.pth'),
        os.path.join(args.output_dir, 'last_model.pth'),
        os.path.join(args.output_dir, 'epoch_' + str(args.max_epochs - 1) + '.pth'),
    ]
    snapshot = None
    for candidate in candidates:
        if os.path.exists(candidate):
            snapshot = candidate
            break
    if snapshot is None:
        raise FileNotFoundError("No checkpoint found. Tried: {}".format(", ".join(candidates)))
    msg = model.load_state_dict(torch.load(snapshot), strict=False)
    print("self trained swin unet", msg)
    return snapshot


def inference(args, model, test_save_path=None):
    db_test = OCT_PNG_dataset(
        base_dir=args.root_path,
        split=args.split_name,
        list_dir=args.list_dir,
        transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size], augment=False)])
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()

    class_dice_sum = np.zeros(args.num_classes - 1, dtype=np.float64)
    class_dice_count = np.zeros(args.num_classes - 1, dtype=np.int64)

    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
        image, label = sampled_batch["image"].cuda(), sampled_batch["label"].cuda()
        case_name = sampled_batch['case_name'][0]
        with torch.no_grad():
            logits = model(image)
            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)

        case_dice = []
        for class_index in range(1, args.num_classes):
            dice_value = dice_for_class(pred, label, class_index)
            case_dice.append(dice_value)
            if dice_value is not None:
                class_dice_sum[class_index - 1] += dice_value
                class_dice_count[class_index - 1] += 1

        case_mean = np.mean([d for d in case_dice if d is not None]) if any(d is not None for d in case_dice) else 0.0
        logging.info('idx %d case %s mean_dice(1-8) %f' % (i_batch, case_name, case_mean))

        if test_save_path is not None:
            pred_np = pred.squeeze(0).cpu().numpy().astype(np.uint8)
            Image.fromarray(pred_np).save(os.path.join(test_save_path, case_name + '.png'))

    class_dice = np.divide(
        class_dice_sum,
        np.maximum(class_dice_count, 1),
        out=np.zeros_like(class_dice_sum),
        where=np.maximum(class_dice_count, 1) > 0
    )
    for i in range(1, args.num_classes):
        logging.info('Mean class %d dice %f' % (i, class_dice[i - 1]))
    performance = np.mean(class_dice) if len(class_dice) > 0 else 0.0
    logging.info('Testing performance in model: mean_dice(1-8): %f' % performance)
    return "Testing Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    snapshot = load_snapshot(args, net)
    snapshot_name = snapshot.split('/')[-1]

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions_png")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)
