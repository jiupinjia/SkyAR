import argparse
import numpy as np
import matplotlib.pyplot as plt
import utils
from matting import*

# settings
parser = argparse.ArgumentParser(description='ZZX TRAIN SEGMENTATION')
parser.add_argument('--dataset', type=str, default='cvprw2020-ade20K-defg', metavar='str',
                    help='dataset: cvprw2020-ade20K-defg or ... (default: cvprw2020-ade20K-defg)')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--in_size', type=int, default=384, metavar='N',
                    help='input image size for training (default: 256)')
parser.add_argument('--print_models', action='store_true', default=False,
                    help='visualize and print networks')
parser.add_argument('--net_G', type=str, default='coord_resnet50', metavar='str',
                    help='net_G: resnet50 or coord_resnet50 (default: coord_resnet50 )')
parser.add_argument('--checkpoint_dir', type=str, default=r'./checkpoints', metavar='str',
                    help='dir to save checkpoints (default: ./checkpoints)')
parser.add_argument('--vis_dir', type=str, default=r'./val_out', metavar='str',
                    help='dir to save results during training (default: ./val_out_G)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0002)')
parser.add_argument('--max_num_epochs', type=int, default=200, metavar='N',
                    help='max number of training epochs (default 200)')
args = parser.parse_args()


if __name__ == '__main__':

    # args.net_G = 'resnet50'
    # args.checkpoint_dir = 'checkpoints_G_resnet50'
    # args.in_size = 384

    # args.net_G = 'coord_resnet50'
    # args.checkpoint_dir = 'checkpoints_G_coord_resnet50'
    # args.in_size = 384

    dataloaders = utils.get_loaders(args)

    # # How to check if the data is loading correctly?
    # dataloaders = utils.get_loaders(args)
    # for i in range(100):
    #     data = next(iter(dataloaders['train']))
    #     vis_A = utils.make_numpy_grid(data['A'])
    #     vis_B = utils.make_numpy_grid(data['B'])
    #     vis = np.concatenate([vis_A, vis_B], axis=0)
    #     print(data['A'].shape)
    #     print(data['B'].shape)
    #     plt.imshow(vis)
    #     plt.show()

    skydet = SkyDetector(args=args, dataloaders=dataloaders)
    skydet.train_models()


