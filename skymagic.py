import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import argparse
from networks import *
from skyboxengine import *
import utils
import torch

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='SKYAR')
parser.add_argument('--path', type=str, default='./config/config-canyon-jupiter.json', metavar='str',
                    help='configurations')

class SkyFilter():

    def __init__(self, args):

        self.ckptdir = args.ckptdir
        self.datadir = args.datadir
        self.input_mode = args.input_mode

        self.in_size_w, self.in_size_h = args.in_size_w, args.in_size_h
        self.out_size_w, self.out_size_h = args.out_size_w, args.out_size_h

        self.skyboxengine = SkyBox(args)

        self.net_G = define_G(input_nc=3, output_nc=1, ngf=64, netG=args.net_G).to(device)
        self.load_model()

        self.video_writer = cv2.VideoWriter('demo.mp4', cv2.VideoWriter_fourcc(*'MP4V'),
                                            20.0, (args.out_size_w, args.out_size_h))
        self.video_writer_cat = cv2.VideoWriter('demo-cat.mp4', cv2.VideoWriter_fourcc(*'MP4V'),
                                            20.0, (2*args.out_size_w, args.out_size_h))

        if os.path.exists(args.output_dir) is False:
            os.mkdir(args.output_dir)

        self.save_jpgs = args.save_jpgs


    def load_model(self):

        print('loading the best checkpoint...')
        checkpoint = torch.load(os.path.join(self.ckptdir, 'best_ckpt.pt'))
        # checkpoint = torch.load(os.path.join(self.ckptdir, 'last_ckpt.pt'))
        self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
        self.net_G.to(device)
        self.net_G.eval()



    def write_video(self, img_HD, syneth):

        frame = np.array(255.0 * syneth[:, :, ::-1], dtype=np.uint8)
        self.video_writer.write(frame)

        frame_cat = np.concatenate([img_HD, syneth], axis=1)
        frame_cat = np.array(255.0 * frame_cat[:, :, ::-1], dtype=np.uint8)
        self.video_writer_cat.write(frame_cat)

        cv2.imshow('frame_cat', frame_cat)
        cv2.waitKey(1)



    def synthesize(self, img_HD, img_HD_prev):

        h, w, c = img_HD.shape

        img = cv2.resize(img_HD, (self.in_size_w, self.in_size_h))

        img = np.array(img, dtype=np.float32)
        img = torch.tensor(img).permute([2, 0, 1]).unsqueeze(0)

        with torch.no_grad():
            G_pred = self.net_G(img.to(device))
            G_pred = torch.nn.functional.interpolate(G_pred, (h, w), mode='bicubic', align_corners=False)
            G_pred = G_pred[0, :].permute([1, 2, 0])
            G_pred = torch.cat([G_pred, G_pred, G_pred], dim=-1)
            G_pred = np.array(G_pred.detach().cpu())
            G_pred = np.clip(G_pred, a_max=1.0, a_min=0.0)

        skymask = self.skyboxengine.skymask_refinement(G_pred, img_HD)

        syneth = self.skyboxengine.skyblend(img_HD, img_HD_prev, skymask)

        return syneth, G_pred, skymask



    def cvtcolor_and_resize(self, img_HD):

        img_HD = cv2.cvtColor(img_HD, cv2.COLOR_BGR2RGB)
        img_HD = np.array(img_HD / 255., dtype=np.float32)
        img_HD = cv2.resize(img_HD, (self.out_size_w, self.out_size_h))

        return img_HD



    def run_imgseq(self):

        print('running evaluation...')
        img_names = os.listdir(self.datadir)
        img_HD_prev = None

        for idx in range(len(img_names)):

            this_dir = os.path.join(self.datadir, img_names[idx])
            img_HD = cv2.imread(this_dir, cv2.IMREAD_COLOR)
            img_HD = self.cvtcolor_and_resize(img_HD)

            if img_HD_prev is None:
                img_HD_prev = img_HD

            syneth, G_pred, skymask  = self.synthesize(img_HD, img_HD_prev)

            if self.save_jpgs:
                fpath = os.path.join(args.output_dir, img_names[idx])
                plt.imsave(fpath[:-4] + '_input.jpg', img_HD)
                plt.imsave(fpath[:-4] + 'coarse_skymask.jpg', G_pred)
                plt.imsave(fpath[:-4] + 'refined_skymask.jpg', skymask)
                plt.imsave(fpath[:-4] + 'syneth.jpg', syneth.clip(min=0, max=1))

            self.write_video(img_HD, syneth)
            print('processing: %d / %d ...' % (idx, len(img_names)))

            img_HD_prev = img_HD




    def run_video(self):

        print('running evaluation...')

        cap = cv2.VideoCapture(self.datadir)
        m_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        img_HD_prev = None

        idx = 0

        while (1):
            ret, frame = cap.read()
            if ret:
                img_HD = self.cvtcolor_and_resize(frame)

                if img_HD_prev is None:
                    img_HD_prev = img_HD

                syneth, G_pred, skymask = self.synthesize(img_HD, img_HD_prev)

                if self.save_jpgs:
                    fpath = os.path.join(args.output_dir, str(idx)+'.jpg')
                    plt.imsave(fpath[:-4] + '_input.jpg', img_HD)
                    plt.imsave(fpath[:-4] + '_coarse_skymask.jpg', G_pred)
                    plt.imsave(fpath[:-4] + '_refined_skymask.jpg', skymask)
                    plt.imsave(fpath[:-4] + '_syneth.jpg', syneth.clip(min=0, max=1))

                self.write_video(img_HD, syneth)
                print('processing: %d / %d ...' % (idx, m_frames))

                img_HD_prev = img_HD
                idx += 1

            else:  # if reach the last frame
                break


    def run(self):
        if self.input_mode == 'seq':
            self.run_imgseq()
        elif self.input_mode == 'video':
            self.run_video()
        else:
            print('wrong input_mode, select one in [seq, video')
            exit()


if __name__ == '__main__':

    config_path = parser.parse_args().path
    args = utils.parse_config(config_path)
    sf = SkyFilter(args)
    sf.run()


