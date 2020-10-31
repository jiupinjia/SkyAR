from skyboxengine import *
from torch import nn
import torchvision
from torch.utils.data import Dataset
import time
import visdom

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
viz = visdom.Visdom(env='train')


class DataMaker(object):
    def __init__(self):
        self.name = 'motion_estimator'
        #
        self.datadir = ''
        self.in_size_w = 384
        self.in_size_h = 384
        self.out_size_w = 845
        self.out_size_h = 480
        self.net_G = define_G(input_nc=3, output_nc=1, ngf=64, netG="coord_resnet50").to(device)
        checkpoint = torch.load(os.path.join("./checkpoints_G_coord_resnet50", 'best_ckpt.pt'))
        # checkpoint = torch.load(os.path.join(self.ckptdir, 'last_ckpt.pt'))
        self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
        self.net_G.to(device)
        self.net_G.eval()

    def prepare_data(self, video_path):
        self.datadir = video_path
        cap = cv2.VideoCapture(self.datadir)
        m_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        img_HD_prev = None
        idx = 0
        while (1):
            ret, frame = cap.read()
            if ret:
                img_HD = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_HD = np.array(img_HD / 255., dtype=np.float32)
                img_HD = cv2.resize(img_HD, (self.out_size_w, self.out_size_h))

                if img_HD_prev is None:
                    img_HD_prev = img_HD

                h, w, c = img_HD.shape

                img = cv2.resize(img_HD, (self.in_size_w, self.in_size_h))

                img = np.array(img, dtype=np.float32)
                img = torch.tensor(img).permute([2, 0, 1]).unsqueeze(0)

                with torch.no_grad():
                    G_pred = self.net_G(img.to(device))
                    G_pred = torch.nn.functional.interpolate(G_pred, (h, w), mode='bilinear', align_corners=False)  # bicubic
                    G_pred = G_pred[0, :].permute([1, 2, 0])
                    G_pred = torch.cat([G_pred, G_pred, G_pred], dim=-1)
                    G_pred = np.array(G_pred.detach().cpu())
                    G_pred = np.clip(G_pred, a_max=1.0, a_min=0.0)

                r, eps = 20, 0.01
                refined_skymask = guidedFilter(img_HD[:, :, 2], G_pred[:, :, 0], r, eps)

                refined_skymask = np.stack(
                    [refined_skymask, refined_skymask, refined_skymask], axis=-1)

                skymask = np.clip(refined_skymask, a_min=0, a_max=1)

                dxdyda = self._skybox_tracking(img_HD, img_HD_prev, skymask, idx)

                print(dxdyda)
                img_HD_prev = img_HD
                idx += 1

            else:  # if reach the last frame
                break

    def _skybox_tracking(self, frame, frame_prev, skymask, frame_indx):

        if np.mean(skymask) < 0.05:
            print('sky area is too small')
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_RGB2GRAY)
        prev_gray = np.array(255*prev_gray, dtype=np.uint8)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        curr_gray = np.array(255*curr_gray, dtype=np.uint8)

        mask = np.array(skymask[:,:,0] > 0.99, dtype=np.uint8)
        front_mask = np.array(skymask[:, :, 0] < 0.9, dtype=np.uint8)

        template_size = int(0.05*mask.shape[0])
        mask = cv2.erode(mask, np.ones([template_size, template_size]))

        flow = cv2.calcOpticalFlowFarneback(prev_gray*front_mask, curr_gray*front_mask, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # show flow
        # hsv = np.zeros_like(frame_prev)
        # hsv[..., 1] = 255
        # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # hsv[..., 0] = ang * 180 / np.pi / 2
        # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        #
        # cv2.imshow('frame1', curr_gray)
        # cv2.imshow('frame2', rgb)
        # k = cv2.waitKey(30)

        # ShiTomasi corner detection
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray, mask=mask, maxCorners=200,
            qualityLevel=0.01, minDistance=30, blockSize=3)

        if prev_pts is None:
            print('no feature point detected')
            return np.array([[0, 0, 0]], dtype=np.float32)
            # return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        # Calculate optical flow (i.e. track feature points)

        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None)
        # Filter only valid points
        idx = np.where(status == 1)[0]
        if idx.size == 0:
            print('no good point matched')
            return np.array([[0, 0, 0]], dtype=np.float32)
            # return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        prev_pts, curr_pts = removeOutliers(prev_pts, curr_pts)

        if curr_pts.shape[0] < 10:
            print('no good point matched')
            return np.array([[0, 0, 0]], dtype=np.float32)
            # return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        # limit the motion to translation + rotation
        dxdyda = estimate_partial_transform((
            np.array(prev_pts), np.array(curr_pts)))

        # if frame_indx == 0:
        #     return np.array([[0, 0, 0]], dtype=np.float32)
        dataname = os.path.splitext(os.path.basename(self.datadir))[0]
        if not os.path.exists('./data/flow/'):
            os.mkdir('./data/flow')
            os.mkdir('./data/flow/Train')
            os.mkdir('./data/flow/Val')
        if not os.path.exists('./data/dxdyda'):
            os.mkdir('./data/dxdyda')
            os.mkdir('./data/dxdyda/Train')
            os.mkdir('./data/dxdyda/Val')
        if frame_indx % 10 == 0:
            np.save('./data/flow/Val/'+dataname+'_flow_'+str(frame_indx), flow)
            np.save('./data/dxdyda/Val/' + dataname + '_dxdyda_' + str(frame_indx), dxdyda)
        else:
            np.save('./data/flow/Train/' + dataname + '_flow_' + str(frame_indx), flow)
            np.save('./data/dxdyda/Train/' + dataname + '_dxdyda_' + str(frame_indx), dxdyda)

        # m = build_transformation_matrix(dxdyda)

        return dxdyda


class MotionDataset(Dataset):
    def __init__(self, data_folder, split):

        self.split = split
        assert self.split in {'Train', 'Val', 'Test'}

        self.flow_folder = data_folder+'/flow/'+self.split+'/'
        self.dxdyda_folder = data_folder+'/dxdyda/'+self.split+'/'

        self.dataset_size = len([name for name in os.listdir(self.flow_folder)])

    def __getitem__(self, i):
        flow = torch.FloatTensor(np.load(self.flow_folder+os.listdir(self.flow_folder)[i]))
        flow = flow.permute(2,0,1)
        dxdyda = torch.FloatTensor(np.load(self.dxdyda_folder+os.listdir(self.dxdyda_folder)[i]))
        return flow, dxdyda

    def __len__(self):
        return self.dataset_size


class MotionEstimator(nn.Module):

    def __init__(self):
        super(MotionEstimator, self).__init__()
        net = torchvision.models.resnet50(pretrained=False)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 3)
        net.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net = net

    def forward(self, flow):
        dxdydz = self.net(flow)
        return dxdydz


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def save_checkpoint(checkpoint_path, checkpoint_name, epoch, epochs_since_improvement, estimator, optimizer, is_best, recent_losses):

    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'estimator': estimator,
             'optimizer': optimizer,
             'recent_losses': recent_losses}
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, checkpoint_path+'/'+'BEST_' + checkpoint_name+ '.pth.tar')
    # elif epoch % 5 == 0:
    torch.save(state, checkpoint_path+'/'+ checkpoint_name + '_'+str(epoch)+'.pth.tar')


def train(train_loader, estimator, optimizer, criterion, epoch):
    estimator.train()
    start_time = time.time()

    for i, (flow, dxdyda) in enumerate(train_loader):

        flow = flow.to(device)
        dxdyda = dxdyda.to(device)

        optimizer.zero_grad()

        # Forward prop.
        pri_dxdyda = estimator(flow)

        # Calculate loss
        loss = criterion(pri_dxdyda, dxdyda)

        loss.backward()

        # Update weights
        optimizer.step()

        epoch_time = time.time() - start_time
        start_time = time.time()
        # Print status
        losses = loss.item()
        print_freq = 10
        if i % print_freq == 0:
            viz.line(X=torch.FloatTensor([epoch * math.ceil(len(train_loader) / print_freq) + i // print_freq]), Y=torch.FloatTensor([loss.item()]), win='Train_Loss', name='1', update='append')
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Epoch Time {epoch_time:.3f}\t'
                  'Loss {loss:.4f}'.format(epoch, i, len(train_loader),
                                                                          epoch_time=epoch_time, loss=losses))


def validate(val_loader, estimator, criterion):

    estimator.eval()  # eval mode (no dropout or batchnorm)
    losses = 0

    with torch.no_grad():
        # Batches
        for i, (flow, dxdyda) in enumerate(val_loader):

            # Move to device, if available
            flow = flow.to(device)
            dxdyda = dxdyda.to(device)

            pri_dxdyda = estimator(flow)

            # Calculate loss
            loss = criterion(pri_dxdyda, dxdyda)

            losses+=loss
        losses = losses/len(val_loader)
        print(
            '\n * Val Loss: {loss:.3f}\n'.format(loss=losses))

    return losses


def main():
    encoder_lr = 1e-6
    batch_size = 12
    workers = 4
    epochs = 200
    data_folder = './data'
    checkpoint_path = './motion_estimator_checkpoint'
    checkpoint_name = 'Resnet50_Motion_Estimate'
    last_checkpoint = './motion_estimator_checkpoint/BEST_Resnet50_Motion_Estimate.pth.tar'

    if last_checkpoint is None:
        start_epoch = 0
        estimator = MotionEstimator()
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, estimator.parameters()),
                                     lr=encoder_lr)
        estimator = estimator.to(device)
        epochs_since_improvement = 0
        highest_losses = 1000
    else:
        checkpoint = torch.load(last_checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        estimator = checkpoint['etimator']
        optimizer = checkpoint['optimizer']
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        highest_losses = checkpoint['recent_losses']

    criterion = nn.MSELoss().to(device)

    train_loader = torch.utils.data.DataLoader(
        MotionDataset(data_folder, 'Train'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        MotionDataset(data_folder, 'Val'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    for epoch in range(start_epoch, epochs):
        # if epochs_since_improvement == 10:
        #     break
        # if epochs_since_improvement > 0 and epochs_since_improvement % 3 == 0:  # 8
        #     adjust_learning_rate(optimizer, 0.01)  # 0.8

        # One epoch's training
        train(train_loader=train_loader, estimator = estimator, optimizer=optimizer, criterion=criterion, epoch=epoch)

        # One epoch's validation
        recent_losses = validate(val_loader=val_loader, estimator=estimator, criterion=criterion)
        viz.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([recent_losses]), win='Val Loss', name='1', update='append')

        # Check if there was an improvement
        is_best = recent_losses < highest_losses
        highest_losses = min(recent_losses, highest_losses)
        if not is_best:
            epochs_since_improvement += 1
            # print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        save_checkpoint(checkpoint_path, checkpoint_name, epoch, epochs_since_improvement, estimator, optimizer, is_best, recent_losses)


if __name__ == '__main__':

    makedata = False
    if makedata:
        datamaker = DataMaker()
        datamaker.prepare_data('./test_videos/canyon.mp4')
    else:
        # activate visdom sever: python -m visdom.server
        main()





