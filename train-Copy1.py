from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from model import _netlocalD,_netG
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  default='streetview', help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot',  default='dataset/testset', help='path to test')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')

parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--nBottleneck', type=int,default=4000,help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred',type=int,default=4,help='overlapping edges')
parser.add_argument('--nef',type=int,default=64,help='of encoder filters in first conv layer')
parser.add_argument('--wtl2',type=float,default=0.998,help='0 means do not use else use with this weight')
parser.add_argument('--wtlD',type=float,default=0.001,help='0 means do not use else use with this weight')

parser.add_argument('--jittering', action='store_true', help='enables jittering')

opt = parser.parse_args()
print(opt)

def main():
    try:
        os.makedirs("result/test/cropped")
        os.makedirs("result/test/real")
        os.makedirs("result/test/recon")
        os.makedirs("model")
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    transform = transforms.Compose([transforms.Resize(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])
    dataset = dset.ImageFolder(root=opt.dataroot, transform=transform )
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=False, num_workers=int(opt.workers))

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = 3
    nef = int(opt.nef)
    nBottleneck = int(opt.nBottleneck)
    wtl2 = float(opt.wtl2)
    overlapL2Weight = 10

    netG = _netG(opt)
    netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
    print(netG)

    criterion = nn.BCELoss()
    criterionMSE = nn.MSELoss()

    input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0

    print(opt.batchSize)
    print(opt.imageSize)

    real_center = torch.FloatTensor(int(opt.batchSize), 3, int(opt.imageSize/2), int(opt.imageSize/2))
    #real_center = torch.FloatTensor(64, 3, 64,64)

    if opt.cuda:
        netG.cuda()
        criterion.cuda()
        criterionMSE.cuda()
        input_real, input_cropped,label = input_real.cuda(),input_cropped.cuda(), label.cuda()
        real_center = real_center.cuda()


    input_real = Variable(input_real)
    input_cropped = Variable(input_cropped)
    label = Variable(label)


    real_center = Variable(real_center)
    #jittering add
    randwf = random.uniform(-1.0,1.0)
    randhf = random.uniform(-1.0,1.0)
    if opt.jittering:
        jitterSizeW = int(opt.imageSize/5*randwf)
        jitterSizeH = int(opt.imageSize/5*randhf)
        print("jittering : W > ",jitterSizeW," H >",jitterSizeH)
    else :
        jitterSizeW = 0
        jitterSizeH = 0
    for i, data in enumerate(dataloader, 0):
        real_cpu, _ = data
        real_center_cpu = real_cpu[:,:,
                                   int(opt.imageSize/4+jitterSizeW):int(opt.imageSize/4+opt.imageSize/2+jitterSizeW),
                                   int(opt.imageSize/4+jitterSizeH):int(opt.imageSize/4+opt.imageSize/2+jitterSizeH)]
        batch_size = real_cpu.size(0)
        input_real.data.resize_(real_cpu.size()).copy_(real_cpu)
        input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)
        real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)
        input_cropped.data[:,0,
               int(opt.imageSize/4+opt.overlapPred+jitterSizeW):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred+jitterSizeW),
               int(opt.imageSize/4+opt.overlapPred+jitterSizeH):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred+jitterSizeH)] = 2*117.0/255.0 - 1.0
        input_cropped.data[:,1,
               int(opt.imageSize/4+opt.overlapPred+jitterSizeW):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred+jitterSizeW),
               int(opt.imageSize/4+opt.overlapPred+jitterSizeH):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred+jitterSizeH)] = 2*104.0/255.0 - 1.0
        input_cropped.data[:,2,
               int(opt.imageSize/4+opt.overlapPred+jitterSizeW):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred+jitterSizeW),
               int(opt.imageSize/4+opt.overlapPred+jitterSizeH):int(opt.imageSize/4+opt.imageSize/2-opt.overlapPred+jitterSizeH)] = 2*104.0/255.0 - 1.0

        label.data.resize_(batch_size).fill_(real_label)
        fake = netG(input_cropped)
        label.data.fill_(fake_label)
        
        errG = criterionMSE(fake,real_center)
        print('errG: %.4f' % errG.data[0])
        
        vutils.save_image(real_cpu,
                'result/test/real/real_samples.png')
        vutils.save_image(input_cropped.data,
                'result/test/cropped/cropped_samples.png')
        recon_image = input_cropped.clone()
        recon_image.data[:,:,
                         int(opt.imageSize/4+jitterSizeW):int(opt.imageSize/4+opt.imageSize/2+jitterSizeW),
                         int(opt.imageSize/4+jitterSizeH):int(opt.imageSize/4+opt.imageSize/2+jitterSizeH)] = fake.data
        vutils.save_image(recon_image.data,
                'result/test/recon/recon_center_samples.png')


if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!