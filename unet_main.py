"""main file that train and test unet method."""

import time
from datetime import date
import numpy as np
from tqdm import tqdm
import torch
import unet_class
# exec(open("unet_main.py").read())

#global UNET
UNET = unet_class.UNet(in_channel=3, out_channel=2)
UNET = UNET.cuda()
def train_step(inputs, labels, optimizer, criterion, batch_size, crop_size):
    """training part"""
    width_out = inputs.shape[2] - 2 * crop_size
    height_out = inputs.shape[3] - 2 * crop_size
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs = UNET(inputs)
    outputs = outputs.permute(0, 2, 3, 1)
    outputs = outputs.resize(batch_size * width_out * height_out, 2)
    labels = labels.resize(batch_size * width_out * height_out)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss


def log_unet(epochs, nb_im, total_loss, batch_loss, time_unet):
    """log unet training"""
    with open('logunet.txt', 'w') as log_file:
        log_file.write("training %d epochs on %d images\n" % (epochs, nb_im))
        log_file.write("last total loss %d last batch loss %d \n" %
                       (total_loss, batch_loss))
        log_file.write("took %0.2f seconds\n" % time_unet)
    print("took %0.2f seconds" % (time_unet))


def train_unet(batch_size, epochs, x_train, y_train, crop_size):
    """loop to train the unet model"""
    #UNET = unet_class.UNet(in_channel=3, out_channel=2)
    #UNET = UNET.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(UNET.parameters(), lr=0.01, momentum=0.99)
    nb_im = len(x_train)
    epoch_iter = np.ceil(nb_im / batch_size).astype(int)
    time0 = time.time()
    for _ in tqdm(range(epochs)):
        total_loss = 0
        for i in range(epoch_iter):
            batch_train_x = torch.as_tensor(
                x_train[i * batch_size: (i + 1) * batch_size]).float()
            batch_train_y = torch.as_tensor(
                y_train[i * batch_size: (i + 1) * batch_size]).long()
            batch_train_x = batch_train_x.cuda()
            batch_train_y = batch_train_y.cuda()
            batch_loss = train_step(batch_train_x,
                                    batch_train_y,
                                    optimizer,
                                    criterion,
                                    batch_size,
                                    crop_size=crop_size)
            total_loss += batch_loss
            torch.cuda.empty_cache()
    time1 = time.time()
    today = date.today()
    unetfile = today.strftime("%y%m%d") + 'unet' + str(epochs) + 'ep.pt'
    torch.save(UNET.state_dict(), unetfile)
    log_unet(epochs, nb_im, total_loss, batch_loss, time1 - time0)


def main():
    """Load image and use it to train unet module."""
    #rep = '/home/bobette/pCloudDrive/Informatique/datascience/ImageSegmentation/'
    repim = ("/home/bobette/pCloudDrive/Informatique/datascience/"
             "ImageSegmentation/data/breast_cancer_cell_seg/Images/*.tif")
    repmask = ("/home/bobette/pCloudDrive/Informatique/datascience/"
               "ImageSegmentation/data/breast_cancer_cell_seg/Masks/*.TIF")
    nb_im = 50
    crop_size = 44
    nb_epochs = 3

    x_train, y_train = unet_class.load_set(
        repim,
        repmask,
        width=764,
        height=892,
        crop_size=crop_size,
        nb_im=nb_im)

    train_unet(batch_size=1, epochs=nb_epochs, x_train=x_train,
               y_train=y_train, crop_size=crop_size)


if __name__ == "__main__":
    main()
