"""image processing and unet training classes"""
import time
from datetime import date
from skimage import io
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import unet_class

class ImProcess():
    """class to process image previously to unet training"""

    def load_set(self, repim, repmask):
        """load images and get the train set
        width and height are required for the image to support the successives
        convolutions
        """
        self.images = io.imread_collection(repim, plugin='tifffile')
        self.masks = io.imread_collection(repmask, plugin='tifffile')


    def __init__(self, width, height, repim, repmask):
        self.width = width
        self.height = height
        self.crop_size = 44
        self.load_set(repim, repmask)


    def update_crop_size(self, crop_size):
        """update default crop_size value is required"""
        self.crop_size = crop_size


    def get_process_set(self, im_from, im_to):
        """prepare the images for the unet job"""
        im_process = []
        labels = []
        for i in range(im_from, im_to):
            im_process.append(self.images[i][:self.width, :self.height])
            labels.append(
                self.masks[i][self.crop_size:self.width - self.crop_size,
                              self.crop_size:self.height - self.crop_size])

        im_process = np.array(im_process) / 255
        im_process = np.transpose(im_process, (0, 3, 1, 2))
        labels = np.array(labels) / 255
        return im_process, labels


    def get_train_set(self, nb_im):
        """init attribut nb_im_train for the number of images used for the
        training"""
        return self.get_process_set(0, nb_im)


def save_image(input_image, file_name, transpose=()):
    """transform tensor into numpy matrix and save it as image"""
    np_image = input_image.squeeze(0).detach().cpu().numpy()
    if not all(transpose):
        np_image = np.transpose(np_image, transpose)
    plt.imshow(np_image)
    plt.savefig(file_name+'.jpg')


class UnetTrain():
    """apply unet method on processed dataset
    training and evaluation"""

    def __init__(self, batch_size, epochs, train_size):
        self.batch_size = batch_size
        self.nb_epochs = epochs
        self.train_size = train_size
        unet = unet_class.UNet(in_channel=3, out_channel=2)
        self.unet = unet.cuda()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.unet.parameters(), lr=0.01, momentum=0.99)


    def get_tensor_set(self, input_set, set_size, output_type='float'):
        """transform into tensor so the unet class can process it"""
        output_tensor = torch.as_tensor(
            input_set[set_size * self.batch_size: (set_size + 1) * self.batch_size])
        output_tensor = (
            output_tensor.float() if output_type == 'float'
            else output_tensor.long())
        return output_tensor.cuda()


    def train_step(self, inputs, labels):
        """training part"""
        width_out = inputs.shape[2] - 2 * self.unet.crop_size
        height_out = inputs.shape[3] - 2 * self.unet.crop_size
        self.optimizer.zero_grad()
        # forward + backward + optimize
        outputs = self.unet(inputs)
        outputs = outputs.permute(0, 2, 3, 1)
        outputs = outputs.resize(self.batch_size * width_out * height_out, 2)
        labels = labels.resize(self.batch_size * width_out * height_out)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss


    def log_unet(self, total_loss, batch_loss, time_unet):
        """log unet training"""
        with open('logunet.txt', 'w') as log_file:
            log_file.write("training %d epochs on %d images\n" % (self.nb_epochs, self.train_size))
            log_file.write("last total loss %d last batch loss %d \n" %
                           (total_loss, batch_loss))
            log_file.write("took %0.2f seconds\n" % time_unet)
        print("took %0.2f seconds" % (time_unet))


    def train_unet(self, x_train, y_train):
        """loop to train the unet model"""
        epoch_iter = np.ceil(self.train_size / self.batch_size).astype(int)
        time0 = time.time()
        for _ in tqdm(range(self.nb_epochs)):
            total_loss = 0
            for i in range(epoch_iter):
                batch_train_x = self.get_tensor_set(x_train, i)
                batch_train_y = self.get_tensor_set(y_train, i, 'long')

                batch_loss = self.train_step(batch_train_x, batch_train_y)
                total_loss += batch_loss
                torch.cuda.empty_cache()
        time1 = time.time()
        torch.save(
            self.unet.state_dict(),
            date.today().strftime("%y%m%d")+'unet'+str(self.nb_epochs)+'ep.pt')
        self.log_unet(total_loss, batch_loss, time1 - time0)
