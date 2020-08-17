"""main file that train and test unet method."""

import sys
import getopt
import unet_tools
# exec(open("unet_main.py").read())


def main(argv):
    """Load image and use it to train unet module."""
    #rep = '/home/bobette/pCloudDrive/Informatique/datascience/ImageSegmentation/'
    try:
        _, args = getopt.getopt(argv, "hg:d", ["help", "grammar="])
    except getopt.GetoptError:
        print('usage: unet_main.py nb_im nb_epochs')
        sys.exit(2)
    nb_im = int(args[0])
    rep_im = ("/home/bobette/pCloudDrive/Informatique/datascience/"
              "ImageSegmentation/data/breast_cancer_cell_seg/Images/*.tif")
    rep_mask = ("/home/bobette/pCloudDrive/Informatique/datascience/"
                "ImageSegmentation/data/breast_cancer_cell_seg/Masks/*.TIF")
    im_process = unet_tools.ImProcess(width=764, height=892, repim=rep_im, repmask=rep_mask)
    x_train, y_train = im_process.get_train_set(nb_im=int(args[0]))

    unet_train = unet_tools.UnetTrain(batch_size=1, epochs=int(args[1]), train_size=nb_im)
    unet_train.train_unet(x_train=x_train, y_train=y_train)

    x_test, y_test = im_process.get_process_set(
        unet_train.train_size,
        len(im_process.images))
    batch_test_x = self.get_tensor_set(x_test, 0)
    batch_test_y = self.get_tensor_set(y_test, 0, 'long')

    unet_tools.save_image()
    #test_xnp = batch_test_x.squeeze(0).detach().cpu().numpy()
    #plt.imshow(np.transpose(test_xnp, (1,2,0))[:,:,0])
    #plt.savefig('outputunet/testx'+str(j)+'.jpg')

if __name__ == "__main__":
    main(sys.argv[1:])
