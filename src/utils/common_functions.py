import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray, gray2rgb, rgba2rgb


def change_gray_range(image: np.ndarray, format: int = 255) -> np.ndarray:
    """
    Change (toggle) the gray range of the image
    @param image: the image
    @param format: the format to change to (1 or 255)
    @return: the image with the new gray range
    """
    if format == 255 and np.max(image) > 1:
        return image
    elif format == 1 and np.max(image) <= 1:
        return image
    if format == 255:
        return (image*255).astype(np.uint8)
    elif format == 1:
        return image/255


def show_images(images, titles=None):
    """
    Show the figures / plots inside the notebook
    @param images: list of images to show
    @param titles: list of titles corresponding to each image
    """
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def show_hist(img):
    """
    Show the histogram of the image
    @param img: the image
    """
    plt.figure()
    imgHist = histogram(img, nbins=256)

    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')


def get_symbol(symbol, restoredImage):
    """
    Get the symbol from the image, with the contour coordinates
    @param symbol: the symbol
    @param restoredImage: the image that contains the symbol
    @return: the symbol image, the coordinates of the contour

    """
    ymin = np.amin(symbol['contour'][:, 0])
    ymax = np.amax(symbol['contour'][:, 0])
    xmin = np.amin(symbol['contour'][:, 1])
    xmax = np.amax(symbol['contour'][:, 1])
    note_symbol = change_gray_range(restoredImage[round(
        ymin-1):round(ymax+1), round(xmin-1):round(xmax+1)].copy(), 255)
    return note_symbol, ymin, ymax, xmin, xmax


def any2gray(img: np.ndarray) -> np.ndarray:
    # Converts any image to grayscaled image, returns a copy of the image
    # Args:
    #    image (np.ndarray): image
    # Returns:
    #    np.ndarray: gray image
    image = img.copy()
    if len(image.shape) > 3:
        image = image[:, :, 0:3, 0]
    if len(image.shape) == 2:
        return image
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            return rgb2gray(image)
        elif image.shape[2] == 4:
            return rgb2gray(rgba2rgb(image))
    else:
        raise Exception("Invalid image shape")


def any2rgb(img: np.ndarray) -> np.ndarray:
    # Converts any image to rgb image, returns a copy of the image
    # Args:
    #    image (np.ndarray): image
    # Returns:
    #    np.ndarray: rgb image
    image = img.copy()
    if len(image.shape) > 3:
        image = image[:, :, 0:3, 0]
    if len(image.shape) == 2:
        return gray2rgb(image)
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            return image
        elif image.shape[2] == 4:
            return rgba2rgb(image)
    else:
        raise Exception("Invalid image shape")
