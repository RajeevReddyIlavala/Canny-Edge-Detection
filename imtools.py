from pylab import *
from numpy import *
from PIL import Image
def histEq(im,no_bins=256):
	im_hist,bins = histogram(im.flatten(),no_bins,density=True)
	cdf = im_hist.cumsum()
	cdf = 255 * cdf / cdf[-1]
	im2 = interp(im.flatten(),bins[:-1],cdf)
	return im2.reshape(im.shape),cdf
