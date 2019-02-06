from PIL import Image
from pylab import * 
from kernels import *
import sys


"""
Method Name : normalize_image() - normalizes the gradient responses, gradient magnitude and non_maxima
"""
def normalize_image(g_x,g_y,g_mag,non_maxima):
	g_x_norm = g_x.copy()
	g_y_norm = g_y.copy()
	g_mag_norm =g_mag.copy()
	non_maxima_norm = non_maxima.copy()
	g_x_norm=np.round(np.absolute(g_x_norm)/3)
	g_y_norm=np.round(np.absolute(g_y_norm)/3)
	g_mag_norm = np.round(((g_mag_norm)/(sqrt(2)*765))*255)
	non_maxima_norm = np.round(((non_maxima_norm)/(sqrt(2)*765))*255)
	return g_x_norm,g_y_norm,g_mag_norm,non_maxima_norm

"""
Method Name : cross_correlation() - performs cross-correlation operation on the image with the given kernel
                                    Here it is used for gaussian smoothing and for calculating horizontal and vertical 
                                    gradient responses.
Input parameters : im - input image
                   kernel - can be gaussian kernel or Horizontal/Vertical gradient operator
"""
def cross_correlation(im,kernel):
	im_correlated = im.copy()
	# if kernel size is 7*7 then k = 3, this is to consider the neighbours of center pixel
	k = int(floor(kernel.shape[0]/2))
	# The below for loop performs the cross-correlation.
	for i in range(k,im.shape[0]-k):
		for j in range(k,im.shape[1]-k):
			im_correlated[i,j] = np.sum(im[i-k:i+k+1,j-k:j+k+1] * kernel)
	#removes the undefined pixels i.e outer layers according to the size of kernel
	im_correlated_cropped = im_correlated[k:im.shape[0]-k, k:im.shape[1]-k]
	return im_correlated_cropped

"""
Method Name : smooth_image
input parameters: im - image to be smoothed
                  kernel_type - by default it will use gaussian kernel, 
	                            if one wants to use mean filter they have to mention while calling like kernel_type = 'mean'
	              kernel_size - if kernel_size is not given it will take a default kernel of size 7*7
	              gaussian sigma - it is used to create gaussian kernels with varying sigma's
output variables : smoothed_image - image result after smoothing
"""
def smooth_image(im,kernel_type="gaussian",kernel_size=None,gaussian_sigma=None):
	if(kernel_type.lower()=='mean'):
		kernel = np.ones(25).reshape(5,5)
	elif(kernel_type.lower()=='gaussian'):
		if(kernel_size is None):
			#default gaussian kernel
			kernel = np.array([[1,1,2,2,2,1,1],[1,2,2,4,2,2,1],[2,2,4,8,4,2,2],[2,4,8,16,8,4,2],[2,2,4,8,4,2,2],[1,2,2,4,2,2,1],[1,1,2,2,2,1,1]])
		else:
			#gaussian_filter() - creates gaussian filters of varying sizes and sigmas
			kernel = gaussian_filter(kernel_size,gaussian_sigma)
	#cross_correlation() - performs cross-correlation operation on image with kernel
	smoothed_image = cross_correlation(im,kernel)
	#np.sum() gives the sum of values in the kernel, here we are normalizing the smoothed image
	smoothed_image = smoothed_image/np.sum(kernel)
	return smoothed_image

"""
Method Name : gradient
input parameters: im - image.
                  operator_type - String, by default it will use prewitt filter, 
	                            if one wants to use Sobel filter they have to 
	                            mention while calling like operator_type = 'sobel'
output variables : g_x - Horizontal gradient response
                   g_y - Vertical gradiient response
                   g_mag - gradient magnitude
                   g_dir - gradient direction
"""
def gradient(im,operator_type='prewitt'):
	# calling gradient_operator for generating horizontal and vertical gradient operators
	op_x,op_y = gradient_operator(operator_type)
	# cross correlation of image with horizontal gradient operator
	g_x = cross_correlation(im,op_x)
	#cross correlation of image with Vertical gradient operator
	g_y = cross_correlation(im,op_y)
	#gradient magnitude
	g_mag = np.sqrt(np.square(g_x)+np.square(g_y))
	#gradient direction, g_dir will have values in radians and in the range of (-180 degrees to 180 degrees but in radians)  
	g_dir = np.arctan2(g_y,g_x)
	return g_x,g_y,g_mag,g_dir

"""
Method Name : non_maxima_suppression()
input parameters: g_mag - gradient magnitude of the image.
                  g_dir - gradient direction, values are in radians from -3.12 to 3.14
output variables : non_maxima - gradient magnitude of the image after non maxima suppression
"""
def non_maxima_suppression(g_mag,g_dir):
	#The below operation gives a matrix of sector values(0,1,2,3), where each value helps us to know the line of gradient
	g_sector = np.floor(np.remainder((g_dir+pi/8)/(pi/4),4)).astype("int")
	#to get postions of pixels that are to the east, North east, North and North west of current pixel
	positions_top = np.array([[0,1],[-1,1],[-1,0],[-1,-1]])
	#to get postions of pixels that are to the west, South west, South and South east of current pixel
	positions_bottom = np.array([[0,-1],[1,-1],[1,0],[1,1]])
	non_maxima = g_mag.copy()
	for i in range(1,g_mag.shape[0]-1):
		for j in range(1,g_mag.shape[1]-1):
			sector = g_sector[i,j]
			#positions of pixel along the line of gradient that has to be compared with current pixel
			neighbour1_pos = [i,j]+positions_top[sector]
			neighbour2_pos = [i,j]+positions_bottom[sector]
			# Comparing the current pixel with its neighbours along the line of gradient
			if(g_mag[i,j]<=g_mag[neighbour1_pos[0],neighbour1_pos[1]]):
				#suppressing the value by replacing with zero if the current pixel value is less than or equal to its neighbour
				non_maxima[i,j]=0
			if(g_mag[i,j]<=g_mag[neighbour2_pos[0],neighbour2_pos[1]]):
				non_maxima[i,j]=0
	#removing the undefined pixels i.e outer layers after non maxima suppression
	non_maxima = non_maxima[1:non_maxima.shape[0]-1,1:non_maxima.shape[1]-1]
	return non_maxima
"""
Method Name : simple_thresholding() - uses p-tile method for simple thresholding
input parameters: im - gradient magnitude after non maxima suppression and normalization
output variables : im_10, im_30, im_50 - set of of thresholded images, thresholded using p-tile method
	               T_10,T_30,T_50 - Thresholds for 10%, 30% and 50% respectively
	               edge_points_10,edge_points_30,edge_points_50 - Number of edge points detected for each threshold.
"""
def simple_thresholding(im):
	counts = [0]*256
	#calculating the frequency of each gray level value in the range (0,255)
	for i in range(0,im.shape[0]):
		for j in range(0,im.shape[1]):
			counts[int(im[i,j])]+=1
	count_cummulative = counts.copy()
	# producing cummulative of the frequencies from rear of the array to front
	for i in range(254,-1,-1):
		count_cummulative[i]=count_cummulative[i+1]+count_cummulative[i]
	#total edge points in the image i.e gray level value>0
	total_edge_pixels = count_cummulative[1]
	num_10 = total_edge_pixels/10
	num_30 = (total_edge_pixels/100)*30
	num_50 = total_edge_pixels/2
	min_10 = total_edge_pixels
	min_30 = total_edge_pixels
	min_50 = total_edge_pixels
	T_10 = None
	T_30 = None
	T_50 = None
	for i in range(255,0,-1):
		diff_10 = abs(num_10 - count_cummulative[i])
		diff_30 = abs(num_30 - count_cummulative[i])
		diff_50 = abs(num_50 - count_cummulative[i])
		if(diff_10<min_10):
			min_10 = diff_10
			T_10 = i
		if(diff_30<min_30):
			min_30 = diff_30
			T_30 = i
		if(diff_50<min_50):
			min_50 = diff_50
			T_50 = i
	im_10 = im.copy()
	im_30 = im.copy()
	im_50 = im.copy()
	#thresholding the image to produce set of binary images where gray level value 255 represents edge point 
	im_10[im_10>=T_10] = 255
	im_10[im_10<T_10] = 0
	im_30[im_30>=T_30] = 255
	im_30[im_30<T_30] = 0
	im_50[im_50>=T_50] = 255
	im_50[im_50<T_50] = 0
	#counting the number of edge points for each threshold method 
	edge_points_10 = np.count_nonzero(im_10==255)
	edge_points_30 = np.count_nonzero(im_30==255)
	edge_points_50 = np.count_nonzero(im_50==255)
	return im_10,im_30,im_50,T_10,T_30,T_50,edge_points_10,edge_points_30,edge_points_50
	
	"""
	Method Name: canny_edge_detector
	input parameters: 
	                im- image array for which edge points has to be detected
	                kernel_type - by default it will use gaussian kernel, 
	                              if one wants to use mean filter they have to mention while calling like kernel_type = 'mean'
	                kernel_size - if kernel_size is not given it will take a default kernel of size 7*7
	                gaussian sigma - it is used to create gaussian kernels with varying sigma's
	                gradient_operator = by default it uses prewitt operator if one needs to use sobel,
	                                    mention while calling like gradient_operator = 'sobel'
	output variables:
	                smoothed_image - image after gaussian smoothing
	                g_x - Horizontal gradient response
	                g_y - Vertical gradient response
	                g_mag - gradient magnitude
	                non_maxima - gradient magnitude after non maxima supression
	                im_10, im_30, im_50 - set of of thresholded images, thresholded using p-tile method
	                T_10,T_30,T_50 - Thresholds for 10%, 30% and 50% respectively
	                edge_points_10,edge_points_30,edge_points_50 - Number of edge points detected for each threshold.
	"""
def canny_edge_detector(im,kernel_type='gaussian',kernel_size=None,gaussian_sigma=None,gradient_operator='prewitt'):
	# smooth_image() for smoothing the image using gaussian filter of any size and sigma, one can also use mean filter
	smoothed_image = smooth_image(im,kernel_type,kernel_size,gaussian_sigma)
	#gradient() for producing gradient responses
	g_x,g_y,g_mag,g_dir = gradient(smoothed_image,gradient_operator)
	#non_maxima_suppression() - applies non maxima suppression on provided image
	non_maxima = non_maxima_suppression(g_mag,g_dir)
	#normalize_image - normalizes gradient responses, gradient magnitude and non_maxima image results.
	g_x,g_y,g_mag,non_maxima = normalize_image(g_x,g_y,g_mag,non_maxima)
	#simple_thresholding() - produces thresholded images using p-tile method
	im_10,im_30,im_50,T_10,T_30,T_50,edge_points_10,edge_points_30,edge_points_50 = simple_thresholding(non_maxima)
	return smoothed_image,g_x,g_y,g_mag,non_maxima,im_10,im_30,im_50,T_10,T_30,T_50,edge_points_10,edge_points_30,edge_points_50

def main():
	image_path = sys.argv[1]
	# Reading the image from the given path
	im = np.array(Image.open(image_path).convert("L")).astype("float")
	# Performing the Canny Edge Detection
	smoothed_image,g_x,g_y,g_mag,non_maxima,im_10,im_30,im_50,T_10,T_30,T_50,edge_points_10,edge_points_30,edge_points_50 = canny_edge_detector(im)
	# Displaying on the console, the thresholds and corresponding number of edge points detected
	print("Threshold value for P=10% : ", T_10,";", "No of Edge Points detected : ",edge_points_10)
	print("Threshold value for P=30% : ", T_30,";", "No of Edge Points detected : ",edge_points_30)
	print("Threshold value for P=50% : ", T_50,";", "No of Edge Points detected : ",edge_points_50)
	# saving the images in the current folder 
	gray()
	imsave('Edge_Detection_Images/smoothed_image.png',smoothed_image)
	imsave('Edge_Detection_Images/horizontal_gradient.png',g_x)
	imsave('Edge_Detection_Images/vertical_gradient.png',g_y)
	imsave('Edge_Detection_Images/edge_magnitude.png',g_mag)
	imsave('Edge_Detection_Images/non_maxima.png',non_maxima)
	imsave('Edge_Detection_Images/BinaryImage_10.png',im_10)
	imsave('Edge_Detection_Images/BinaryImage_30.png',im_30)
	imsave('Edge_Detection_Images/BinaryImage_50.png',im_50)


if __name__ == "__main__":
	main()

