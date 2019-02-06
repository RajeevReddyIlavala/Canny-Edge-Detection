import numpy as np

"""
Method Name : gaussian_filter()
input parameters: n - size of gaussian kernel
                  sigma - Gaussian sigma for gaussian kernel
output variables : kernel - Gaussian kernel with given size and sigma, 
Note: The values in the kernel obtained here does not sum up to one.
"""
def gaussian_filter(n,sigma):
	k = floor(n/2)
	kernel = np.zeros((n,n))
	for i in range(0,n):
		for j in range(0,n):
			x = i-k
			y = j-k
			kernel[i,j] =  e** -((x**2+y**2)/(2*(sigma**2)))
	kernel = np.round(kernel/kernel[0,0])
	return kernel


"""
Method Name : gradient_operator() - generates horizontal and vertical gradient operators
input parameters: operator_type - it's a string, expected values are 'prewitt' or 'sobel'
output variables : op_x - horizontal gradient operator
                   op_y - vertical gradient operator
"""
def gradient_operator(operator_type):
	p = np.ones((1,3))
	q = np.array([1,0,-1])[np.newaxis,:]
	if(operator_type.lower() == "prewitt"):
		op_x = -1*p.T @ q
		op_y = q.T @ p
	if(operator_type.lower() == "sobel"):
		p[0,1] =2
		op_x = -1*p.T @ q
		op_y = q.T @ p
	return op_x,op_y