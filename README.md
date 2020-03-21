# Convolutions As Matrix Multiplication

Conversion to matrix multiplication is not the most efficient way to implement convolutions with modern day Deep Learning Libraries employing the more novel techniques such as the Fast Fourier Transform(FFT) for larger filter sizes and Winograd Transform for smaller filter sizes(3x3 or 5x5). 
But this techique does present itself as a good starting point in understanding how CNNs can be implemented on TPUs, a hardware which is most famously associated with Standard Neural Networks.

This repository presents the pythonic code for this operation. More specifically, the code will impliment a correlation operation. This can be converted to a Convolution operation by simply flipping the kernel matrix by a 180 degrees.

For a detailed definition and inter-relation of the Convolution and Correlation Operations refer to: http://www.mathworks.com/help/images/what-is-image-filtering-in-the-spatial-domain.html
