/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <stdlib.h>


__global__
void calc_min_or_max(float* d_min_or_max, 
              float* d_data, 
              unsigned int size,
              int minmax)
{
  unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
  for(unsigned int s = size/2; s > 0; s /= 2) {
      if(tid < s) {
          if(minmax == 0) {
              d_data[tid] = min(d_data[tid], d_data[tid+s]);
          } else {
              d_data[tid] = max(d_data[tid], d_data[tid+s]);
          }
      }
      __syncthreads();
  }
  if(tid==0){
    *d_min_or_max = d_data[0];
  }
}



float get_min_or_max(const float* const d_logLuminance, 
                      size_t numRows,
                      size_t numCols,
                      size_t max_or_min)
{
  unsigned int photoSize = numRows*numCols;

  // ans
  float* d_min_or_max;
  cudaMalloc((void**)&d_min_or_max, sizeof(float));
  cudaMemset(d_min_or_max, 0, sizeof(float));

  float* d_data;
  cudaMalloc((void**)&d_data, photoSize*sizeof(float));
  cudaMemcpy(d_data, d_logLuminance, photoSize*sizeof(float), cudaMemcpyDeviceToDevice);

  unsigned int blockNum = 1024;
  unsigned int blockSize = (int)ceil( (float)photoSize/(float)blockNum );
  calc_min_or_max<<<blockNum, blockSize>>>(d_min_or_max, d_data, photoSize, max_or_min);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());  

  float h_min_or_max;
  cudaMemcpy(&h_min_or_max, d_min_or_max, sizeof(float), cudaMemcpyDeviceToHost);
  return h_min_or_max;

}



__global__
void histogram_kernel(unsigned int* d_histo, const float* d_logLuminance, const int numBins, const float min_logLum, const float max_logLum, const int photoSize) {

  int index = threadIdx.x + blockDim.x * blockIdx.x;
  float lumRange = max_logLum - min_logLum;
  int binIdx = (d_logLuminance[index] - min_logLum) / lumRange * numBins;
  atomicAdd(&d_histo[binIdx], 1);

}

__global__
void hillis_steele_exclusive_scan(unsigned int* d_pdf, unsigned int* d_histo, const size_t numBins)
{
  int tid = threadIdx.x;
  // use left shift '<<' to multiply by 2 each iteration
  for (unsigned int s = 1; s < numBins; s <<= 1) {
    int left_neighborid = tid - s;
    if (left_neighborid >= 0)
      d_histo[tid] = d_histo[tid] + d_histo[left_neighborid];
    __syncthreads();
  }

  // convert the above inclusive scan to an exclusive one
  
  if (tid == 0) {
    d_pdf[tid] = 0;
  } else {
    d_pdf[tid] = d_histo[tid-1];
  }
}


void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  int threadPerBlock, blockNum;
  int photoSize = numRows*numCols;

  min_logLum = get_min_or_max(d_logLuminance, numRows, numCols, 0);
  max_logLum = get_min_or_max(d_logLuminance, numRows, numCols, 1);


  // init d_histo
  unsigned int* d_histo;
  cudaMalloc((void**)&d_histo, sizeof(unsigned int)*numBins);
  cudaMemset(d_histo, 0, sizeof(unsigned int)*numBins);
  // Calc d_histo
  threadPerBlock = 1024;
  blockNum = (int)ceil( (float)photoSize/(float)threadPerBlock );
  histogram_kernel<<<blockNum, threadPerBlock>>>(d_histo, d_logLuminance, numBins, min_logLum, max_logLum, photoSize);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


  // hillis_steele exclusive Scan to calc cdf
  hillis_steele_exclusive_scan<<<1, numBins>>>(d_cdf, d_histo, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


  // Free mem
  cudaFree(d_histo);

  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
}





