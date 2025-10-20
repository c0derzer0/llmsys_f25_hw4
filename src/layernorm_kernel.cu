#include "includes/block_reduce.h"
#include "includes/kernels.h"
#include "includes/cuda_util.h"

#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {

const float LN_EPSILON = 1e-8f;
#define TILE_DIM 32


/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
ln_res: [batch_size * seq_len, hidden_size], ln result.
vars: [batch_size * seq_len], variance per token
means: [batch_size * seq_len], means per token, can be nullput
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/
template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {
  
  /// BEGIN ASSIGN4_2_1
  /// TODO
  // Hints:
  // 1. Compute x and x^2 with reinterpret_cast by casting to float4 for speedup
  // 2. Compute reduce sum with blockReduce and add epsilon with LN_EPSILON
  // 3. Compute layernorm result with reinterpret_cast by casting to float4 for speedup
  
  // Step 1
  float l_sum = 0;
  float l_sum_sq = 0;
  float sum_values[2];
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;  
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    l_sum += val.x + val.y + val.z + val.w;
    l_sum_sq += (val.x * val.x) + (val.y * val.y) + (val.z * val.z) + (val.w * val.w);
  }
  sum_values[0] = l_sum;
  sum_values[1] = l_sum_sq;
  
  // Step 2
  blockReduce<ReduceType::kSum, 2>(sum_values);
  
  // write shared
  __shared__ float mean;
  __shared__ float mean_of_squares;
  __shared__ float var;

  if (threadIdx.x == 0) {
    l_sum = sum_values[0];
    l_sum_sq = sum_values[1];
    mean = __fdividef(l_sum, hidden_size*4);
    mean_of_squares = __fdividef(l_sum_sq, hidden_size*4);
    var = (mean_of_squares - (mean * mean)) + LN_EPSILON;

    if (means) means[blockIdx.x] = mean;
    vars[blockIdx.x] = var;
  }
  __syncthreads();
  // Step 3

  float4 *ln_res_f4 = reinterpret_cast<float4 *>(ln_res) + blockIdx.x * (hidden_size); 
  const float4 *scale_f4 = reinterpret_cast<const float4 *>(scale); 
  const float4 *bias_f4 = reinterpret_cast<const float4 *>(bias); 
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    float4 scale_val = scale_f4[idx];
    float4 bias_val = bias_f4[idx];
    float4 result;
    result.x = ((val.x - mean) / sqrtf(var)) * scale_val.x + bias_val.x;
    result.y = ((val.y - mean) / sqrtf(var)) * scale_val.y + bias_val.y;
    result.z = ((val.z - mean) / sqrtf(var)) * scale_val.z + bias_val.z;
    result.w = ((val.w - mean) / sqrtf(var)) * scale_val.w + bias_val.w;
    ln_res_f4[idx] = result;
  }
  /// END ASSIGN4_2_1
}

extern "C" {
void launch_layernorm(float *ln_res, float *vars, float *means,
                              const float *inp, const float *scale,
                              const float *bias, int batch_size, int hidden_dim,
                              cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  int float_size = sizeof(float);
  int input_size = batch_size * hidden_dim * float_size;
  int scale_size = hidden_dim * float_size;
  int bias_size = hidden_dim * float_size;
  int output_size = batch_size * hidden_dim * float_size;
  int mean_size = batch_size * float_size;
  int var_size = batch_size * float_size;


  float *d_ln_res, *d_vars, *d_means, *d_inp, *d_scale, *d_bias;
  cudaMalloc((void **)&d_ln_res, output_size);
  cudaMalloc((void **)&d_vars, var_size);
  cudaMalloc((void **)&d_means, mean_size);
  cudaMalloc((void **)&d_inp, input_size);
  cudaMalloc((void **)&d_scale, scale_size);
  cudaMalloc((void **)&d_bias, bias_size);

  cudaMemcpy(d_inp, inp, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, scale, scale_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);

  // For using float4
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      d_ln_res, d_vars, d_means, d_inp, d_scale, d_bias, hidden_dim);

  // Copy back to the host
  cudaMemcpy(ln_res, d_ln_res, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vars, d_vars, var_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(means, d_means, mean_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm Error: %s\n", cudaGetErrorString(err));
    // Handle the error (e.g., by exiting the program)
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(d_ln_res);
  cudaFree(d_vars);
  cudaFree(d_means);
  cudaFree(d_inp);
  cudaFree(d_scale);
  cudaFree(d_bias);

}
}

/**
@brief: ker_ln_bw_dgamma_dbetta
Layer norm backword kernel, compute the gradient of gamma and betta.
dbetta = sum(dout, dim=0)
dgamma = sum(xhat * dout, dim=0)
xhat = (input - mean) * rsqrt(var) or
  (output - betta) / gamma

@thread
gridDim.x = hidden_size / 32
blockDim.x = 32
blockDim.y = 32

@param
gamma_grad: [hidden_size], gradient of gamma
betta_grad: [hidden_size], gradient of betta
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat, maybe nullptr
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat, maybe nullptr
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
(gamma && betta) ^ (vars && means) should be true
*/
template <typename T>
__global__ void ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad,
                                        const T *out_grad,
                                        const T *inp, const T *gamma,
                                        const T *betta, const T *vars,
                                        const T *means, int rows, int width) {

  /// BEGIN ASSIGN4_2_2
  /// TODO
  // Hints:
  // 1. Compute the partial gradients by looping across inp rows
  // 2. Store the partial gradients in the shared memory arrays
  // 3. Compute the reduce sum of the shared memory arrays with g.shfl_down
  //      -> More hints about `g.shfl_down`:
  //      -> https://developer.nvidia.com/blog/cooperative-groups/#:~:text=Using%20thread_block_tile%3A%3Ashfl_down()%20to%20simplify%20our%20warp%2Dlevel%20reduction%20does%20benefit%20our%20code%3A%20it%20simplifies%20it%20and%20eliminates%20the%20need%20for%20shared%20memory
  //      -> The highlighted line gives you a conceptual understanding of what the g.shfl_down is doing. Usually, the threads inside a block need to load everything to shared memory and work together to reduce the result (like what you have implemented in the hw1 for reduce function). 
  //      -> Now g.shfl_down helps you do so without consuming any shared memory. g.shfl_down makes it more efficient.
  // 4. Assign the final result to the correct position in the global output

  __shared__ float betta_buffer[TILE_DIM][TILE_DIM];
  __shared__ float gamma_buffer[TILE_DIM][TILE_DIM];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

  
  // Step 1
  float local_betta_grad = 0.0f;
  float local_gamma_grad = 0.0f;
  int col = threadIdx.x + blockIdx.x * blockDim.x;

  for (uint row = threadIdx.y; row < rows; row += blockDim.y) {
    if (col < width) {
      float inp_val = inp[row * width + col];
      float out_grad_val = out_grad[row * width + col];

      float xhat;
      if (means){
        float vars_val = vars[row];
        float means_val = means[row];
        xhat = (inp_val - means_val) * rsqrtf(vars_val);
      } else {
        float gamma_val = gamma[col];
        float betta_val = betta[col];
        xhat = (inp_val - betta_val) / gamma_val;
      }

      local_betta_grad += out_grad_val;
      local_gamma_grad += out_grad_val * xhat;
    }
  }
  // Step 2
  betta_buffer[threadIdx.y][threadIdx.x] = local_betta_grad;
  gamma_buffer[threadIdx.y][threadIdx.x] = local_gamma_grad;
  __syncthreads();

  // Step 3
  if (threadIdx.y == 0) {
    for (int k = 0; k < TILE_DIM; ++k) {
      int out_col = blockIdx.x * TILE_DIM + k;
      if (out_col >= width) break;
  
      float sum_beta  = betta_buffer[threadIdx.x][k]; 
      float sum_gamma = gamma_buffer[threadIdx.x][k];
  
      // sums over rows
      for (int offset = TILE_DIM / 2; offset > 0; offset >>= 1) {
        sum_beta  += g.shfl_down(sum_beta,  offset);
        sum_gamma += g.shfl_down(sum_gamma, offset);
      }
  
      // Step 4

      if (threadIdx.x == 0) {
        betta_grad[out_col] = sum_beta;
        gamma_grad[out_col] = sum_gamma;
      }
    }
  }

  /// END ASSIGN4_2_2
}

/**
@brief: ker_ln_bw_dinp
Layer norm backword kernel, compute the gradient of input.
dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
  * rsqrt(var)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
       (output - betta) / gamma if mean is nullptr
dxhat = dout * gamma


@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
inp_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
residual_grad: [batch_size * seq_len, hidden_size], gradient of residual input,
  usually appear in pre-layer-norm for transformer layer, maybe nullptr
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat and dxhat
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat and dinp
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
*/
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, int hidden_dim) {
  
  /// BEGIN ASSIGN4_2_2
  /// TODO
  // Hints:
  // 1. Compute dxhat=dy*w with reinterpret_cast by casting to float4 for speedup
  // 2. Compute xhat with reinterpret_cast by casting to float4 for speedup
  // 3. Compute reduce sum for dxhat and dxhat*xhat with blockReduce
  // 4. Compute final gradient
  
  // Step 1
  const float4 *out_grad_f4 = reinterpret_cast<const float4 *>(out_grad) + blockIdx.x * (hidden_dim); 
  float4 *inp_grad_f4 = reinterpret_cast<float4 *>(inp_grad) + blockIdx.x * (hidden_dim); 
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * (hidden_dim);
  const float4 *gamma_f4 = reinterpret_cast<const float4 *>(gamma);
  const float4 *betta_f4 = reinterpret_cast<const float4 *>(betta);
  float sum_dxhat = 0.0f;
  float sum_dxhat_xhat = 0.0f; 
  float sum_values[2];
  // Step 2
   
  for (uint idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    float4 inp_val = inp_f4[idx];
    float4 xhat_f4;
    float4 dxhat_f4;
    float4 betta_val = betta_f4[idx];
    float4 gamma_val = gamma_f4[idx];
    float4 out_grad_val = out_grad_f4[idx];
 
    if (means){
      xhat_f4.x = (inp_val.x - means[blockIdx.x]) * rsqrtf(vars[blockIdx.x]);
      xhat_f4.y = (inp_val.y - means[blockIdx.x]) * rsqrtf(vars[blockIdx.x]);
      xhat_f4.z = (inp_val.z - means[blockIdx.x]) * rsqrtf(vars[blockIdx.x]);
      xhat_f4.w = (inp_val.w - means[blockIdx.x]) * rsqrtf(vars[blockIdx.x]);

    } else {
      xhat_f4.x = (inp_val.x - betta_val.x) / gamma_val.x;
      xhat_f4.y = (inp_val.y - betta_val.y) / gamma_val.y;
      xhat_f4.z = (inp_val.z - betta_val.z) / gamma_val.z;
      xhat_f4.w = (inp_val.w - betta_val.w) / gamma_val.w;
    }

    dxhat_f4.x = out_grad_val.x * gamma_val.x; 
    dxhat_f4.y = out_grad_val.y * gamma_val.y; 
    dxhat_f4.z = out_grad_val.z * gamma_val.z; 
    dxhat_f4.w = out_grad_val.w * gamma_val.w; 

    sum_dxhat += dxhat_f4.x + dxhat_f4.y + dxhat_f4.z + dxhat_f4.w;
    sum_dxhat_xhat += dxhat_f4.x * xhat_f4.x + dxhat_f4.y * xhat_f4.y + dxhat_f4.z * xhat_f4.z + dxhat_f4.w * xhat_f4.w;
  }
  sum_values[0] = sum_dxhat;
  sum_values[1] = sum_dxhat_xhat;
  
  // Step 3
  blockReduce<ReduceType::kSum, 2>(sum_values);
  __shared__ float s_sum_dxhat;
  __shared__ float s_sum_dxhat_xhat;
  if (threadIdx.x == 0) {
    s_sum_dxhat = sum_values[0];
    s_sum_dxhat_xhat = sum_values[1];
  }
  __syncthreads();

  // Step 4
  for (uint idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    float4 inp_val = inp_f4[idx];
    float4 xhat_f4;
    float4 dxhat_f4;
    float4 betta_val = betta_f4[idx];
    float4 gamma_val = gamma_f4[idx];
    float4 out_grad_val = out_grad_f4[idx];
    float4 inp_grad_val;
 
    if (means){
      xhat_f4.x = (inp_val.x - means[blockIdx.x]) * rsqrtf(vars[blockIdx.x]);
      xhat_f4.y = (inp_val.y - means[blockIdx.x]) * rsqrtf(vars[blockIdx.x]);
      xhat_f4.z = (inp_val.z - means[blockIdx.x]) * rsqrtf(vars[blockIdx.x]);
      xhat_f4.w = (inp_val.w - means[blockIdx.x]) * rsqrtf(vars[blockIdx.x]);

    } else {
      xhat_f4.x = (inp_val.x - betta_val.x) / gamma_val.x;
      xhat_f4.y = (inp_val.y - betta_val.y) / gamma_val.y;
      xhat_f4.z = (inp_val.z - betta_val.z) / gamma_val.z;
      xhat_f4.w = (inp_val.w - betta_val.w) / gamma_val.w;
    }

    dxhat_f4.x = out_grad_val.x * gamma_val.x; 
    dxhat_f4.y = out_grad_val.y * gamma_val.y; 
    dxhat_f4.z = out_grad_val.z * gamma_val.z; 
    dxhat_f4.w = out_grad_val.w * gamma_val.w; 

    inp_grad_val.x = (dxhat_f4.x - (s_sum_dxhat + (xhat_f4.x * s_sum_dxhat_xhat)) / (hidden_dim * 4)) * rsqrtf(vars[blockIdx.x]);
    inp_grad_val.y = (dxhat_f4.y - (s_sum_dxhat + (xhat_f4.y * s_sum_dxhat_xhat)) / (hidden_dim * 4)) * rsqrtf(vars[blockIdx.x]);
    inp_grad_val.z = (dxhat_f4.z - (s_sum_dxhat + (xhat_f4.z * s_sum_dxhat_xhat)) / (hidden_dim * 4)) * rsqrtf(vars[blockIdx.x]);
    inp_grad_val.w = (dxhat_f4.w - (s_sum_dxhat + (xhat_f4.w * s_sum_dxhat_xhat)) / (hidden_dim * 4)) * rsqrtf(vars[blockIdx.x]);
  
    inp_grad_f4[idx] = inp_grad_val;
  }
  /// END ASSIGN4_2_2
}

extern "C" {
void launch_layernorm_bw(float *gamma_grad, float *betta_grad, float *inp_grad,
                         const float *out_grad, const float *inp, const float *gamma,
                         const float *betta, const float *vars,
                         const float *means, int batch_size, int hidden_dim,
                         cudaStream_t stream_1, cudaStream_t stream_2) {
  
  // Allocate device memory
  float *d_gamma_grad, *d_betta_grad, *d_inp_grad, *d_out_grad, *d_inp, *d_gamma, *d_betta, *d_vars, *d_means;
  int grad_output_size = batch_size * hidden_dim * sizeof(float);
  int gamma_betta_size = hidden_dim * sizeof(float);
  int vars_means_size = batch_size * sizeof(float);

  cudaMalloc((void **)&d_gamma_grad, gamma_betta_size);
  cudaMalloc((void **)&d_betta_grad, gamma_betta_size);
  cudaMalloc((void **)&d_inp_grad, grad_output_size);
  cudaMalloc((void **)&d_out_grad, grad_output_size);
  cudaMalloc((void **)&d_inp, grad_output_size);
  cudaMalloc((void **)&d_gamma, gamma_betta_size);
  cudaMalloc((void **)&d_betta, gamma_betta_size);
  cudaMalloc((void **)&d_vars, vars_means_size);
  cudaMalloc((void **)&d_means, vars_means_size);

  // Copy memory to device
  cudaMemcpy((void *)d_out_grad, out_grad, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_inp, inp, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_gamma, gamma, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_betta, betta, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_vars, vars, vars_means_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_means, means, vars_means_size, cudaMemcpyHostToDevice);

  // Launch kernels
  // Compute grad of gamma and betta
  // This calculates the number of blocks needed to cover the data along the specified dimension, rounds it up.
  // The result is then multiplied by TILE_DIM to ensure that the grid size is a multiple of TILE_DIM.
  dim3 grid_dim(((hidden_dim + TILE_DIM - 1) / TILE_DIM) * TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);
  ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream_1>>>(
      d_gamma_grad, d_betta_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars,
      d_means, batch_size, hidden_dim);

  // Compute grad of input
  if (hidden_dim % 4 != 0 || hidden_dim > 4096) {
    throw std::runtime_error("hidden_dim % 4 != 0 || hidden_dim > 4096");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  ker_ln_bw_dinp<<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means, hidden_dim);

  // Synchronize and check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm_bw Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy back to host
  cudaMemcpy(gamma_grad, d_gamma_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(betta_grad, d_betta_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(inp_grad, d_inp_grad, grad_output_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_gamma_grad);
  cudaFree(d_betta_grad);
  cudaFree(d_inp_grad);
  cudaFree((void *)d_out_grad);
  cudaFree((void *)d_inp);
  cudaFree((void *)d_gamma);
  cudaFree((void *)d_betta);
  cudaFree((void *)d_vars);
  cudaFree((void *)d_means);
}}
}}
