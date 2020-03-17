#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/div_rtn.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/AccumulateType.h>
//#include <ATen/native/Pool.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>

/*
#include <THC/THCNumerics.cuh>
#include <c10/macros/Macros.h>
*/

/*
void checkAllSame(CheckedFrom c, ArrayRef<TensorArg> tensors, void(*fn)(CheckedFrom, const TensorArg&, const TensorArg&)) {
  const TensorArg* t0 = nullptr;
  for (auto& t : tensors) {
    if (!t->defined()) continue;
    if (t0 != nullptr) {
      fn(c, *t0, t);
    } else {
      t0 = &t;
    }
  }
}

void checkSameGPU(CheckedFrom c, const TensorArg& t1, const TensorArg& t2) {
  if (! (t1->is_cuda()) || ! (t2->is_cuda())) {
    std::ostringstream oss;
    if (! t1->is_cuda()) {
      oss << "Tensor for " << t1 << " is on CPU, ";
    }
    if (! t2->is_cuda()) {
      oss << "Tensor for " << t2 << " is on CPU, ";
    }
    oss << "but expected " << ((!(t1->is_cuda() || t2->is_cuda())) ? "them" : "it")
        << " to be on GPU (while checking arguments for " << c << ")";
    AT_ERROR(oss.str());
  }
  TORCH_CHECK(
    t1->get_device() == t2->get_device(),
    "Expected tensor for ", t1, " to have the same device as tensor for ", t2,
    "; but device ", t1->get_device(), " does not equal ", t2->get_device(),
    " (while checking arguments for ", c, ")");
}

void checkAllSameGPU(CheckedFrom c, ArrayRef<TensorArg> tensors) {
  checkAllSame(c, tensors, checkSameGPU);
}

__device__ inline int min(int a, int b) {
  return a <= b ? a : b;
}

__device__ inline int max(int a, int b) {
  return a >= b ? a : b;
}
*/

template <typename dest_t, typename src_t>
static inline dest_t safe_downcast(src_t v)
{
  TORCH_CHECK(std::numeric_limits<dest_t>::min() <= v && v <= std::numeric_limits<dest_t>::max(),
              "integer out of range");

  return static_cast<dest_t>(v);
}

template<typename T>
static inline T pooling_output_shape_pad_lr(
        T inputSize, T kernelSize, T pad_l, T pad_r, T stride, T dilation,
        bool ceil_mode) {
    T outputSize = div_rtn<T>(
        inputSize + pad_l + pad_r - dilation * (kernelSize - 1) - 1 +
        (ceil_mode ? stride - 1 : 0), stride) + 1;
    if (pad_l) {
        // ensure that the last pooling starts inside the image
        // needed to avoid problems in ceil mode
        if ((outputSize - 1) * stride >= inputSize + pad_l)
          --outputSize;
    }
    return outputSize;
}

template<typename T>
static inline T pooling_output_shape(
      T inputSize, T kernelSize, T pad, T stride, T dilation, bool ceil_mode) {
    return pooling_output_shape_pad_lr(
        inputSize, kernelSize, pad, pad, stride, dilation, ceil_mode);
}

static inline void pool2d_shape_check(
  const at::Tensor& input,
  int kH, int kW, int dH, int dW, int padH, int padW, int dilationH, int dilationW,
  int64_t nInputPlane,
  int64_t inputHeight, int64_t inputWidth,
  int64_t outputHeight, int64_t outputWidth)
{
  const int64_t ndim = input.ndimension();
  const int64_t nOutputPlane = nInputPlane;

  TORCH_CHECK(kW > 0 && kH > 0,
              "kernel size should be greater than zero, but got ",
              "kH: ", kH, " kW: ", kW);
  TORCH_CHECK(dW > 0 && dH > 0,
              "stride should be greater than zero, but got "
              "dH: ", dH, " dW: ", dW);
  TORCH_CHECK(dilationH > 0 && dilationW > 0,
              "dilation should be greater than zero, but got ",
              "dilationH: ", dilationH, " dilationW: ", dilationW);

  TORCH_CHECK(input.numel() > 0 && (ndim == 3 || ndim == 4),
              "non-empty 3D or 4D input tensor expected but got ndim: ", ndim);
  TORCH_CHECK(kW/2 >= padW && kH/2 >= padH,
              "pad should be smaller than half of kernel size, but got ",
              "padW = ", padW, ", padH = ", padH, ", kW = ", kW, ", kH = ", kH);

  TORCH_CHECK(outputWidth >= 1 && outputHeight >= 1,
              "Given input size: (",
              nInputPlane, "x", inputHeight, "x", inputWidth, "). ",
              "Calculated output size: (",
              nOutputPlane, "x", outputHeight, "x", outputWidth, "). ",
              "Output size is too small");
}

template <typename scalar_t, typename accscalar_t>
__global__ void conv_rectify_cuda_frame(
    const int nthreads,
    //const scalar_t* const bottom_data,
    const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    scalar_t* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    //const int c = (index / pooled_width / pooled_height) % channels;
    //const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    //const scalar_t* const bottom_slice = bottom_data + (n * channels + c) * height * width;
    //accscalar_t aveval = bottom_slicep[];//accscalar_t(0);
    //for (int h = hstart; h < hend; ++h) {
    //  for (int w = wstart; w < wend; ++w) {
    //    aveval += bottom_slice[h * width + w];
    //  }
    //}
    accscalar_t mul_factor = accscalar_t(1.0) * pool_size / ((hend - hstart) * (wend - wstart));
    top_data[index] = ScalarConvert<accscalar_t, scalar_t>::to(top_data[index] * mul_factor);
  }
}

void conv_rectify_cuda_tempalte(
  at::Tensor& output,
  const at::Tensor& input_,
  at::IntArrayRef kernel_size,
  at::IntArrayRef stride,
  at::IntArrayRef padding,
  bool ceil_mode)
{
  //at::TensorArg output_arg{ output, "output", 1 };
  //at::TensorArg input_arg{ input_, "input_", 2 };

  //checkAllSameGPU("avg_pool2d_out_cuda", {output_arg, input_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
    "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "avg_pool2d: padding must either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK((input_.ndimension() == 3 || input_.ndimension() == 4),
    "non-empty 3D or 4D (batch mode) tensor expected for input");

  const int64_t nbatch = input_.ndimension() == 4 ? input_.size(-4) : 1;
  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);

  pool2d_shape_check(
    input_,
    kH, kW, dH, dW, padH, padW, 1, 1,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight, outputWidth);

  at::Tensor input = input_.contiguous();

  //output.resize_({nbatch, nInputPlane, outputHeight, outputWidth});

  const int32_t count = safe_downcast<int32_t, int64_t>(output.numel());
  const uint32_t  num_threads = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);
  const uint32_t num_blocks = at::cuda::ATenCeilDiv<uint32_t>(count, num_threads);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "conv_rectify_cuda_frame", ([&] {
        //using accscalar_t = acc_type<scalar_t, true>;
        scalar_t *output_data = output.data_ptr<scalar_t>();
        conv_rectify_cuda_frame<scalar_t, scalar_t>
            <<<num_blocks, num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            count,
                nbatch,
                nInputPlane,
                inputHeight, inputWidth,
                outputHeight, outputWidth,
                kH, kW,
                dH, dW,
                padH, padW,
                output_data);
  }));


  AT_CUDA_CHECK(cudaGetLastError());

  //if (input.ndimension() == 3) {
  //  output.resize_({nInputPlane, outputHeight, outputWidth});
  //}
}

at::Tensor CONV_RECTIFY_CUDA(
  at::Tensor& output,
  const at::Tensor& input,
  at::IntArrayRef kernel_size,
  at::IntArrayRef stride,
  at::IntArrayRef padding,
  bool ceil_mode) {
  //at::Tensor output = at::empty({0}, input.options());
  conv_rectify_cuda_tempalte(
    output,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode);
  //return output;
}

at::Tensor RECTIFIED_CONVOLUTION_FORWARD(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    at::IntArrayRef output_padding,
    int64_t groups) {
  auto& ctx = at::globalContext();

  //at::Tensor output = at::cudnn_convolution(
  at::Tensor output = at::convolution(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    false,
    output_padding,
    groups);
    //ctx.benchmarkCuDNN(),
    //ctx.deterministicCuDNN());

  const int64_t kH = weight.size(-2);
  const int64_t kW = weight.size(-1);
  at::IntArrayRef kernel_size = at::IntArrayRef{kH, kW};

  CONV_RECTIFY_CUDA(output, input, kernel_size, stride, padding, false);
  return output;
}

//std::vector<at::Tensor> RECTIFIED_CONVOLUTION_BACKWARD(
//    at::Tensor& grad_output,
//    const at::Tensor& input,
//    const at::Tensor& weight,
//    at::IntArrayRef stride,
//    at::IntArrayRef padding,
//    at::IntArrayRef dilation,
//    int64_t groups
//  ) {
//  const int64_t kH = weight.size(-2);
//  const int64_t kW = weight.size(-1);
//  at::IntArrayRef kernel_size = at::IntArrayRef{kH, kW};
//
//  // rectify
//  CONV_RECTIFY_CUDA(grad_output, input, kernel_size, stride, padding, false);
//
//  auto& ctx = at::globalContext();
//  at::Tensor grad_input = at::cudnn_convolution_backward_input(
//    input.size(),
//    grad_output,
//    weight,
//    padding,
//    stride,
//    dilation,
//    groups,
//    ctx.benchmarkCuDNN(),
//    ctx.deterministicCuDNN());
//
//  at::Tensor grad_weight = at::cudnn_convolution_backward_weight(
//    weight.size(),
//    grad_output,
//    input,
//    padding,
//    stride,
//    dilation,
//    groups,
//    ctx.benchmarkCuDNN(),
//    ctx.deterministicCuDNN());
//
//  return {grad_input, grad_weight};
//}
