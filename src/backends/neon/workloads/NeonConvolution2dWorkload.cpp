//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonConvolution2dWorkload.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <neon/workloads/NeonWorkloadUtils.hpp>

#include <arm_compute/runtime/NEON/functions/NEConvolutionLayer.h>
#include <arm_compute/runtime/Scheduler.h>

#include <armnn/Types.hpp>
#include <Half.hpp>
#include <exception>

namespace armnn
{
int MethodToInt(arm_compute::ConvolutionMethod method) { 
    // 0 : Default
    // 1 : Gemm_Direct
    // 2 : General
    // 3 : Winograd
    switch (method)
    {
    case arm_compute::ConvolutionMethod::GEMM_CONV2D:
        return 1;
    case arm_compute::ConvolutionMethod::GEMM:
        return 2;
    case arm_compute::ConvolutionMethod::WINOGRAD:
        return 3;
    case arm_compute::ConvolutionMethod::DIRECT:
    case arm_compute::ConvolutionMethod::FFT:
    default:
        printf("!!!!! Direct, FFT Found!!");
        return -1;
    }
}
using namespace armcomputetensorutils;

arm_compute::Status NeonConvolution2dWorkloadValidate(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      const Convolution2dDescriptor& descriptor,
                                                      const TensorInfo& weights,
                                                      const Optional<TensorInfo>& biases,
                                                      bool isFastMathEnabled,
                                                      const ActivationDescriptor* activationDescriptor)
{
    // arm_compute::NEConvolutionLayer supports both const and non const
    // weights. However, in the case of non const weights we'd have to call
    // prepare or configure for each inference which we're not setup to do just yet.
    if (!weights.IsConstant())
    {
        return arm_compute::Status{arm_compute::ErrorCode::RUNTIME_ERROR,
                                   "ArmNN NeonConvolution2dWorkload does not support non constant weights."};
    }

    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);
    arm_compute::TensorInfo aclWeightsInfo = BuildArmComputeTensorInfo(weights, descriptor.m_DataLayout);
    aclWeightsInfo.set_are_values_constant(weights.IsConstant());

    const arm_compute::Size2D aclDilationInfo = BuildArmComputeSize2D(descriptor.m_DilationX,
                                                                      descriptor.m_DilationY);

    arm_compute::TensorInfo aclBiasesInfo;
    arm_compute::TensorInfo *optionalAclBiasesInfo = nullptr;

    if (descriptor.m_BiasEnabled)
    {
        ARMNN_ASSERT(biases.has_value());
        // Same for bias as weights. We don't currently support non const.
        if (!biases.value().IsConstant())
        {
            return arm_compute::Status{arm_compute::ErrorCode::RUNTIME_ERROR,
                                       "ArmNN NeonConvolution2dWorkload does not support non constant bias."};
        }
        aclBiasesInfo = BuildArmComputeTensorInfo(biases.value(), descriptor.m_DataLayout);
        aclBiasesInfo.set_are_values_constant(biases.value().IsConstant());
        optionalAclBiasesInfo = &aclBiasesInfo;
    }

    arm_compute::PadStrideInfo layerInfo = BuildArmComputePadStrideInfo(descriptor);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertActivationDescriptorToAclActivationLayerInfo(
            activationDescriptor);

    return arm_compute::NEConvolutionLayer::validate(&aclInputInfo,
                                                     &aclWeightsInfo,
                                                     optionalAclBiasesInfo,
                                                     &aclOutputInfo,
                                                     layerInfo,
                                                     arm_compute::WeightsInfo(),
                                                     aclDilationInfo,
                                                     activationInfo,
                                                     isFastMathEnabled);
}

NeonConvolution2dWorkload::NeonConvolution2dWorkload(
    const Convolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info,
    std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager,
    const bool isFastMathEnabled)
    : NeonBaseWorkload<Convolution2dQueueDescriptor>(descriptor, info)
{
        using arm_compute::NEConvolutionLayer;

    uint32_t numInputs = m_Data.m_Parameters.m_BiasEnabled ? 3: 2;
    m_Data.ValidateInputsOutputs("NeonConvolution2dWorkload", numInputs, 1);

    arm_compute::ITensor& input = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    m_KernelTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_KernelTensor, m_Data.m_Weight->GetTensorInfo(), m_Data.m_Parameters.m_DataLayout);
    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        m_BiasTensor = std::make_unique<arm_compute::Tensor>();
        BuildArmComputeTensor(*m_BiasTensor, m_Data.m_Bias->GetTensorInfo(), m_Data.m_Parameters.m_DataLayout);
    }

    arm_compute::PadStrideInfo padStrideInfo = BuildArmComputePadStrideInfo(m_Data.m_Parameters);

    const arm_compute::Size2D aclDilationInfo = BuildArmComputeSize2D(m_Data.m_Parameters.m_DilationX,
                                                                      m_Data.m_Parameters.m_DilationY);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);
    
    // 0 : Default
    // 1 : Gemm_Direct
    // 2 : General
    // 3 : Winograd
    for (int conv_i = 1; conv_i <= 3; conv_i++) { 
        arm_compute::Scheduler::get().get_convolution_kernel(); 
        arm_compute::Scheduler::get().set_conv_method(conv_i); 
        auto tmp_layer = std::make_unique<arm_compute::NEConvolutionLayer>();
        tmp_layer->configure(&input,
                                    m_KernelTensor.get(),
                                    m_BiasTensor.get(),
                                    &output,
                                    padStrideInfo,
                                    arm_compute::WeightsInfo(),
                                    aclDilationInfo,
                                    activationInfo,
                                    isFastMathEnabled);
        int method = MethodToInt(tmp_layer->get_convolution_method(input.info(),
                                                    m_KernelTensor->info(),
                                                    output.info(),
                                                    padStrideInfo,
                                                    arm_compute::WeightsInfo(),
                                                    aclDilationInfo,
                                                    activationInfo,
                                                    isFastMathEnabled));
        std::cout << "Create Convolution : " << method << " " << conv_i << std::endl;
        if (method == conv_i) { 
            auto kernels = arm_compute::Scheduler::get().get_convolution_kernel(); 
            std::cout << "kernels size : " << kernels.size() << " ()\n";
            for (int k_i = 0; k_i < kernels.size(); k_i++) { 
                std::cout << "\t" << kernels[k_i] << " ()\n";
                switch (conv_i) { 
                case 1:
                    m_DirectConvolutionKernelName.push_back(kernels[k_i]);
                    break;
                case 2:
                    m_GeneralConvolutionKernelName.push_back(kernels[k_i]);
                    break;
                case 3:
                    m_WinogradConvolutionKernelName.push_back(kernels[k_i]);
                    break;
                default:
                    printf("error!!!!\n\n\n");
                    throw "error!";
                }
            }
        }
    }

    int method_id = 0;
    std::string kernel_name = "";

    if (arm_compute::Scheduler::get().conv_method_callback != nullptr) { 
        auto data = arm_compute::Scheduler::get().conv_method_callback(m_DirectConvolutionKernelName, 
                                                                        m_GeneralConvolutionKernelName,
                                                                        m_WinogradConvolutionKernelName);
        method_id = std::get<0>(data);
        int kernel_id = std::get<1>(data);

        switch (method_id) { 
            case 1:
                kernel_name = m_DirectConvolutionKernelName[kernel_id];
                break;
            case 2:
                kernel_name = m_GeneralConvolutionKernelName[kernel_id];
                break;
            case 3:
                kernel_name = m_WinogradConvolutionKernelName[kernel_id];
                break;
        }
    }
            
    arm_compute::Scheduler::get().set_conv_method(method_id); 
    arm_compute::Scheduler::get().set_gemm_kernelOps(kernel_name);
    auto mm_ConvolutionLayer = std::make_unique<arm_compute::NEConvolutionLayer>(memoryManager);
    mm_ConvolutionLayer->configure(&input,
                                m_KernelTensor.get(),
                                m_BiasTensor.get(),
                                &output,
                                padStrideInfo,
                                arm_compute::WeightsInfo(),
                                aclDilationInfo,
                                activationInfo,
                                isFastMathEnabled);

    m_ConvolutionMethod =
        mm_ConvolutionLayer->get_convolution_method(input.info(),
                                                 m_KernelTensor->info(),
                                                 output.info(),
                                                 padStrideInfo,
                                                 arm_compute::WeightsInfo(),
                                                 aclDilationInfo,
                                                 activationInfo,
                                                 isFastMathEnabled);


    // Add details for profiling output
    WorkloadInfo detailsInfo;

    detailsInfo.m_InputTensorInfos = info.m_InputTensorInfos;
    detailsInfo.m_OutputTensorInfos = info.m_OutputTensorInfos;
    detailsInfo.m_WeightsTensorInfo = armnn::Optional<armnn::TensorInfo>(descriptor.m_Weight->GetTensorInfo());
    detailsInfo.m_ConvolutionMethod = armnn::Optional<std::string>(GetConvolutionMethodString(m_ConvolutionMethod));
    if (descriptor.m_Parameters.m_BiasEnabled)
    {
        detailsInfo.m_BiasTensorInfo = armnn::Optional<armnn::TensorInfo>(descriptor.m_Bias->GetTensorInfo());
    }

    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonConvolution2dWorkload_Construct",
                                         descriptor.m_Parameters,
                                         detailsInfo,
                                         GetGuid());

    m_ConvolutionLayer.reset(mm_ConvolutionLayer.release());

    ARMNN_ASSERT(m_ConvolutionLayer);

    InitializeArmComputeTensorData(*m_KernelTensor, m_Data.m_Weight);

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        InitializeArmComputeTensorData(*m_BiasTensor, m_Data.m_Bias);
    }

    m_ConvolutionLayer->prepare();
    FreeTensorIfUnused(m_KernelTensor);
    FreeTensorIfUnused(m_BiasTensor);
}

void NeonConvolution2dWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonConvolution2dWorkload_Execute", this->GetGuid());
    m_ConvolutionLayer->run();
}

arm_compute::ConvolutionMethod NeonConvolution2dWorkload::GetConvolutionMethod() const
{
    return m_ConvolutionMethod;
}

} //namespace armnn
