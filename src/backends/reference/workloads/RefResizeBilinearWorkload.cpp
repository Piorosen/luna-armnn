//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefResizeBilinearWorkload.hpp"

#include "RefWorkloadUtils.hpp"
#include "Resize.hpp"
#include "BaseIterator.hpp"
#include "Profiling.hpp"

#include "BaseIterator.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"

namespace armnn
{

void RefResizeBilinearWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefResizeBilinearWorkload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefResizeBilinearWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefResizeBilinearWorkload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    std::unique_ptr<Decoder<float>> decoderPtr = MakeDecoder<float>(inputInfo, inputs[0]->Map());
    Decoder<float> &decoder = *decoderPtr;
    std::unique_ptr<Encoder<float>> encoderPtr = MakeEncoder<float>(outputInfo, outputs[0]->Map());
    Encoder<float> &encoder = *encoderPtr;

    Resize(decoder, inputInfo, encoder, outputInfo, m_Data.m_Parameters.m_DataLayout, armnn::ResizeMethod::Bilinear);
}

} //namespace armnn