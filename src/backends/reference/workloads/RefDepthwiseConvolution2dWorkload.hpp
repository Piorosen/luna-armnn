//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "RefBaseWorkload.hpp"
#include <armnn/backends/WorkloadData.hpp>
#include "Decoders.hpp"
#include "Encoders.hpp"

#include <armnn/TypesUtils.hpp>

namespace armnn
{

class RefDepthwiseConvolution2dWorkload : public RefBaseWorkload<DepthwiseConvolution2dQueueDescriptor> {
public:
    explicit RefDepthwiseConvolution2dWorkload(const DepthwiseConvolution2dQueueDescriptor &descriptor,
                                               const WorkloadInfo &info);
    void Execute() const override;
    void ExecuteAsync(WorkingMemDescriptor& workingMemDescriptor)  override;

private:
    void Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const;

};

} //namespace armnn
