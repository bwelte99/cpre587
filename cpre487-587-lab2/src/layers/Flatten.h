#pragma once

#include "Layer.h"
#include "../Utils.h"
#include "../Types.h"

namespace ML {
    class FlattenLayer : public Layer {
        public:
            FlattenLayer(const LayerParams inParams, const LayerParams outParams)
                : Layer(inParams, outParams, LayerType::FLATTEN) {};

            // Allocate all resources needed for the layer & Load all of the required data for the layer
            template<typename T>
            void allocateLayer() {
                Layer::allocateOutputBuffer<Array3D<T>>();
            }

            // Fre all resources allocated for the layer
            template<typename T>
            void freeLayer() {
                Layer::freeOutputBuffer<Array3D<T>>();
            }

            // Virtual functions
            virtual void computeNaive(const LayerData &dataIn) const override;
            virtual void computeThreaded(const LayerData &dataIn) const override;
            virtual void computeTiled(const LayerData &dataIn) const override;
            virtual void computeSIMD(const LayerData &dataIn) const override;
    };

}