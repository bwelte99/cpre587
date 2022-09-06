#include <iostream>

#include "../Utils.h"
#include "../Types.h"
#include "Layer.h"
#include "MaxPooling.h"


namespace ML {
    // --- Begin Student Code ---

    // Compute the convultion for the layer data
    void MaxPoolingLayer::computeNaive(const LayerData &dataIn) const {
        printf("pooling layer: We in here\n");


    }
    void MaxPoolingLayer::computeTiled(const LayerData &dataIn) const {
        printf("pooling layer: I like tile floors\n");


    }
        void MaxPoolingLayer::computeThreaded(const LayerData &dataIn) const {
        printf("pooling layer: Kirby's epic yarn\n");


    }
        void MaxPoolingLayer::computeSIMD(const LayerData &dataIn) const {
        printf("pooling layer: Brrrrrrt\n");


    }
};