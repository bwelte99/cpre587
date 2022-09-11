#include <iostream>

#include "../Utils.h"
#include "../Types.h"
#include "Layer.h"
#include "Flatten.h"


namespace ML {
    // --- Begin Student Code ---

    // Compute the convolution for the layer data
    void FlattenLayer::computeNaive(const LayerData &dataIn) const {

        //std::cout << "\tDEBUG: flat layer" << std::endl;

        //read in layer data parameters
        const LayerParams poolInputParams = this->getInputParams();
        const LayerParams poolOutputParams = this->getOutputParams();

        //Read in input data (NOTE: Array3D_fp32 = float***)
        Array3D_fp32 flatInput = dataIn.getData<fp32***>();

        //assign the outData LayerData object's data pointer to a floating point array
        Array1D_fp32 flatOutput = this->getOutputData().getData<fp32*>();

        //Store Input dimensions
        const int H = poolInputParams.dims[0];  //input height
        const int W = poolInputParams.dims[1];  //input width
        const int C = poolInputParams.dims[2];  //number of input channels

        //Store Output dimensions
        const int P = poolOutputParams.dims[0]; //length of output array

        //TODO: do pooling
        for (int h = 0; h < H; h++){
            for (int w = 0; w < W; w++){
                for (int c = 0; c < C; c++){
                    flatOutput[c + (w * C) + (h * W * C)] = flatInput[h][w][c];
                }
            }
        }

        //std::cout << "\t\t...done computing" << std::endl;

    }


    // Compute the convolution using threads
    void FlattenLayer::computeThreaded(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }


    // Compute the convolution using a tiled approach
    void FlattenLayer::computeTiled(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }


    // Compute the convolution using SIMD
    void FlattenLayer::computeSIMD(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }
};