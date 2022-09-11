#include <iostream>

#include "../Utils.h"
#include "../Types.h"
#include "Layer.h"
#include "MaxPooling.h"


namespace ML {
    // --- Begin Student Code ---

    // Compute the convolution for the layer data
    void MaxPoolingLayer::computeNaive(const LayerData &dataIn) const {

        //std::cout << "\tDEBUG: pool layer" << std::endl;

        //read in input & output parameters
        const LayerParams poolInputParams = this->getInputParams();
        const LayerParams poolOutputParams = this->getOutputParams();

        //Read in pooling input data (NOTE: Array3D_fp32 = float***)
        Array3D_fp32 poolInput = dataIn.getData<fp32***>();

        //assign the outData LayerData object's data pointer to a floating point array
        Array3D_fp32 poolOutput = this->getOutputData().getData<fp32***>();

        //Store Input dimensions
        const int H = poolInputParams.dims[0];  //input height
        const int W = poolInputParams.dims[1];  //input width
        const int C = poolInputParams.dims[2];  //number of input channels

        //Store Output dimensions
        const int P = poolOutputParams.dims[0]; //output height
        const int Q = poolOutputParams.dims[1]; //output width
        const int M = poolOutputParams.dims[2]; //number of output channels

        //TODO: do pooling
        for (int p = 0; p < P; p++){
            for (int q = 0; q < Q; q++){
                for (int m = 0; m < M; m++){
                    //get the maximum from the 4 pixels in the pool
                    float max = poolInput[2 * p][2 * q][m];
                    if (max < poolInput[(2 * p) + 1][2 * q][m]){
                        max = poolInput[(2 * p) + 1][2 * q][m];
                    }
                    if (max < poolInput[2 * p][(2 * q) + 1][m]){
                        max = poolInput[2 * p][(2 * q) + 1][m];
                    }
                    if (max < poolInput[(2 * p) + 1][(2 * q) + 1][m]){
                        max = poolInput[(2 * p) + 1][(2 * q) + 1][m];
                    }

                    //assign the maximum to the output
                    poolOutput[p][q][m] = max;

                }
            }
        }

        //std::cout << "\t\t...done computing" << std::endl;

    }


    // Compute the pooling using threads
    void MaxPoolingLayer::computeThreaded(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }


    // Compute the pooling using a tiled approach
    void MaxPoolingLayer::computeTiled(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }


    // Compute the pooling using SIMD
    void MaxPoolingLayer::computeSIMD(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }
};