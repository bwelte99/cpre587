#include <iostream>

#include "../Utils.h"
#include "../Types.h"
#include "Layer.h"
#include "Softmax.h"
#include <tgmath.h>


namespace ML {
    // --- Begin Student Code ---

    // Compute the output for the dense layer
    void SoftmaxLayer::computeNaive(const LayerData &dataIn) const {

        //std::cout << "\tDEBUG: smax layer" << std::endl;

        //read in softmax layer parameters
        const LayerParams smaxInputParams = this->getInputParams();
        const LayerParams smaxOutputParams = this->getOutputParams();

        //Read in smax input data (NOTE: Array3D_fp32 = float***)
        Array1D_fp32 smaxInput = dataIn.getData<fp32*>();

        //assign the outData LayerData object's data pointer to a floating point array
        Array1D_fp32 smaxOutput = this->getOutputData().getData<fp32*>();

        //Store Input dimensions
        const int N = smaxInputParams.dims[0];  //number of inputs (and outputs if parametrized correctly)

        /*
        std::cout << "\t DEBUG: N = " << N << std::endl;
        */

       float sum = 0;

        // calculate the sum of the exponential function for all inputs
        for (int n = 0; n < N; n++){
            sum += (float)exp(smaxInput[n]);
        }

        //calculate softmax outputs
        for (int n = 0; n < N; n++){
            smaxOutput[n] = exp(smaxInput[n]) / sum;
        }
    }


    // Compute the convolution using threads
    void SoftmaxLayer::computeThreaded(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }


    // Compute the convolution using a tiled approach
    void SoftmaxLayer::computeTiled(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }


    // Compute the convolution using SIMD
    void SoftmaxLayer::computeSIMD(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }
};