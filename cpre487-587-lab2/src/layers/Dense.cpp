#include <iostream>

#include "../Utils.h"
#include "../Types.h"
#include "Layer.h"
#include "Dense.h"


namespace ML {
    // --- Begin Student Code ---

    // Compute the output for the dense layer
    void DenseLayer::computeNaive(const LayerData &dataIn) const {

        //std::cout << "\tDEBUG: dense layer" << std::endl;

        //read in dense layer parameters
        const LayerParams denseInputParams = this->getInputParams();
        const LayerParams denseOutputParams = this->getOutputParams();
        const LayerParams denseWeightParams = this->getWeightParams();
        const LayerParams denseBiasParams = this->getBiasParams();

        //Read in convolution input data (NOTE: Array3D_fp32 = float***)
        Array1D_fp32 denseInput = dataIn.getData<fp32*>();

        //assign the outData LayerData object's data pointer to a floating point array
        Array1D_fp32 denseOutput = this->getOutputData().getData<fp32*>();
        
        //assign the dense layer weights to a floating point array
        Array2D_fp32 denseWeights = this->getWeightData().getData<fp32**>();
        Array1D_fp32 denseBiases = this->getBiasData().getData<fp32*>();

        //Store Input dimensions
        const int H = denseInputParams.dims[0];  //number of inputs

        //Store Output dimensions
        const int W = denseOutputParams.dims[0]; //number of outputs

        //Store Weight dimensions
        const int P = denseWeightParams.dims[0];    //number of weights in one weight vector (should equal the number of inputs)
        const int Q = denseWeightParams.dims[1];    //number of separate weight vectors (should be one for each output)

        /*
        std::cout << "\t DEBUG: H = " << H << std::endl;
        std::cout << "\t DEBUG: W = " << W << std::endl;
        std::cout << "\t DEBUG: P = " << P << std::endl;
        std::cout << "\t DEBUG: Q = " << Q << std::endl;
        */

        for (int w = 0; w < W; w++){
            float sum = 0;

            //calculate dot product of input and wth set of weights
            for (int h = 0; h < H; h++){
                sum += denseInput[h] * denseWeights[w][h];
            }

            //add bias
            sum += denseBiases[w];
            
            //ReLU
            if (sum < 0.0) {
                sum = 0.0;
            }

            denseOutput[w] = sum;
        }
    }


    // Compute the convolution using threads
    void DenseLayer::computeThreaded(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }


    // Compute the convolution using a tiled approach
    void DenseLayer::computeTiled(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }


    // Compute the convolution using SIMD
    void DenseLayer::computeSIMD(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }
};