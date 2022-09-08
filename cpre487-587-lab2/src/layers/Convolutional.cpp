#include <iostream>

#include "../Utils.h"
#include "../Types.h"
#include "Layer.h"
#include "Convolutional.h"


namespace ML {
    // --- Begin Student Code ---

    // Compute the convolution for the layer data
    void ConvolutionalLayer::computeNaive(const LayerData &dataIn) const {

        std::cout << "conv layer: DEBUG" << std::endl;

        //read in convolution parameters
        const LayerParams convInputParams = this->getInputParams();
        const LayerParams convOutputParams = this->getOutputParams();
        const LayerParams convWeightParams = this->getWeightParams();
        const LayerParams convBiasParams = this->getBiasParams();

        //Read in convolution input data (NOTE: Array3D_fp32 = float***)
        Array3D_fp32 convInput = dataIn.getData<fp32***>();

        //assign the outData LayerData object's data pointer to a floating point array
        Array3D_fp32 convOutput = this->getOutputData().getData<fp32***>();
        //sample output assignment
        convOutput[0][0][0] = 42.0;

        std::cout << "\tKernel dimensions: " << convWeightParams.dims[0] << " x " << convWeightParams.dims[1] << " x " << convWeightParams.dims[2] << std::endl;

        //Store Input dimensions
        const int H = convInputParams.dims[0];  //input height
        const int W = convInputParams.dims[1];  //input width
        const int C = convInputParams.dims[2];  //number of input fmaps

        //Store Output dimensions
        const int P = convOutputParams.dims[0];
        const int Q = convOutputParams.dims[1];

        //Store Kernel dimensions
        const int R = convWeightParams.dims[0];   //filter height
        const int S = convWeightParams.dims[1];   //filter width
        const int M = convWeightParams.dims[2];   //number of filters/output fmaps


        //TODO: assign one output pixel based on kernel dimensions
        //where are weightparams & biasparams?

        
        std::cout << "\t input elements are " << convInputParams.elementSize << " bytes"<< std::endl;
        std::cout << "\t input dims are " << convInputParams.dims[0] << " by " << convInputParams.dims[1] << " by " << convInputParams.dims[2] << std::endl;

        /*
        //attempting to print the data:
        for (int i = 0; i < H; i++){
            for (int j = 0; j < W; j++){
                printf("%1.2f ", convInput[j][i][0]);
            }
            printf("\n");
        }
        */
      
        
    }


    // Compute the convolution using threads
    void ConvolutionalLayer::computeThreaded(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }


    // Compute the convolution using a tiled approach
    void ConvolutionalLayer::computeTiled(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }


    // Compute the convolution using SIMD
    void ConvolutionalLayer::computeSIMD(const LayerData &dataIn) const {
        // TODO: Your Code Here...


    }
};