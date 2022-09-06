#include <iostream>

#include "../Utils.h"
#include "../Types.h"
#include "Layer.h"
#include "Convolutional.h"


namespace ML {
    // --- Begin Student Code ---

    // Compute the convlution for the layer data
    void ConvolutionalLayer::computeNaive(const LayerData &dataIn) const {

        std::cout << "conv layer: DEBUG" << std::endl;

        //Read in convolution input & parameters (NOTE: Array3D_fp32 = float***)
        Array3D_fp32 convInput = dataIn.getData<fp32***>();
        const LayerParams convInputParams = this->getInputParams();
        const LayerParams convOutputParams = this->getOutputParams();
        const LayerParams convWeightParams = this->getWeightParams();
        const LayerParams convBiasParams = this->getBiasParams();

        std::cout << "\tKernel dimensions: " << convWeightParams.dims[0] << " x " << convWeightParams.dims[1] << " x " << convWeightParams.dims[2] << std::endl;

        //Store dimensions
        //TODO


        //TODO: assign one output pixel based on kernel dimensions
        //where are weightparams & biasparams?




        /*
        std::cout << "\t elements are " << convParams.elementSize << " bytes"<< std::endl;
        std::cout << "\t dims are " << convParams.dims[0] << " by " << convParams.dims[1] << " by " << convParams.dims[2] << std::endl;

        //attempting to print the data:
        for (int i = 0; i < convParams.dims[0]; i++){
            for (int j = 0; j < convParams.dims[1]; j++){
                printf("%1.2f ", ***(convInput + j + (i * convParams.dims[1])));
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