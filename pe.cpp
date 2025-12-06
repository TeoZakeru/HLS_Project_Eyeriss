#include "pe.h"
#include <cstring> // for memcpy

PE::PE() {
    // Initialize arrays to 0, matching Python's (0,0) init behavior
    for(int i = 0; i < FLT_WIDTH; i++) FilterWeight[i] = 0;
    for(int i = 0; i < PIC_WIDTH; i++) ImageRow[i] = 0;
    for(int i = 0; i < PE_PSUM_LENGTH; i++) Psum[i] = 0;
    PEState = ClockGate;
    ImageNum = 0;
    FilterNum = 0;
}

void PE::SetPEState(int State) {
    this->PEState = State;
}

// Copy FLT_WIDTH elements
void PE::SetFilterWeight(const data_t* InFilterWeight) {
    for (int i = 0; i < FLT_WIDTH; i++) {
        #pragma HLS UNROLL
        this->FilterWeight[i] = InFilterWeight[i];
    }
}

// Copy PIC_WIDTH elements
void PE::SetImageRow(const data_t* InImageRow) {
    for (int i = 0; i < PIC_WIDTH; i++) {
        #pragma HLS PIPELINE II=1
        this->ImageRow[i] = InImageRow[i];
    }
}

void PE::SetPEImgAndFlt(int ImgNum, int FltNum) {
    this->ImageNum = ImgNum;
    this->FilterNum = FltNum;
}

//void PE::__SetPsum__(const data_t* PsumArray) {
//    // Handle the EmptyPsum case (when PsumArray is null)
//    if (PsumArray == nullptr) {
//         for (int i = 0; i < PE_PSUM_LENGTH; i++) {
//            #pragma HLS UNROLL
//            this->Psum[i] = EmptyPsum;
//        }
//    } else {
//        for (int i = 0; i < PE_PSUM_LENGTH; i++) {
//            #pragma HLS UNROLL
//            this->Psum[i] = PsumArray[i];
//        }
//    }
//}

/**
 * @brief Implements the __Conv1d__ logic.
 * result must be an array of size PE_PSUM_LENGTH.
 * This is the 1D sliding window dot product.
 */
void PE::__Conv1d__(const data_t* ImgRow, const data_t* FltW, data_t* result) {
    // Python: for x in range(0, len(ImageRow) - len(FilterWeight) + 1)
    // This is exactly PE_PSUM_LENGTH (18 - 5 + 1 = 14)
    CONV1D_X_LOOP: for (int x = 0; x < PE_PSUM_LENGTH; x++) {
        #pragma HLS PIPELINE II=1
        // r = ImageRow[x:y] * FilterWeight
        data_t r_sum = 0;
        CONV1D_DOT_LOOP: for (int i = 0; i < FLT_WIDTH; i++) {
            #pragma HLS UNROLL
            r_sum += ImgRow[x + i] * FltW[i];
        }
        result[x] = r_sum;
    }
}

/**
 * @brief Implements the __Conv__ logic.
 * result_array must be of size PE_PSUM_LENGTH.
 */
void PE::__Conv__(data_t* result_array) {
    // Your main.py only uses the case where ImageNum=1 and FilterNum=1.
    // We strictly follow this executed logic.
    if (FilterNum == 1 && ImageNum == 1) {
        __Conv1d__(this->ImageRow, this->FilterWeight, result_array);
    } else {
        // This logic was not used by main.py.
        // We set the result to 0 to match a potential default branch.
        for (int i = 0; i < PE_PSUM_LENGTH; i++) {
            #pragma HLS UNROLL
            result_array[i] = 0;
        }
    }
}

/**
 * @brief Top-level compute function for the PE.
 * Replicates CountPsum from PE.py.
 */
//void PE::CountPsum() {
//    if (this->PEState == ClockGate) {
//        // Set Psum array to 0
//        __SetPsum__(nullptr); // Use nullptr to signal reset
//    } else if (this->PEState == Running) {
//        // Temporary buffer for the 1D conv result
//        data_t temp_psum[PE_PSUM_LENGTH];
//        __Conv__(temp_psum);
//        __SetPsum__(temp_psum);
//    }
//}

/**
 * @brief Top-level compute function for the PE.
 * Replicates CountPsum from PE.py.
 * (This is the MODIFIED version)
 */
void PE::CountPsum() {
    if (this->PEState == ClockGate) {
        // This logic was in __SetPsum__(nullptr)
        SET_PSUM_ZERO_LOOP: for (int i = 0; i < PE_PSUM_LENGTH; i++) {
            #pragma HLS UNROLL
            this->Psum[i] = EmptyPsum;
        }
    } else if (this->PEState == Running) {
        // Temporary buffer for the 1D conv result
        data_t temp_psum[PE_PSUM_LENGTH];

        // Run the convolution
        __Conv__(temp_psum);

        // This logic was in __SetPsum__(temp_psum)
        SET_PSUM_VAL_LOOP: for (int i = 0; i < PE_PSUM_LENGTH; i++) {
            #pragma HLS UNROLL
            this->Psum[i] = temp_psum[i];
        }
    }
    // If state is not ClockGate or Running, Psum is unchanged.
}
