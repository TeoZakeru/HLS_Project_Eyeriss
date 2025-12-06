#include <iostream>
#include <cstdlib> // for rand(), srand()
#include <ctime>   // for time()
#include "configs.h"
#include "systolic_array.h" // To get the 'hw' function prototype

/**
 * @brief Software "golden" reference model.
 * This function *must* implement the exact (and unusual) logic 
 * from the Python code's __PsumTransport__ and __Conv1d__ to be a valid test.
 *
 * Logic: output[j_out][k] = sum( dot(Picture[i_filter + j_out][k : k+5], FilterWeight[i_filter]) )
 * ...for i_filter in 0..4
 */
void conv2d_sw(const data_t Picture[PIC_HEIGHT][PIC_WIDTH],
               const data_t FilterWeight[FLT_HEIGHT][FLT_WIDTH],
               data_t Output[OUT_HEIGHT][OUT_WIDTH])
{
    // output[j_out][k]
    SW_LOOP_J_OUT: for (int j_out = 0; j_out < OUT_HEIGHT; j_out++) {
        SW_LOOP_K: for (int k = 0; k < OUT_WIDTH; k++) { // k is the 1D conv output index
            
            data_t partial_sum = 0;
            
            // Sum over filter rows
            // sum( ... for i_filter in 0..4 )
            SW_LOOP_I_FILT: for (int i_filter = 0; i_filter < FLT_HEIGHT; i_filter++) {
                
                // Get the correct image row: Picture[i_filter + j_out]
                int i_img = i_filter + j_out;
                
                if (i_img < PIC_HEIGHT) {
                    // Perform the dot product: dot(Picture[...][k : k+5], FilterWeight[i_filter])
                    data_t dot_prod = 0;
                    SW_LOOP_DOT: for (int i_flt_col = 0; i_flt_col < FLT_WIDTH; i_flt_col++) {
                        // k is the sliding window offset
                        if (k + i_flt_col < PIC_WIDTH) {
                            dot_prod += Picture[i_img][k + i_flt_col] * FilterWeight[i_filter][i_flt_col];
                        }
                    }
                    partial_sum += dot_prod;
                }
            }
            Output[j_out][k] = partial_sum;
        }
    }
    
    // Apply ReLU
    SW_RELU_I: for (int i=0; i < OUT_HEIGHT; ++i) {
        SW_RELU_J: for (int j=0; j < OUT_WIDTH; ++j) {
            if (Output[i][j] < 0) {
                Output[i][j] = 0;
            }
        }
    }
}

// Helper function to print matrices
void print_matrix(const char* name, const data_t* matrix, int rows, int cols) {
    std::cout << "--- " << name << " (" << rows << "x" << cols << ") ---" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout.width(6); // %6d
            std::cout << matrix[i * cols + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    // Python: np.random.seed(2)
    srand(2); 

    // Allocate memory (static to avoid stack overflow)
    static data_t picture[PIC_HEIGHT][PIC_WIDTH];
    static data_t filter_weight[FLT_HEIGHT][FLT_WIDTH];
    static data_t output_hw[OUT_HEIGHT][OUT_WIDTH];
    static data_t output_sw[OUT_HEIGHT][OUT_WIDTH];

    // Initialize picture: np.random.randint(0, 255, (18, 18), dtype=int)
    for (int i = 0; i < PIC_HEIGHT; ++i) {
        for (int j = 0; j < PIC_WIDTH; ++j) {
            picture[i][j] = rand() % 255;
        }
    }
    
    // Initialize filter: np.random.randint(-5, 6, (5, 5), dtype=int)
    for (int i = 0; i < FLT_HEIGHT; ++i) {
        for (int j = 0; j < FLT_WIDTH; ++j) {
            filter_weight[i][j] = (rand() % 11) - 5; // -5 to +5
        }
    }

    std::cout << "Running Vitis HLS Testbench..." << std::endl;
     print_matrix("Input Image", (data_t*)picture, PIC_HEIGHT, PIC_WIDTH);
     print_matrix("Filter Matrix", (data_t*)filter_weight, FLT_HEIGHT, FLT_WIDTH);

    // Run the hardware function
    eyeriss_conv2d_hw(picture, filter_weight, output_hw);

    // Run the software "golden" reference
    conv2d_sw(picture, filter_weight, output_sw);

    // Print outputs (optional, can be noisy)
     print_matrix("Output Matrix (HW)", (data_t*)output_hw, OUT_HEIGHT, OUT_WIDTH);
     print_matrix("Output Matrix (SW)", (data_t*)output_sw, OUT_HEIGHT, OUT_WIDTH);

    // Compare HW and SW results
    int errors = 0;
    for (int i = 0; i < OUT_HEIGHT; ++i) {
        for (int j = 0; j < OUT_WIDTH; ++j) {
            if (output_hw[i][j] != output_sw[i][j]) {
                errors++;
                if (errors < 10) { // Print first 10 errors
                    std::cout << "Mismatch at [" << i << "][" << j << "]: "
                              << "HW=" << output_hw[i][j] << ", SW=" << output_sw[i][j] << std::endl;
                }
            }
        }
    }

    if (errors == 0) {
        std::cout << ">>> SUCCESS: HW and SW results match!" << std::endl;
        return 0; // Return 0 for success
    } else {
        std::cout << ">>> FAILURE: Found " << errors << " mismatches." << std::endl;
        return 1; // Return 1 for failure
    }
}
