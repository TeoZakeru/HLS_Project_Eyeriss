#include "systolic_array.h"

// --- EyerissF Class Implementation ---

EyerissF::EyerissF() {
    // Constructor is called when 'eyeriss_inst' is created.
    // The PEArray is automatically initialized by calling PE::PE() for each element.
    __InitPEs__();
}

/**
 * @brief This is the main computational logic, replicating Conv2d.
 */
void EyerissF::Conv2d(const data_t Picture[PIC_HEIGHT][PIC_WIDTH],
                      const data_t FilterWeight[FLT_HEIGHT][FLT_WIDTH],
                      int ImageNum, int FilterNum,
                      data_t ConvedArray[OUT_HEIGHT][OUT_WIDTH])
{
    int PictureColumnLength, FilterWeightColumnLength;

    // 1. __DataDeliver__
    __DataDeliver__(Picture, FilterWeight, ImageNum, FilterNum,
                    PictureColumnLength, FilterWeightColumnLength);

    // 2. __run__
    __run__();

    // 3. __PsumTransport__
    // Intermediate result buffer, before ReLU
    data_t tempResult[OUT_HEIGHT][OUT_WIDTH];
    #pragma HLS ARRAY_PARTITION variable=tempResult complete dim=2
    __PsumTransport__(tempResult, PictureColumnLength, FilterWeightColumnLength);

    // 4. Relu
    Relu2D(tempResult); // Apply ReLU in-place

    // 5. Copy to output
    // (This step is skipped, as the top-level function will write from its buffer)
    // We write the result directly to the output argument.
    for(int i=0; i<OUT_HEIGHT; ++i) {
        for(int j=0; j<OUT_WIDTH; ++j) {
            #pragma HLS PIPELINE II=1
            ConvedArray[i][j] = tempResult[i][j];
        }
    }

    // 6. __SetALLPEsState__(configs.ClockGate)
    __SetALLPEsState__(ClockGate);
}

void EyerissF::__InitPEs__() {
    // The PE constructor already initializes state.
    // This function is implicitly called by PEArray member initialization.
}

void EyerissF::__SetALLPEsState__(int State) {
    SET_STATE_LOOP_X: for (int x = 0; x < EyerissHeight; x++) {
        SET_STATE_LOOP_Y: for (int y = 0; y < EyerissWidth; y++) {
            #pragma HLS UNROLL
            this->PEArray[x][y].SetPEState(State);
        }
    }
}

void EyerissF::__SetPEsRunningState__(int PictureColumnLength, int FilterWeightColumnLength) {
    int RowLimit = PictureColumnLength + 1 - FilterWeightColumnLength; // 14
    int ColLimit = FilterWeightColumnLength; // 5

    SET_RUN_LOOP_C: for (int c = 0; c < ColLimit; c++) {
        SET_RUN_LOOP_R: for (int r = 0; r < RowLimit; r++) {
            #pragma HLS UNROLL
            // Replicates Python's 'try...except...pass' for out-of-bounds
            if (c < EyerissHeight && r < EyerissWidth) {
                this->PEArray[c][r].SetPEState(Running);
            }
        }
    }
}

void EyerissF::__SetALLPEImgNumAndFltNum__(int ImageNum, int FilterNum) {
    SET_NUM_LOOP_X: for (int x = 0; x < EyerissHeight; x++) {
        SET_NUM_LOOP_Y: for (int y = 0; y < EyerissWidth; y++) {
            #pragma HLS UNROLL
            this->PEArray[x][y].SetPEImgAndFlt(ImageNum, FilterNum);
        }
    }
}

void EyerissF::__DataDeliver__(const data_t Picture[PIC_HEIGHT][PIC_WIDTH],
                              const data_t FilterWeight[FLT_HEIGHT][FLT_WIDTH],
                              int ImageNum, int FilterNum,
                              int& PictureColumnLength, int& FilterWeightColumnLength)
{
    PictureColumnLength = PIC_HEIGHT;
    FilterWeightColumnLength = FLT_HEIGHT;

    __SetALLPEImgNumAndFltNum__(ImageNum, FilterNum);
    __SetPEsRunningState__(PictureColumnLength, FilterWeightColumnLength);

    // Load FilterWeight
    // Python: self.PEArray[ColumnELement][RowElement].SetFilterWeight(FilterWeight[ColumnELement])
    LOAD_FLT_LOOP_C: for (int c = 0; c < FilterWeightColumnLength; c++) {
        LOAD_FLT_LOOP_R: for (int r = 0; r < EyerissWidth; r++) {
            #pragma HLS UNROLL
            if (c < EyerissHeight) { // Check PE array bounds
                this->PEArray[c][r].SetFilterWeight(FilterWeight[c]);
            }
        }
    }

    // Load Picture (diagonal loading)
    // This is the exact logic from your Python script
    LOAD_PIC_LOOP_ROW: for (int pic_row = 0; pic_row < PictureColumnLength; pic_row++) {
        int DeliverinitR = 0;
        int DeliverinitH = pic_row;
        LOAD_PIC_LOOP_DIAG: for (int c = 0; c < pic_row + 1; c++) {
            #pragma HLS PIPELINE II=1
            // Replicates Python's 'try...except...pass'
            if (DeliverinitH >= 0 && DeliverinitH < EyerissHeight &&
                DeliverinitR >= 0 && DeliverinitR < EyerissWidth) {
                this->PEArray[DeliverinitH][DeliverinitR].SetImageRow(Picture[pic_row]);
            }
            DeliverinitR = DeliverinitR + 1;
            DeliverinitH = DeliverinitH - 1;
        }
    }
}

void EyerissF::__run__() {
    // This calls CountPsum() on every PE.
    // The PE's internal logic will check its own state.
    RUN_LOOP_X: for (int x = 0; x < EyerissHeight; x++) {
        RUN_LOOP_Y: for (int y = 0; y < EyerissWidth; y++) {
            #pragma HLS UNROLL
            this->PEArray[x][y].CountPsum();
        }
    }
}

void EyerissF::__PsumTransport__(data_t Result[OUT_HEIGHT][OUT_WIDTH],
                                int PictureColumnLength, int FilterWeightColumnLength)
{
    // Python logic:
    // result.append(np.sum(line, axis=0, dtype=int))
    // This translates to:
    // Result[j_out][k] = sum( PE[i_filter][j_out].Psum[k] for i_filter in 0..4 )

    int RowLimit = PictureColumnLength + 1 - FilterWeightColumnLength; // 14 (OUT_HEIGHT)
    int ColLimit = FilterWeightColumnLength; // 5 (FLT_HEIGHT)

    PSUM_LOOP_J_OUT: for (int j_out = 0; j_out < RowLimit; j_out++) { // RowElement
        PSUM_LOOP_K: for (int k = 0; k < PE_PSUM_LENGTH; k++) { // The inner 1D conv result index
            #pragma HLS PIPELINE II=1
            data_t col_sum = 0;
            PSUM_LOOP_I_FILT: for (int i_filter = 0; i_filter < ColLimit; i_filter++) { // ColumnElement
                #pragma HLS UNROLL
                // Bounds check (Python logic)
                if (i_filter < EyerissHeight && j_out < EyerissWidth) {
                     col_sum += this->PEArray[i_filter][j_out].Psum[k];
                }
            }
            Result[j_out][k] = col_sum;
        }
    }
}

void EyerissF::Relu2D(data_t array[OUT_HEIGHT][OUT_WIDTH]) {
    RELU_LOOP_I: for (int i = 0; i < OUT_HEIGHT; i++) {
        RELU_LOOP_J: for (int j = 0; j < OUT_WIDTH; j++) {
            #pragma HLS PIPELINE II=1
            if (array[i][j] < 0) {
                array[i][j] = 0;
            }
        }
    }
}


// ===================================================================
// TOP-LEVEL SYNTHESIS FUNCTION
// ===================================================================
extern "C" {
void eyeriss_conv2d_hw(
    const data_t Picture[PIC_HEIGHT][PIC_WIDTH],
    const data_t FilterWeight[FLT_HEIGHT][FLT_WIDTH],
    data_t ConvedArray[OUT_HEIGHT][OUT_WIDTH]
) {
    // Define interfaces for HLS
    // These connect to AXI buses for memory access
    #pragma HLS INTERFACE m_axi port=Picture offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=FilterWeight offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=ConvedArray offset=slave bundle=gmem0
    
    // AXI-Lite interface for control
    #pragma HLS INTERFACE s_axilite port=return

    // Local buffers (on-chip BRAMs) to hold data
    static data_t pic_buf[PIC_HEIGHT][PIC_WIDTH];
    static data_t flt_buf[FLT_HEIGHT][FLT_WIDTH];
    static data_t out_buf[OUT_HEIGHT][OUT_WIDTH];
    #pragma HLS ARRAY_PARTITION variable=flt_buf complete dim=0

    // --- Load Data ---
    // Load Picture from DRAM to BRAM
    LOAD_PIC_BRAM_I: for(int i=0; i<PIC_HEIGHT; ++i) {
        LOAD_PIC_BRAM_J: for(int j=0; j<PIC_WIDTH; ++j) {
            #pragma HLS PIPELINE II=1
            pic_buf[i][j] = Picture[i][j];
        }
    }
    // Load Filter from DRAM to BRAM
    LOAD_FLT_BRAM_I: for(int i=0; i < FLT_HEIGHT; ++i) {
        LOAD_FLT_BRAM_J: for(int j=0; j < FLT_WIDTH; ++j) {
            #pragma HLS PIPELINE II=1
            flt_buf[i][j] = FilterWeight[i][j];
        }
    }

    // Instantiate the main class.
    // 'static' ensures it's synthesized as persistent hardware logic
    static EyerissF eyeriss_inst;

    // Run the computation
    // We pass 1, 1 for ImageNum, FilterNum as in main.py
    eyeriss_inst.Conv2d(pic_buf, flt_buf, 1, 1, out_buf);
    
    // --- Write Data Back ---
    // Write result from BRAM to DRAM
    WRITE_OUT_BRAM_I: for(int i=0; i<OUT_HEIGHT; ++i) {
        WRITE_OUT_BRAM_J: for(int j=0; j<OUT_WIDTH; ++j) {
            #pragma HLS PIPELINE II=1
            ConvedArray[i][j] = out_buf[i][j];
        }
    }
}
} // extern "C"
