#include <slave.h>
#include "pcg_def.h"

#include <crts.h>

typedef struct{
	double *p;
	double *z;
	double beta;
	int cells;
} Para;

typedef struct{
	struct CsrMatrix csr_matrix;
	double* vec;
	double* result;
	double* val;
	int* row_off;
    int* val_off;
} spmvPara;

#define dataBufferSize 5000
__thread_local crts_rply_t DMARply = 0;
__thread_local unsigned int DMARplyCount = 0;

__thread_local double csr_val[dataBufferSize] __attribute__ ((aligned(64)));
__thread_local int csr_col[dataBufferSize] __attribute__ ((aligned(64)));
__thread_local int csr_row[dataBufferSize] __attribute__ ((aligned(64)));
__thread_local double result[dataBufferSize] __attribute__ ((aligned(64)));


void slave_spmv(spmvPara* para) {
	spmvPara slavePara;

	CRTS_dma_iget(&slavePara, para, sizeof(spmvPara), &DMARply);
	DMARplyCount++;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);

	struct CsrMatrix csr_matrix = slavePara.csr_matrix;
	double* vec = slavePara.vec; 

	int row_off = slavePara.row_off[CRTS_tid];
	int row_len = slavePara.row_off[CRTS_tid+1] - row_off;
	int val_off = slavePara.val_off[CRTS_tid];
	int val_len = slavePara.val_off[CRTS_tid+1] - val_off;

	printf("row_off: %d, row_len: %d, val_off: %d, val_len: %d\n", row_off, row_len, val_off, val_len);

	CRTS_dma_iget(&csr_row, csr_matrix.row_off + row_off, (row_len+1) * sizeof(int), &DMARply);
	CRTS_dma_iget(&csr_col, csr_matrix.cols + val_off, val_len * sizeof(int), &DMARply);
	CRTS_dma_iget(&csr_val, csr_matrix.data + val_off, val_len * sizeof(double), &DMARply);
	DMARplyCount += 3;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);

	for(int i = 0; i < row_len; i++) {
		int start = csr_row[i];
        int num = csr_row[i+1] - csr_row[i];
        double temp = 0;
        for(int j = 0; j < num; j++) {                      
            temp += vec[csr_col[start+j-val_off]] * csr_val[start+j-val_off]; 
        }
        result[i]=temp;
    }

	CRTS_dma_iput(slavePara.result+row_off, &result, row_len * sizeof(double), &DMARply);
	DMARplyCount++;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);
}

void slave_pre_spmv(spmvPara* para) {
	spmvPara slavePara;

	CRTS_dma_iget(&slavePara, para, sizeof(spmvPara), &DMARply);
	DMARplyCount++;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);

	struct CsrMatrix csr_matrix = slavePara.csr_matrix;
	double* vec = slavePara.vec; 
	double* result = slavePara.result;
	double* val = slavePara.val;

	int row_off = slavePara.row_off[CRTS_tid];
	int row_len = slavePara.row_off[CRTS_tid+1] - row_off;

	for(int i = row_off; i < row_off + row_len; i++) {
        int start = csr_matrix.row_off[i];
        int num = csr_matrix.row_off[i+1] - csr_matrix.row_off[i];
        double temp = 0;
        for(int j = 0; j < num; j++) {                      
            temp += vec[csr_matrix.cols[start+j]] * val[start+j]; 
        }
        result[i]=temp;
    }
}

/*
void slave_spmv(spmvPara* para) {
	spmvPara slavePara;

	CRTS_dma_iget(&slavePara, para, sizeof(spmvPara), &DMARply);
	DMARplyCount++;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);

	struct CsrMatrix csr_matrix = slavePara.csr_matrix;

	int row_len = csr_matrix.rows / 64;
	int rest = csr_matrix.rows % 64;
	int row_off;
	if (CRTS_tid < rest) {
		row_len++;
		row_off = CRTS_tid * row_len;
	} else {
		row_off = CRTS_tid * row_len + rest;
	}

	CRTS_dma_iget(&csr_row, csr_matrix.row_off + row_off, (row_len + 1) * sizeof(int), &DMARply);
	CRTS_dma_iget(&vec, slavePara.vec, slavePara.cells * sizeof(double), &DMARply);
	DMARplyCount += 2;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);

	int off = csr_row[0];
	int len = csr_row[row_len] - csr_row[0];

	CRTS_dma_iget(&csr_val, csr_matrix.data + off, len * sizeof(double), &DMARply);
	CRTS_dma_iget(&csr_col, csr_matrix.cols + off, len * sizeof(int), &DMARply);
	DMARplyCount += 2;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);

	for (int i = 0; i < row_len; i++) {
		int start = csr_row[i];
		int num = csr_row[i+1] - csr_row[i];
		double temp = 0;
		for (int j = 0; j < num; j++) {
			temp += vec[csr_col[start+j-off]] * csr_val[start+j-off];
		}
		result[i] = temp;
	}

	CRTS_dma_iput(slavePara.result+row_off, &result, row_len * sizeof(double), &DMARply);
	DMARplyCount++;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);
}
*/