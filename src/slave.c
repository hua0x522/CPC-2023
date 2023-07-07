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
} spmvPara;

#define dataBufferSize 2000
__thread_local crts_rply_t DMARply = 0;
__thread_local unsigned int DMARplyCount = 0;
__thread_local double p[dataBufferSize] __attribute__ ((aligned(64)));
__thread_local double z[dataBufferSize] __attribute__ ((aligned(64)));

// __thread_local double csr_val[dataBufferSize] __attribute__ ((aligned(64)));
// __thread_local int csr_col[dataBufferSize] __attribute__ ((aligned(64)));
// __thread_local int csr_row[dataBufferSize] __attribute__ ((aligned(64)));
// __thread_local double vec[dataBufferSize] __attribute__ ((aligned(64)));
// __thread_local double result[dataBufferSize] __attribute__ ((aligned(64)));

void slave_example(Para* para){
	Para slavePara;
	//接收结构体数据
	CRTS_dma_iget(&slavePara, para, sizeof(Para), &DMARply);
	DMARplyCount++;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);
	double beta = slavePara.beta;
	int cells = slavePara.cells;
	
	//计算从核接收数组数据长度和接收位置
	int len = cells / 64;
	int rest = cells % 64;
	int addr;
	if(CRTS_tid < rest){
		len++;
		addr = CRTS_tid * len;
	}else{
		addr = CRTS_tid * len + rest;
	}
	//接收数组数据
	CRTS_dma_iget(&p, slavePara.p + addr, len * sizeof(double), &DMARply);
	CRTS_dma_iget(&z, slavePara.z + addr, len * sizeof(double), &DMARply);
	DMARplyCount += 2;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);
			
	//计算
	int i = 0;
	for(; i < len; i++){
		p[i] = z[i] + beta * p[i];
	}
	//传回计算结果
	CRTS_dma_iput(slavePara.p+addr, &p, len * sizeof(double), &DMARply);
	DMARplyCount++;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);
}

void slave_spmv(spmvPara* para) {
	spmvPara slavePara;

	CRTS_dma_iget(&slavePara, para, sizeof(spmvPara), &DMARply);
	DMARplyCount++;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);

	struct CsrMatrix csr_matrix = slavePara.csr_matrix;
	double* vec = slavePara.vec; 
	double* result = slavePara.result;

	int row_len = csr_matrix.rows / 64;
	int rest = csr_matrix.rows % 64;
	int row_off;
	if (CRTS_tid < rest) {
		row_len++;
		row_off = CRTS_tid * row_len;
	} else {
		row_off = CRTS_tid * row_len + rest;
	}

	for(int i = row_off; i < row_off + row_len; i++) {
        int start = csr_matrix.row_off[i];
        int num = csr_matrix.row_off[i+1] - csr_matrix.row_off[i];
        double temp = 0;
        for(int j = 0; j < num; j++) {                      
            temp += vec[csr_matrix.cols[start+j]] * csr_matrix.data[start+j]; 
        }
        result[i]=temp;
    }
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

	int row_len = csr_matrix.rows / 64;
	int rest = csr_matrix.rows % 64;
	int row_off;
	if (CRTS_tid < rest) {
		row_len++;
		row_off = CRTS_tid * row_len;
	} else {
		row_off = CRTS_tid * row_len + rest;
	}

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