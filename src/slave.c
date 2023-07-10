#include <slave.h>
#include "pcg_def.h"
#include <math.h>

#include <crts.h>

typedef struct{
	double *A;
	double *B;
	double *C;
	double *D;
	double x;
	int cells;
} Para;

typedef struct{
	struct CsrMatrix csr_matrix;
	double* vec;
	double* result;
	double* val;
	int* row_off;
    int* val_off;
	int isNew;
} spmvPara;

#define dataBufferSize 10000
__thread_local crts_rply_t DMARply = 0;
__thread_local unsigned int DMARplyCount = 0;

__thread_local double csr_val[dataBufferSize] __attribute__ ((aligned(64)));
__thread_local int csr_col[dataBufferSize] __attribute__ ((aligned(64)));
__thread_local int csr_row[dataBufferSize / 5] __attribute__ ((aligned(64)));
__thread_local double result[dataBufferSize / 5] __attribute__ ((aligned(64)));

__thread_local double A[dataBufferSize / 5] __attribute__ ((aligned(64)));
__thread_local double B[dataBufferSize / 5] __attribute__ ((aligned(64)));
__thread_local double C[dataBufferSize / 5] __attribute__ ((aligned(64)));
__thread_local double D[dataBufferSize / 5] __attribute__ ((aligned(64)));

void slave_loop1(Para* para){
	Para slavePara;
	
	CRTS_dma_iget(&slavePara, para, sizeof(Para), &DMARply);
	DMARplyCount++;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);
	double beta = slavePara.x;
	int cells = slavePara.cells;
	
	int len = cells / 64;
	int rest = cells % 64;
	int addr;
	if(CRTS_tid < rest){
		len++;
		addr = CRTS_tid * len;
	}else{
		addr = CRTS_tid * len + rest;
	}
	
	CRTS_dma_iget(&A, slavePara.A + addr, len * sizeof(double), &DMARply);
	CRTS_dma_iget(&B, slavePara.B + addr, len * sizeof(double), &DMARply);
	DMARplyCount += 2;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);
			
	int i = 0;
	for(; i < len; i++){
		A[i] = B[i] + beta * A[i];
	}
	
	CRTS_dma_iput(slavePara.A+addr, &A, len * sizeof(double), &DMARply);
	DMARplyCount++;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);
}

void slave_loop2(Para* para) {
	Para slavePara;
	
	CRTS_dma_iget(&slavePara, para, sizeof(Para), &DMARply);
	DMARplyCount++;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);

	int cells = slavePara.cells;
	int len = cells / 64;
	int rest = cells % 64;
	int addr;
	if(CRTS_tid < rest){
		len++;
		addr = CRTS_tid * len;
	}else{
		addr = CRTS_tid * len + rest;
	}
	
	CRTS_dma_iget(&A, slavePara.A + addr, len * sizeof(double), &DMARply);
	CRTS_dma_iget(&B, slavePara.B + addr, len * sizeof(double), &DMARply);
	CRTS_dma_iget(&C, slavePara.C + addr, len * sizeof(double), &DMARply);
	CRTS_dma_iget(&D, slavePara.D + addr, len * sizeof(double), &DMARply);
	DMARplyCount += 4;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);
			
	int i = 0;
	for(; i < len; i++){
		A[i] = A[i] + slavePara.x * B[i];
        C[i] = C[i] - slavePara.x * D[i];
	}

	CRTS_dma_iput(slavePara.A+addr, &A, len * sizeof(double), &DMARply);
	CRTS_dma_iput(slavePara.C+addr, &C, len * sizeof(double), &DMARply);
	DMARplyCount += 2;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);
}

void slave_gsumProd(Para* para) {
	Para slavePara;
	CRTS_dma_iget(&slavePara, para, sizeof(Para), &DMARply);
	DMARplyCount++;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);

	int cells = slavePara.cells;
	double ret = 0;
	int len = cells / 64;
	int rest = cells % 64;
	int addr;
	if(CRTS_tid < rest){
		len++;
		addr = CRTS_tid * len;
	}else{
		addr = CRTS_tid * len + rest;
	}
	
	CRTS_dma_iget(&A, slavePara.A + addr, len * sizeof(double), &DMARply);
	CRTS_dma_iget(&B, slavePara.B + addr, len * sizeof(double), &DMARply);
	DMARplyCount += 2;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);
			
	int i = 0;
	for(; i < len; i++) {
		ret += A[i] * B[i];
	}

	slavePara.C[CRTS_tid] = ret;
}

void slave_gsumMag(Para* para) {
	Para slavePara;
	CRTS_dma_iget(&slavePara, para, sizeof(Para), &DMARply);
	DMARplyCount++;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);

	int cells = slavePara.cells;
	double ret = 0;
	int len = cells / 64;
	int rest = cells % 64;
	int addr;
	if(CRTS_tid < rest){
		len++;
		addr = CRTS_tid * len;
	}else{
		addr = CRTS_tid * len + rest;
	}
	
	CRTS_dma_iget(&A, slavePara.A + addr, len * sizeof(double), &DMARply);
	DMARplyCount++;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);
			
	int i = 0;
	for(; i < len; i++) {
		ret += fabs(A[i]);
	}

	slavePara.B[CRTS_tid] = ret;
}

void slave_v_dot_product(Para* para) {
	Para slavePara;
	
	CRTS_dma_iget(&slavePara, para, sizeof(Para), &DMARply);
	DMARplyCount++;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);

	int cells = slavePara.cells;
	int len = cells / 64;
	int rest = cells % 64;
	int addr;
	if(CRTS_tid < rest){
		len++;
		addr = CRTS_tid * len;
	}else{
		addr = CRTS_tid * len + rest;
	}
	
	CRTS_dma_iget(&A, slavePara.A + addr, len * sizeof(double), &DMARply);
	CRTS_dma_iget(&B, slavePara.B + addr, len * sizeof(double), &DMARply);
	DMARplyCount += 2;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);
			
	int i = 0;
	for(; i < len; i++){
		C[i] = A[i] * B[i];
	}
	
	CRTS_dma_iput(slavePara.C+addr, &C, len * sizeof(double), &DMARply);
	DMARplyCount++;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);
}

void slave_v_sub_dot_product(Para* para) {
	Para slavePara;
	
	CRTS_dma_iget(&slavePara, para, sizeof(Para), &DMARply);
	DMARplyCount++;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);

	int cells = slavePara.cells;
	int len = cells / 64;
	int rest = cells % 64;
	int addr;
	if(CRTS_tid < rest){
		len++;
		addr = CRTS_tid * len;
	}else{
		addr = CRTS_tid * len + rest;
	}
	
	CRTS_dma_iget(&A, slavePara.A + addr, len * sizeof(double), &DMARply);
	CRTS_dma_iget(&B, slavePara.B + addr, len * sizeof(double), &DMARply);
	CRTS_dma_iget(&C, slavePara.C + addr, len * sizeof(double), &DMARply);
	DMARplyCount += 3;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);
			
	int i = 0;
	for(; i < len; i++){
		D[i] = (A[i] - B[i]) * C[i];
	}
	
	CRTS_dma_iput(slavePara.D+addr, &D, len * sizeof(double), &DMARply);
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

	int row_off = slavePara.row_off[CRTS_tid];
	int row_len = slavePara.row_off[CRTS_tid+1] - row_off;
	int val_off = slavePara.val_off[CRTS_tid];
	int val_len = slavePara.val_off[CRTS_tid+1] - val_off;

	if (slavePara.isNew) {
		CRTS_dma_iget(&csr_row, csr_matrix.row_off + row_off, (row_len+1) * sizeof(int), &DMARply);
		CRTS_dma_iget(&csr_col, csr_matrix.cols + val_off, val_len * sizeof(int), &DMARply);
		DMARplyCount += 2;
		CRTS_dma_wait_value(&DMARply, DMARplyCount);
	}
	
	CRTS_dma_iget(&csr_val, csr_matrix.data + val_off, val_len * sizeof(double), &DMARply);
	DMARplyCount++;
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

	int row_off = slavePara.row_off[CRTS_tid];
	int row_len = slavePara.row_off[CRTS_tid+1] - row_off;
	int val_off = slavePara.val_off[CRTS_tid];
	int val_len = slavePara.val_off[CRTS_tid+1] - val_off;

	if (slavePara.isNew) {
		CRTS_dma_iget(&csr_row, csr_matrix.row_off + row_off, (row_len+1) * sizeof(int), &DMARply);
		CRTS_dma_iget(&csr_col, csr_matrix.cols + val_off, val_len * sizeof(int), &DMARply);
		DMARplyCount += 2;
		CRTS_dma_wait_value(&DMARply, DMARplyCount);
	}
	
	CRTS_dma_iget(&csr_val, slavePara.val + val_off, val_len * sizeof(double), &DMARply);
	DMARplyCount++;
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

void slave_pcg_init_precondition_csr(spmvPara* para) {
	spmvPara slavePara;

	CRTS_dma_iget(&slavePara, para, sizeof(spmvPara), &DMARply);
	DMARplyCount++;
	CRTS_dma_wait_value(&DMARply, DMARplyCount);

	struct CsrMatrix csr_matrix = slavePara.csr_matrix;
	int row_off = slavePara.row_off[CRTS_tid];
	int row_len = slavePara.row_off[CRTS_tid+1] - row_off;
	double* preD = slavePara.val;
	double* pre_mat_val = slavePara.result;

	for(int i = row_off ; i < row_off + row_len; i++) {
        for(int j = csr_matrix.row_off[i]; j < csr_matrix.row_off[i+1]; j++){
            if(csr_matrix.cols[j] == i) {
                pre.pre_mat_val[j] = 0.;	 
                preD[i] = 1.0/csr_matrix.data[j];
            } else {
                pre_mat_val[j] = csr_matrix.data[j];
            }
        }
    }
}