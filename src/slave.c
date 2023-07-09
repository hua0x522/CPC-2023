#include <slave.h>
#include "pcg_def.h"
#include <math.h>
#include <simd.h>
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
} spmvPara;

#define dataBufferSize 2000
__thread_local crts_rply_t DMARply = 0;
__thread_local unsigned int DMARplyCount = 0;
__thread_local double A[dataBufferSize] __attribute__ ((aligned(64)));
__thread_local double B[dataBufferSize] __attribute__ ((aligned(64)));
__thread_local double C[dataBufferSize] __attribute__ ((aligned(64)));
__thread_local double D[dataBufferSize] __attribute__ ((aligned(64)));

__thread_local double csr_data[dataBufferSize / 10] __attribute__ ((aligned(64)));
__thread_local int csr_col[dataBufferSize / 10] __attribute__ ((aligned(64)));
__thread_local int csr_row[dataBufferSize] __attribute__ ((aligned(64)));

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
	int _len = len / 8 * 8;
	doublev8 va, vb, vc;
	for(; i < _len; i += 8){
		simd_load(va, A+i);
		simd_load(vb, B+i);
		vc = va * vb;
		simd_store(vc, C+i);
	}
	for (; i < len; i++) {
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
	int _len = len / 8 * 8;
	doublev8 va, vb, vc, vd;
	for(; i < _len; i += 8){
		simd_load(va, A+i);
		simd_load(vb, B+i);
		simd_load(vc, C+i);
		vd = (va - vb) * vc;
		simd_store(vd, D+i);
	}
	for (; i < len; i++) {
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
