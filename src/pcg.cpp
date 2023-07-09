#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <crts.h>
#include "pcg.h"

typedef struct{
	double *A;
	double *B;
    double *C;
    double *D;
	double x;
	int cells;
} Para;
extern "C" void slave_loop1(Para *para);
extern "C" void slave_loop2(Para *para);
extern "C" void slave_gsumProd(Para *para);
extern "C" void slave_gsumMag(Para *para);
extern "C" void slave_v_dot_product(Para *para);
extern "C" void slave_v_sub_dot_product(Para *para);

typedef struct{
	struct CsrMatrix csr_matrix;
	double* vec;
	double* result;
	double* val;
    int* row_off;
    int* val_off;
    int isNew;
} spmvPara;
extern "C" void slave_spmv(spmvPara *para);
extern "C" void slave_pre_spmv(spmvPara *para);
int cells;
int row_off[65];
int val_off[65];
double* gAPtr;
int isNew;

void loop1(PCG& pcg);
void loop2(double* x, PCG& pcg);
void csr_div_thread(const CsrMatrix &csr_matrix);

PCGReturn pcg_solve(const LduMatrix &ldu_matrix, double *source, double *x, int maxIter, double tolerance, double normfactor) {
    static int isInit = 0;
	if(isInit == 0){
		CRTS_init();
		isInit = 1;
	}

    int iter = 0;
    cells = ldu_matrix.cells;
    int faces = ldu_matrix.faces;
    isNew = 1;

    PCG pcg;
    pcg.r = (double *)malloc(cells*sizeof(double));
    pcg.z = (double *)malloc(cells*sizeof(double));
    pcg.p = (double *)malloc(cells*sizeof(double));
    pcg.Ax = (double *)malloc(cells*sizeof(double));
    pcg.x = x;
    pcg.source = source;

    Precondition pre;
    pre.preD = (double *)malloc(cells*sizeof(double));
    pre.pre_mat_val = (double *)malloc((cells + faces*2)*sizeof(double));

    CsrMatrix csr_matrix;
    ldu_to_csr(ldu_matrix, csr_matrix); 

    pcg_init_precondition_csr(csr_matrix, pre);

    csr_div_thread(csr_matrix);

    // AX = A * X	 
    csr_spmv(csr_matrix, x, pcg.Ax);
    isNew = 0;
    // r = b - A * x
    for(int i = 0; i < cells; i++) {
        pcg.r[i] = source[i] - pcg.Ax[i];
    }
	
    pcg.residual = pcg_gsumMag(pcg.r, cells);
    double init_residual = pcg.residual;

    gAPtr = (double*)malloc(csr_matrix.rows*sizeof(double));
    
    if(fabs(pcg.residual / normfactor) > tolerance ) {
        do {
            if(iter == 0) {
                // z = M(-1) * r
                pcg_precondition_csr(csr_matrix, pre, pcg.r, pcg.z);
                // tol_0= swap(r) * z
                pcg.sumprod = pcg_gsumProd(pcg.r, pcg.z, cells);
                // iter ==0 ; p = z					 
                memcpy(pcg.p, pcg.z, cells*sizeof(double));	 
            } else {
                pcg.sumprod_old = pcg.sumprod; 
                // z = M(-1) * r
                pcg_precondition_csr(csr_matrix, pre, pcg.r, pcg.z);
                // tol_0= swap(r) * z
                pcg.sumprod = pcg_gsumProd(pcg.r, pcg.z, cells);
                // beta = tol_1 / tol_0 
                // p = z + beta * p				 
                pcg.beta = pcg.sumprod / pcg.sumprod_old;
				
                loop1(pcg);
            }

            // Ax = A * p			 
            csr_spmv(csr_matrix, pcg.p, pcg.Ax);

            // alpha = tol_0 / tol_1 = (swap(r) * z) / ( swap(p) * A * p) 
            pcg.alpha = pcg.sumprod / pcg_gsumProd(pcg.p, pcg.Ax, cells);

            loop2(x, pcg);

            // tol_1 = swap(z) * r				 
            pcg.residual = pcg_gsumMag(pcg.r, cells);
        } while( ++iter < maxIter  && (pcg.residual/normfactor) >= tolerance);
    }

    free(gAPtr);
    
    INFO("PCG: init residual = %e, final residual = %e, iterations: %d\n", init_residual, pcg.residual, iter);
    free_pcg(pcg);
    free_csr_matrix(csr_matrix);
    free_precondition(pre);

    PCGReturn pcg_return;
    pcg_return.residual = pcg.residual;
    pcg_return.iter = iter;
    return pcg_return;
}

void csr_div_thread(const CsrMatrix &csr_matrix) {
    for (int i = 0; i <= 64; i++) {
        row_off[i] = 0;
        val_off[i] = 0;
    }
    int size = csr_matrix.data_size / 64 + 1;
    int cnt = 0;
    for (int i = 1; i <= csr_matrix.rows; i++) {
        if (csr_matrix.row_off[i] - val_off[cnt] >= size) {
            cnt++;
            val_off[cnt] = csr_matrix.row_off[i];
            row_off[cnt] = i;
        }
    }
    while (cnt < 64) {
        cnt++;
        val_off[cnt] = csr_matrix.data_size;
        row_off[cnt] = csr_matrix.rows;
    }
}

void ldu_to_csr(const LduMatrix &ldu_matrix, CsrMatrix &csr_matrix) {
    csr_matrix.rows = ldu_matrix.cells;
    csr_matrix.data_size = 2*ldu_matrix.faces + ldu_matrix.cells;
    csr_matrix.row_off = (int *)malloc((csr_matrix.rows + 1)*sizeof(int));
    csr_matrix.cols = (int *)malloc(csr_matrix.data_size*sizeof(int));
    csr_matrix.data = (double *)malloc(csr_matrix.data_size*sizeof(double));

    int row, col, offset;
    int *tmp = (int *)malloc((csr_matrix.rows + 1)*sizeof(int));

    csr_matrix.row_off[0] = 0;
    for(int i = 1; i < csr_matrix.rows + 1; i++)
        csr_matrix.row_off[i] = 1; 
        
    for(int i = 0; i < ldu_matrix.faces; i++){
        row	= ldu_matrix.uPtr[i] ;
        col = ldu_matrix.lPtr[i] ;
        csr_matrix.row_off[row+1]++;
        csr_matrix.row_off[col+1]++;
    }

    for(int i = 0;i< ldu_matrix.cells; i++){
        csr_matrix.row_off[i+1] += csr_matrix.row_off[i];
    }

    memcpy(&tmp[0], &csr_matrix.row_off[0], (ldu_matrix.cells + 1)*sizeof(int));
    // lower
    for(int i = 0; i < ldu_matrix.faces; i++ ){
        row = ldu_matrix.uPtr[i];
        col = ldu_matrix.lPtr[i];
        offset = tmp[row]++;
        csr_matrix.cols[offset] = col;
        csr_matrix.data[offset] = ldu_matrix.lower[i];
    }

    // diag
    for(int i = 0; i < ldu_matrix.cells; i++){
        offset = tmp[i]++;
        csr_matrix.cols[offset] = i;
        csr_matrix.data[offset] = ldu_matrix.diag[i];
    }

    // upper
    for(int i = 0; i < ldu_matrix.faces; i++){
        row = ldu_matrix.lPtr[i];
        col = ldu_matrix.uPtr[i];
        offset = tmp[row]++;
        csr_matrix.cols[offset] = col;
        csr_matrix.data[offset] = ldu_matrix.upper[i];
    }

    free(tmp);
}

void csr_spmv(const CsrMatrix &csr_matrix, double *vec, double *result) {
    /*for(int i = 0; i < csr_matrix.rows; i++) {
        int start = csr_matrix.row_off[i];
        int num = csr_matrix.row_off[i+1] - csr_matrix.row_off[i];
        double temp = 0;
        for(int j = 0; j < num; j++) {                      
            temp += vec[csr_matrix.cols[start+j]] * csr_matrix.data[start+j]; 
        }
        result[i]=temp;
    }*/
    spmvPara para;
    para.csr_matrix = csr_matrix;
    para.result = result;
    para.vec = vec;
    para.val_off = val_off;
    para.row_off = row_off;
    para.isNew = isNew;
    athread_spawn(slave_spmv, &para);
	athread_join();
}

void csr_precondition_spmv(const CsrMatrix &csr_matrix, double *vec, double *val, double *result) {
    /*for(int i = 0; i < csr_matrix.rows; i++) {
        int start = csr_matrix.row_off[i];
        int num = csr_matrix.row_off[i+1] - csr_matrix.row_off[i];
        double temp = 0;
        for(int j = 0; j < num; j++) {                      
            temp += vec[csr_matrix.cols[start+j]] * val[start+j]; 
        }
        result[i]=temp;
    }*/
    spmvPara para;
    para.csr_matrix = csr_matrix;
    para.result = result;
    para.vec = vec;
    para.val = val;
    para.row_off = row_off;
    para.val_off = val_off;
    para.isNew = isNew; 
    athread_spawn(slave_pre_spmv, &para);
	athread_join();
}

void v_dot_product(const int nCells, const double *vec1, const double *vec2, double *result) {
    // for(int cell = 0; cell < nCells; cell++) {
    //     result[cell] = vec1[cell] * vec2[cell];
    // }
    Para para;
	para.A = vec1;
	para.B = vec2;
    para.C = result;
	para.cells = nCells;

	athread_spawn(slave_v_dot_product, &para);
	athread_join();
}

void v_sub_dot_product(const int nCells, const double *sub, const double *subed, const double *vec, double *result) {
    // for(int cell = 0; cell < nCells; cell++) {
    //     result[cell] = (sub[cell] - subed[cell])*vec[cell];
    // }
    Para para;
	para.A = sub;
	para.B = subed;
    para.C = vec;
    para.D = result;
	para.cells = nCells;

	athread_spawn(slave_v_sub_dot_product, &para);
	athread_join();
}

void pcg_init_precondition_csr (const CsrMatrix &csr_matrix, Precondition &pre) {
    for(int i = 0 ; i < csr_matrix.rows; i++) {
        for(int j = csr_matrix.row_off[i]; j < csr_matrix.row_off[i+1]; j++){
            if(csr_matrix.cols[j] == i) {
                pre.pre_mat_val[j] = 0.;	 
                pre.preD[i] = 1.0/csr_matrix.data[j];
            } else {
                pre.pre_mat_val[j] = csr_matrix.data[j];
            }
        }
    }
}

void pcg_precondition_csr(const CsrMatrix &csr_matrix, const Precondition &pre, double *rAPtr, double *wAPtr) {
    v_dot_product(csr_matrix.rows, pre.preD, rAPtr, wAPtr);
    csr_precondition_spmv(csr_matrix, wAPtr, pre.pre_mat_val, gAPtr);
    v_sub_dot_product(csr_matrix.rows, rAPtr, gAPtr, pre.preD, wAPtr);
}

void loop1(PCG& pcg) {
    /*for(int i = 0; i < cells; i++) {
        pcg.p[i] = pcg.z[i] + pcg.beta * pcg.p[i];
    }*/
	
	Para para;
	para.A = pcg.p;
	para.B = pcg.z;
	para.x = pcg.beta;
	para.cells = cells;

	athread_spawn(slave_loop1, &para);
	athread_join();
}

void loop2(double* x, PCG& pcg) {
    // x = x + alpha * p
    // r = r - alpha * Ax

    /*for(int i = 0; i < cells; i++) {
        x[i] = x[i] + pcg.alpha * pcg.p[i];
        pcg.r[i] = pcg.r[i] - pcg.alpha * pcg.Ax[i];
    }*/
    Para para;
    para.A = x;
    para.B = pcg.p;
    para.C = pcg.r;
    para.D = pcg.Ax;
    para.x = pcg.alpha;
    para.cells = cells;

    athread_spawn(slave_loop2, &para);
	athread_join();    
}

double pcg_gsumMag(double *r, int size) {
    double ret = .0;
    double temp_ret[64];
    // for(int i = 0; i < size; i++) {
    //     ret += fabs(r[i]);
    // }

    Para para;
    para.A = r;
    para.B = temp_ret;
    para.cells = size;

    athread_spawn(slave_gsumMag, &para);
	athread_join();   

    for (int i = 0; i < 64; i++) {
        ret += temp_ret[i];
    }

    return ret;
}

double pcg_gsumProd(double *z, double *r, int size) {
    double ret = .0;
    double temp_ret[64];
    // for(int i = 0; i < size; i++) {
    //     ret += z[i] * r[i];
    // }

    Para para;
    para.A = z;
    para.B = r;
    para.C = temp_ret;
    para.cells = size;

    athread_spawn(slave_gsumProd, &para);
	athread_join();   

    for (int i = 0; i < 64; i++) {
        ret += temp_ret[i];
    }

    return ret;
}

void free_pcg(PCG &pcg) {
    free(pcg.r);
    free(pcg.z);
    free(pcg.p);
    free(pcg.Ax);
}

void free_csr_matrix(CsrMatrix &csr_matrix) {
    free(csr_matrix.cols);
    free(csr_matrix.data);
    free(csr_matrix.row_off);
}

void free_precondition(Precondition &pre) {
    free(pre.preD);
    free(pre.pre_mat_val);
}
