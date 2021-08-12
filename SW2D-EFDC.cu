
//************************************************************************************************
// Coupled 2D-3D Hydrologic-Hydrodynamic model (SW2D-EFDC)
//
// Developer: Tomas Carlotto

//************************************************************************************************
// Prerequisites for using parallel code:
//         Computer equipped with NVIDIA GPU (compatible with CUDA technology).
//         Software required: CUDA™ Toolkit 8.0 or later 
//                  
//         System: Linux
//************************************************************************************************

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <stdio.h>
#include <math.h>
#include <ctime>
#include <chrono>
#include <sstream>

__global__ void init_inf(int rows, int cols, double *d_ho, double *d_h, int *d_inf, double *d_baseo, int N, double NaN){

	int id_inf = blockDim.x*blockIdx.x + threadIdx.x;

	while (id_inf < N){
		int inj = id_inf % cols;
		int ini = id_inf / cols;

		d_ho[id_inf] = d_h[id_inf];

		if (d_baseo[id_inf] != NaN){
			d_inf[id_inf] = 1;
		}
		else{
			d_inf[id_inf] = 0;
		}

		// ***************************************************************************
		id_inf += gridDim.x * blockDim.x;
	}
}

__global__ void initiald(int rows, int cols, double *d_h, int *d_infx, int *d_infy, int *d_inf, double *d_hm, double *d_hn, double *d_baseo, int N, double NaN){

	int id_init = blockDim.x*blockIdx.x + threadIdx.x;
	double hmn;

	while (id_init < N){
		int inj = id_init % cols;
		int ini = id_init / cols;

		// ***************************************************************************
		if (ini == 0){
			hmn = d_h[id_init];
			d_infx[id_init] = 1;
		}
		else if (ini == rows){
			hmn = d_h[id_init - cols];
			d_infx[id_init] = 1;
		}
		else{
			hmn = 0.50*(d_h[id_init] + d_h[id_init - cols]);
			d_infx[id_init] = abs(d_inf[id_init] - d_inf[id_init - cols]);
		}
		d_hm[id_init] = hmn;
		// ****************************************************************************
		if (inj == 0){
			hmn = d_h[id_init];
			d_infy[id_init] = 1;
		}
		else if (inj == cols){
			hmn = d_h[id_init - 1];
			d_infy[id_init] = 1;
		}
		else{
			hmn = 0.50*(d_h[id_init] + d_h[id_init - 1]);
			d_infy[id_init] = abs(d_inf[id_init] - d_inf[id_init - 1]);
		}
		d_hn[id_init] = hmn;
		// ***************************************************************************
		id_init += gridDim.x * blockDim.x;
	}
}

// ************************************************************
//          2D FLOW CALCULATION
// ************************************************************

__global__ void flux(double *d_th, double gg, double rn, int* d_inf, double* d_h, int* d_infx, int* d_infy, \
	double* d_baseo, double* d_um, double* d_hm, double* d_uu1, double* d_umo, double* d_vv1, \
	double* d_vva, double* d_vn, double*d_hn, double* d_vno, double* d_uua, \
	double* ho, int N, int cols, int rows, double dx, double dy, double dt2)
{
	double hhn, hhs, hhnp, hhsp, hhe, hhw, hhep, hhwp, hhan, sgnm, hh3, u13, u11uur, u11uul, umr, uml, \
		u11, u12vvu, u12vvd, umu, umd, u12, sqx, ram, v13, v11, v11uur, v11uul, vnr, vnl, v12vvu, v12vvd, vnu, vnd, v12, sqy;

	int f_id = blockDim.x*blockIdx.x + threadIdx.x;

	while (f_id < N){
		int inj = f_id % cols;
		int ini = f_id / cols;

		//      ----------------------
		//      X - DIRECTION
		//      ----------------------

		if (ini > 0 && ini < rows) {
			if (d_inf[f_id] != 0) {
				if (d_inf[f_id - cols] != 0){
					if ((d_h[f_id - cols] > d_th[0]) || (d_h[f_id] > d_th[0])){
						if (d_infx[f_id] != 1){
							hhe = d_h[f_id] + d_baseo[f_id];
							hhw = d_h[f_id - cols] + d_baseo[f_id - cols];
							hhep = d_h[f_id] - d_th[0];
							hhwp = d_h[f_id - cols] - d_th[0];

							//      ----------------------
							//      DRY BED TREATMENT (1)
							//      ----------------------

							if (hhe<d_baseo[f_id - cols]){
								if (d_h[f_id - cols]>d_th[0]){
									d_um[f_id] = 0.5440*d_h[f_id - cols] * sqrt(gg*d_h[f_id - cols]);
								}
								else{
									d_um[f_id] = 0;
								}
							}
							else if (hhw < d_baseo[f_id]){
								if (d_h[f_id]>d_th[0]){
									d_um[f_id] = -0.544*d_h[f_id] * sqrt(gg*d_h[f_id]);
								}
								else{
									d_um[f_id] = 0;
								}
							}
							//      ----------------------
							//      DRY BED TREATMENT (2)
							//      ----------------------
							else if (hhep*hhwp < 0){
								if ((d_h[f_id]>0) || (d_h[f_id - cols]>0)){
									hhan = hhep - hhwp;
									sgnm = hhan / abs(hhan);
									hh3 = fmax((d_h[f_id] + d_baseo[f_id]), (d_h[f_id - cols] + d_baseo[f_id - cols])) - fmax(d_baseo[f_id], d_baseo[f_id - cols]);
									d_um[f_id] = -sgnm*0.350*hh3*sqrt(2.00*gg*hh3);
								}
								else{
									d_um[f_id] = 0;
								}
							}

							else{

								//      ----------------------
								//      GRAVITY TERM
								//      ----------------------
								u13 = gg*d_hm[f_id] * (dt2 / dx)*(d_h[f_id] + d_baseo[f_id] - d_h[f_id - cols] - d_baseo[f_id - cols]);

								//      ----------------------
								//      CONVECTION TERM
								//      ----------------------

								u11uur = 0.50*(d_uu1[f_id + cols] + d_uu1[f_id]);
								u11uul = 0.50*(d_uu1[f_id] + d_uu1[f_id - cols]);
								umr = u11uur*(d_umo[f_id + cols] + d_umo[f_id])*0.50 + abs(u11uur)*(d_umo[f_id] - d_umo[f_id + cols])*0.50;
								uml = u11uul*(d_umo[f_id] + d_umo[f_id - cols])*0.50 + abs(u11uul)*(d_umo[f_id - cols] - d_umo[f_id])*0.50;
								u11 = (dt2 / dx)*(umr - uml);

								if (inj == 0){
									u12 = 0;
								}
								else if (inj == cols){
									u12 = 0;
								}
								else{
									u12vvu = 0.50*(d_vv1[f_id + 1] + d_vv1[f_id - cols + 1]);
									u12vvd = 0.50*(d_vv1[f_id] + d_vv1[f_id - cols]);
									umu = u12vvu*(d_umo[f_id + 1] + d_umo[f_id])*0.50 + abs(u12vvu)*(d_umo[f_id] - d_umo[f_id + 1])*0.50;
									umd = u12vvd*(d_umo[f_id - 1] + d_umo[f_id])*0.50 + abs(u12vvd)*(d_umo[f_id - 1] - d_umo[f_id])*0.50;
									u12 = (dt2 / dy)*(umu - umd);
								}
								//      ----------------------
								//      FRICTION TERM
								//      ----------------------
								sqx = sqrt(d_uu1[f_id] * d_uu1[f_id] + d_vva[f_id] * d_vva[f_id]);
								double expx = pow(double(d_hm[f_id]), double(1.3333330));
								ram = gg*rn*rn*sqx / expx;
								if (d_hm[f_id] <= d_th[0]){
									ram = 0.00;
								}
								//      ----------------------
								//      UM
								//      ----------------------
								d_um[f_id] = ((1.00 - dt2*ram*0.50)*d_umo[f_id] + (-u11 - u12 - u13)) / (1.00 + dt2*ram*0.50);
							}

						}
						else{
							d_um[f_id] = 0;
						}
					}
					else{
						d_um[f_id] = 0;
					}
				}
				else{
					d_um[f_id] = 0;
				}
			}
			else{
				d_um[f_id] = 0;
			}
		}
		else{
			d_um[f_id] = 0;
		}


		//      ----------------------
		//      Y - DIRECTION
		//      ----------------------

		if (inj > 0 && inj < cols) {
			if (d_inf[f_id] != 0) {
				if (d_inf[f_id - 1] != 0){
					if ((d_h[f_id - 1] > d_th[0]) || (d_h[f_id] > d_th[0])){
						if (d_infy[f_id] != 1){
							hhn = d_h[f_id] + d_baseo[f_id];
							hhs = d_h[f_id - 1] + d_baseo[f_id - 1];
							hhnp = d_h[f_id] - d_th[0];
							hhsp = d_h[f_id - 1] - d_th[0];


							//      ----------------------
							//      DRY BED TREATMENT (1)
							//      ----------------------

							if (hhn<d_baseo[f_id - 1]){
								if (d_h[f_id - 1]>d_th[0]){
									d_vn[f_id] = 0.5440*d_h[f_id - 1] * sqrt(gg*d_h[f_id - 1]);
								}
								else{
									d_vn[f_id] = 0;
								}
							}
							else if (hhs < d_baseo[f_id]){
								if (d_h[f_id]>d_th[0]){
									d_vn[f_id] = -0.544*d_h[f_id] * sqrt(gg*d_h[f_id]);
								}
								else{
									d_vn[f_id] = 0;
								}
							}
							//      ----------------------
							//      DRY BED TREATMENT (2)
							//      ----------------------
							else if (hhnp*hhsp < 0){
								if ((d_h[f_id]>0) || (d_h[f_id - 1]>0)){
									hhan = hhnp - hhsp;
									sgnm = hhan / abs(hhan);
									hh3 = fmax((d_h[f_id] + d_baseo[f_id]), (d_h[f_id - 1] + d_baseo[f_id - 1])) - fmax(d_baseo[f_id], d_baseo[f_id - 1]);
									d_vn[f_id] = -sgnm*0.350*hh3*sqrt(2.00*gg*hh3);
								}
								else{
									d_vn[f_id] = 0;
								}
							}

							else{

								//      ----------------------
								//      GRAVITY TERM
								//      ----------------------
								v13 = gg*d_hn[f_id] * (dt2 / dy)*(d_h[f_id] + d_baseo[f_id] - d_h[f_id - 1] - d_baseo[f_id - 1]);

								//      ----------------------
								//      CONVECTION TERM
								//      ----------------------			

								if (ini == 0){
									v11 = 0;
								}
								else if (ini == rows){
									v11 = 0;
								}
								else{
									v11uur = 0.50*(d_uu1[f_id + cols] + d_uu1[f_id + cols - 1]);
									v11uul = 0.50*(d_uu1[f_id] + d_uu1[f_id - 1]);
									vnr = v11uur*(d_vno[f_id + cols] + d_vno[f_id])*0.50 + abs(v11uur)*(d_vno[f_id] - d_vno[f_id + cols])*0.50;
									vnl = v11uul*(d_vno[f_id] + d_vno[f_id - cols])*0.50 + abs(v11uul)*(d_vno[f_id - cols] - d_vno[f_id])*0.50;
									v11 = (dt2 / dx)*(vnr - vnl);
								}

								v12vvu = 0.50*(d_vv1[f_id + 1] + d_vv1[f_id]);
								v12vvd = 0.50*(d_vv1[f_id] + d_vv1[f_id - 1]);
								vnu = v12vvu*(d_vno[f_id + 1] + d_vno[f_id])*0.50 + abs(v12vvu)*(d_vno[f_id] - d_vno[f_id + 1])*0.50;
								vnd = v12vvd*(d_vno[f_id - 1] + d_vno[f_id])*0.50 + abs(v12vvd)*(d_vno[f_id - 1] - d_vno[f_id])*0.50;
								v12 = (dt2 / dy)*(vnu - vnd);
								//      ----------------------
								//      FRICTION TERM
								//      ----------------------
								sqy = sqrt(d_uua[f_id] * d_uua[f_id] + d_vv1[f_id] * d_vv1[f_id]);
								double expy = pow(double(d_hn[f_id]), double(1.3333330));
								ram = gg*rn*rn*sqy / expy;
								if (d_hn[f_id] <= d_th[0]){
									ram = 0.00;
								}
								//      ----------------------
								//      VN
								//      ----------------------
								d_vn[f_id] = ((1.00 - dt2*ram*0.50)*d_vno[f_id] + (-v11 - v12 - v13)) / (1.00 + dt2*ram*0.50);
							}

						}
						else{
							d_vn[f_id] = 0;
						}
					}
					else{
						d_vn[f_id] = 0;
					}
				}
				else{
					d_vn[f_id] = 0;
				}
			}
			else{
				d_vn[f_id] = 0;
			}
		}
		else{
			d_vn[f_id] = 0;
		}

		//      +++++++++++++++++++++++++++++++++++++++++++
		//      CONTINUITY EQUATION
		//      +++++++++++++++++++++++++++++++++++++++++++

		//      +++++++++++++++++++++++++++++++++++++++++++


		f_id += gridDim.x * blockDim.x;
	}

}

//   +++++++++++++++++++++++++++++
//   PREPARING NEXT CALCULATION
//   +++++++++++++++++++++++++++++
//   ------------------------------
//   hm, hn, calculation
//   ------------------------------

__global__ void hm_hn(double* d_hm, double* d_hn, double* d_h, int N, int cols, int rows){


	int hmhn_id = blockDim.x*blockIdx.x + threadIdx.x;

	while (hmhn_id < N){
		int inj = hmhn_id % cols;
		int ini = hmhn_id / cols;
		//   ------------------------------
		//   hm, hn, calculation
		//   ------------------------------
		if ((inj>0)){
			d_hn[hmhn_id] = 0.50*(d_h[hmhn_id] + d_h[hmhn_id - 1]);
		}

		if ((ini>0)){
			d_hm[hmhn_id] = 0.50*(d_h[hmhn_id] + d_h[hmhn_id - cols]);
		}

		if (ini == 0){
			d_hm[hmhn_id] = d_h[hmhn_id];
		}
		if (ini == rows){
			d_hm[hmhn_id] = d_h[hmhn_id - cols];
		}

		if (inj == 0){
			d_hn[hmhn_id] = d_h[hmhn_id];
		}
		if (inj == cols){
			d_hn[hmhn_id] = d_h[hmhn_id - 1];
		}

		hmhn_id += gridDim.x * blockDim.x;
	}

}

//   ------------------------------
//   uu1, vv1, calculation         
//   ------------------------------

__global__ void uu1_vv1(double *d_th, double* d_hm, double* d_hn, double* d_uu1, double* d_um, double* d_vv1, double* d_vn, int N, int cols, int rows){


	int u1v1_id = blockDim.x*blockIdx.x + threadIdx.x;

	while (u1v1_id < N){
		int inj = u1v1_id % cols;
		int ini = u1v1_id / cols;

		//   ------------------------------
		//   uu1, vv1, calculation         
		//   ------------------------------

		if (d_hm[u1v1_id] >= d_th[0]){
			d_uu1[u1v1_id] = d_um[u1v1_id] / d_hm[u1v1_id];
		}
		else{
			d_uu1[u1v1_id] = 0.00;
		}

		if (d_hn[u1v1_id] >= d_th[0]){
			d_vv1[u1v1_id] = d_vn[u1v1_id] / d_hn[u1v1_id];
		}
		else{
			d_vv1[u1v1_id] = 0.0;
		}

		if (ini == rows){
			if (d_hm[u1v1_id] >= d_th[0]){
				d_uu1[u1v1_id] = d_um[u1v1_id] / d_hm[u1v1_id];
			}
			else{
				d_uu1[u1v1_id] = 0.00;
			}
		}

		if (inj == cols){
			if (d_hn[u1v1_id] >= d_th[0]){
				d_vv1[u1v1_id] = d_vn[u1v1_id] / d_hn[u1v1_id];
			}
			else{
				d_vv1[u1v1_id] = 0.00;
			}
		}

		u1v1_id += gridDim.x * blockDim.x;
	}

}

//   ------------------------------
//   uu, vv calculation
//   ------------------------------

__global__ void uu_vv(double *d_th, double* d_h, double* d_uu1, double* d_vv1, double*d_uu, double*d_vv, int N, int cols){


	int uuvv_id = blockDim.x*blockIdx.x + threadIdx.x;

	while (uuvv_id < N){
		int inj = uuvv_id % cols;
		int ini = uuvv_id / cols;

		//   ------------------------------
		//   uu, vv calculation
		//   ------------------------------

		if (d_h[uuvv_id] >= d_th[0]){
			d_uu[uuvv_id] = (d_uu1[uuvv_id + cols] + d_uu1[uuvv_id]) / 2.00;
			d_vv[uuvv_id] = (d_vv1[uuvv_id + 1] + d_vv1[uuvv_id]) / 2.00;
		}
		else{
			d_uu[uuvv_id] = 0.00;
			d_vv[uuvv_id] = 0.00;
		}

		uuvv_id += gridDim.x * blockDim.x;
	}

}

//   ------------------------------
//   uua, vva calculation
//   ------------------------------

__global__ void uua_vva(double* d_uu1, double* d_vv1, double*d_uua, double*d_vva, int N, int cols, int rows){


	int ua_va_id = blockDim.x*blockIdx.x + threadIdx.x;

	while (ua_va_id < N){
		int inj = ua_va_id % cols;
		int ini = ua_va_id / cols;

		//   ------------------------------
		//   uua, vva calculation
		//   ------------------------------		
		if (inj>0){
			d_uua[ua_va_id] = 0.250*(d_uu1[ua_va_id] + d_uu1[ua_va_id + cols] + d_uu1[ua_va_id - 1] + d_uu1[ua_va_id + cols - 1]);
		}		
		if (ini>0){
			d_vva[ua_va_id] = 0.250*(d_vv1[ua_va_id] + d_vv1[ua_va_id + 1] + d_vv1[ua_va_id - cols] + d_vv1[ua_va_id - cols + 1]);
		}
		if (inj == 0){
			d_uua[ua_va_id] = 0.50*(d_uu1[ua_va_id] + d_uu1[ua_va_id + cols]);
		}
		if (inj == cols){
			d_uua[ua_va_id] = 0.50*(d_uu1[ua_va_id] + d_uu1[ua_va_id + cols]);
		}
		if (ini == 0){
			d_vva[ua_va_id] = 0.50*(d_vv1[ua_va_id] + d_vv1[ua_va_id + 1]);
		}
		if (ini == rows){
			d_vva[ua_va_id] = 0.50*(d_vv1[ua_va_id] + d_vv1[ua_va_id + 1]);
		}
		ua_va_id += gridDim.x * blockDim.x;
	}

}

//**************************************************************************************************************
__host__ void stream_flow(int cols, int rows, double xcoor, double ycoor, double time, double dtrain, double *h_rain, double **h_qq, double *h_ql, double dtoq, double *h_brx, double *h_bry, double dx, double dy, int nst, double* h_rr){

	// nst = Input Numbers

	double ql;
	int it, qiny, qinx;
	for (int i = 0; i < nst; i++){
		if (time <= 1.00){
			ql = h_qq[0][i] * time;
		}
		else{
			it = int(time / dtoq);
			ql = h_qq[it][i] + (h_qq[it + 1][i] - h_qq[it][i]) / (dtoq * (time - dtoq * (it)));
		}
		ql = ql / (dx*dy);  //m3 / s->m / s

		qinx = round(abs(xcoor - h_brx[i]));
		qiny = rows - round(abs(ycoor - h_bry[i]));

		h_ql[(qiny + 1)*cols - (cols - (qinx + 1))] = ql;
	}
	if (time <= 1.00){
		h_rr[0] = h_rain[0] * time;
	}
	else{
		it = int(time / dtrain);
		h_rr[0] = h_rain[it] + (h_rain[it + 1] - h_rain[it]) / (dtrain * (time - dtrain * (it)));// [mm]
	}
	h_rr[0] = h_rr[0] / (dtrain*1000.0); //  mm->m / s
}

__global__ void gpu_evaporation_calc(double albedo, double* d_T, double*d_Rg, double* d_Rs, double* d_pw, double* d_lv, double*d_Evapo, double dtime,int N){
	
	int ev_id = blockDim.x*blockIdx.x + threadIdx.x;
	while (ev_id < N){
		d_Rs[ev_id] = (1 - albedo)*d_Rg[ev_id]; //net radiation[w/m^2/h];

		//Water density as a function of temperature[kg/m ^ 3] 5°C < Temperature < 40°C
		//ITS-90 Density of Water Formulation for Volumetric Standards Calibration (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4909168/)

		d_pw[ev_id] = 999.85308 + 6.32693*(0.01)*d_T[ev_id] - 8.523829*(0.001)*d_T[ev_id] * d_T[ev_id] + 6.943248*(0.00001)*d_T[ev_id]*d_T[ev_id]*d_T[ev_id] - 3.821216*(0.0000001)*d_T[ev_id]*d_T[ev_id]*d_T[ev_id]*d_T[ev_id];
		
		//Latent heat of vaporization [j/kg]; 273k < Temperature < 308k
	    //A new formula for latent heat of vaporization of water as a function of temperature
		//By B.HENDERSON - SELLERS(1984)
		//Department of Mathematics, University of Salford		

		d_lv[ev_id] = (((d_T[ev_id] + 273.15) / ((d_T[ev_id] + 273.15) - 33.91))*((d_T[ev_id] + 273.15) / ((d_T[ev_id] + 273.15) - 33.91)))*1.91846*1000000;		
		
		//Applied hydrology Cap 3.5(Chow etal 1988)
	    //Evaporation by the energy balance method
		d_Evapo[ev_id] = (d_Rs[ev_id] / (d_lv[ev_id] * d_pw[ev_id]))*dtime*1000.0; //[mm/dtime];
		

		if (d_Evapo[ev_id] < 0){
	       d_Evapo[ev_id] = 0;
		}

		ev_id += gridDim.x * blockDim.x;

	}
}

__host__ void evaporation_load(double time, double dtrain, double* h_Evapo, double* h_Ev){

	int it;
	if (time <= 1.00){
		h_Ev[0] = h_Evapo[0] * time;
	}
	else{
		it = int(time / dtrain);
		h_Ev[0] = h_Evapo[it] + (h_Evapo[it + 1] - h_Evapo[it]) / (dtrain * (time - dtrain * (it))); // [mm]
	}
	h_Ev[0] = h_Ev[0] / (dtrain*1000.0); //  mm->m / s

}

__global__ void continuity(double dt2, int cols,int rows, double dx, double dy, double *d_rr, double *d_Ev, double *d_ql, double *d_h, double *d_ho, double *d_um, double *d_vn, double INT, double INF, double LWL, double EV_WL_min, int *d_inf, int N){

	// Posicao do vertedouro Peri Lake
	//int qiny = 397;
	//int qinx = 142;

	// UFSC 
	//int qinx1 = round(abs(744107.8622132 - 745209.0));
	//int qiny1 = 1588 - round(abs(6943856.7347817 - 6944966.0));
	//int qinx2 = round(abs(744107.8622132 - 745209.0));
	//int qiny2 = 1588 - round(abs(6943856.7347817 - 6944980.0));
	//int qinx3 = round(abs(744107.8622132 - 744912.0));
	//int qiny3 = 1588 - round(abs(6943856.7347817 - 6944300.0));
	// ======================================

	double percent_P2flow, evapo;
	int ct_id = blockDim.x*blockIdx.x + threadIdx.x;
	while (ct_id < N){
		int ini = ct_id % cols;
		int inj = ct_id / cols;


		//************ Percentage of precipitation that becomes runoff **************

		if (d_ho[ct_id] > EV_WL_min){
			percent_P2flow = 1.00 - LWL;         // 
			evapo = d_Ev[0];                  // 
		}
		else{
			percent_P2flow = 1.00 - (INT + INF); // 
			evapo = 0.000;                    
		}
				 
		// 
		/*
		//if (((ini == qinx1) && ((inj >= qiny2) && (inj <= qiny1))) || ((ini == qinx3) && (inj == qiny3))){
		if ((ini == qinx1) || ((ini == qinx3) && (inj == qiny3))){
			if (d_h[ct_id] >= 0.05){        // Cota do vertedouro 280 cm na régua. Equivalente a 2.2088 m no nível do lago no ponto (142, 397)
				d_h[ct_id] = d_ho[ct_id] - dt2*((d_um[ct_id + cols] - d_um[ct_id]) / dx + (d_vn[ct_id + 1] - d_vn[ct_id]) / dy - d_ql[ct_id] - d_rr[0] * percent_P2flow + evapo)*0.0000;
			}
			else{
				d_h[ct_id] = d_ho[ct_id] - dt2*((d_um[ct_id + cols] - d_um[ct_id]) / dx + (d_vn[ct_id + 1] - d_vn[ct_id]) / dy - d_ql[ct_id] - d_rr[0] * percent_P2flow + evapo)*(1.0000);
			}
		}
		else{
			d_h[ct_id] = d_ho[ct_id] - dt2*((d_um[ct_id + cols] - d_um[ct_id]) / dx + (d_vn[ct_id + 1] - d_vn[ct_id]) / dy - d_ql[ct_id] - d_rr[0] * percent_P2flow + evapo);
		}
		*/
		
		
		//if ((ini < rows-1) && (inj<cols)){
			d_h[ct_id] = d_ho[ct_id] - dt2*((d_um[ct_id + cols] - d_um[ct_id]) / dx + (d_vn[ct_id + 1] - d_vn[ct_id]) / dy - d_ql[ct_id] - d_rr[0] * percent_P2flow + evapo);
		//}

		d_h[ct_id] = fmax(d_h[ct_id], 0.00);
		if (d_inf[ct_id] == 0){
			d_h[ct_id] = 0.00;
		}
		ct_id += gridDim.x * blockDim.x;

	}

}


//**************************************************************************************************************


__global__ void forward(int cols, int rows, double *d_umo, double *d_um, double *d_vno, double *d_vn, double *d_ho, double *d_h, int N){

	int fw_id = blockDim.x*blockIdx.x + threadIdx.x;

	while (fw_id<N){

		int inj = fw_id % cols;
		int ini = fw_id / cols;

		d_umo[fw_id] = d_um[fw_id];
		d_vno[fw_id] = d_vn[fw_id];
		d_ho[fw_id] = d_h[fw_id];

		if (ini == rows){
			d_umo[fw_id] = d_um[fw_id];
		}
		if (inj == cols){
			d_vno[fw_id] = d_vn[fw_id];
		}

		fw_id += gridDim.x * blockDim.x;
	}

}

__global__ void treat_error(int cols, int rows, double *d_th, int *d_inf, double *d_um, double *d_vn, double *d_h, int N){

	int er_id = blockDim.x*blockIdx.x + threadIdx.x;

	while (er_id<N){

		int inj = er_id % cols;
		int ini = er_id / cols;

		if ((d_inf[er_id] == 1) || (d_inf[er_id] == 2)){
			if (d_h[er_id]<d_th[0]){

				if (ini<rows){
					if (d_um[er_id + cols] > 0){ d_um[er_id + cols] = 0; }
				}

				if (d_um[er_id] < 0.0){ d_um[er_id] = 0.00; }

				if (inj<cols){
					if (d_vn[er_id + 1] > 0){ d_vn[er_id + 1] = 0; }
				}

				if (d_vn[er_id] < 0){ d_vn[er_id] = 0; }

			}
		}

		er_id += gridDim.x * blockDim.x;
	}

}

__host__ void couple_BC_efdc(int pressure_OpenBC, int volumeFlow_BC, int dimx, int dimy,double dx,double dy, int *Qnqs, int **north_bc, int **south_bc, int **east_bc, int **west_bc, double **north_val, double **south_val, double **east_val, double **west_val, int *mask, double *h_h,double *h_vn,double *h_um,int t, double time){

	int ini, inj;
	//int bc_id=0;	
	int n_id = -1;
	int s_id = -1;
	int e_id = -1;
	int w_id = -1;
	Qnqs[0] = 0;
	int posxy;
	
	for (ini = 0; ini<dimy; ini++){ 
		posxy = dimy*dimx - (ini + 1)*dimx;
		posxy = posxy - 1;
		for (inj = 0; inj<dimx; inj++) {
			posxy = posxy + 1;
			
	//int bc_id = blockDim.x*blockIdx.x + threadIdx.x;
	//while (bc_id<N){
	//	int inj = bc_id % dimx;  // 0 to dimx  (cols)     
	//	int ini = bc_id / dimx;  // 0 to dimy  (rows)

		if (mask[posxy] != 1){
			if ((ini > 0) && (ini < dimy - 1)){
				//======================
				if (inj>0 && inj < dimx - 1){ //dimx-1   	
		
					// South                posxy - dimx
					if (mask[posxy] != mask[posxy - dimx]){
						Qnqs[0] = Qnqs[0] + 1;
						s_id = s_id + 1;
						south_bc[s_id][0] = inj+1;
						south_bc[s_id][1] = ini+2;
						south_bc[s_id][2] = s_id+1;
						south_val[t][0] = time;
						if (pressure_OpenBC==1){
							south_val[t][s_id + 1] = h_h[posxy - dimx];
						}
						else if (volumeFlow_BC == 1){
							if (h_um[posxy]<0){
								south_val[t][s_id + 1] = -h_um[posxy]*h_h[posxy]*dx;
							}
							else{
								south_val[t][s_id + 1] = 0.00;
							}
						}
					}
					// North                posxy + dimx
					if (mask[posxy] != mask[posxy + dimx]){
						Qnqs[0] = Qnqs[0] + 1;
						n_id = n_id + 1;
						north_bc[n_id][0] = inj+1;
						north_bc[n_id][1] = ini;
						north_bc[n_id][2] = n_id + 1;
						north_val[t][0] = time;
						if (pressure_OpenBC == 1){
							north_val[t][n_id + 1] = h_h[posxy + dimx];
						}
						else if (volumeFlow_BC == 1){
							if (h_um[posxy]>0){
								north_val[t][n_id + 1] = h_um[posxy] *h_h[posxy]* dx;
							}
							else{
								north_val[t][n_id + 1] = 0.00;
							}
						}
					}
					// East                 posxy - 1
					if (mask[posxy] != mask[posxy - 1]){
						Qnqs[0] = Qnqs[0] + 1;
						e_id = e_id + 1;
						east_bc[e_id][0] = inj;
						east_bc[e_id][1] = ini+1;
						east_bc[e_id][2] = e_id + 1;
						east_val[t][0] = time;
						if (pressure_OpenBC == 1){
							east_val[t][e_id + 1] = h_h[posxy-1];
						}
						else if (volumeFlow_BC == 1){
							if (h_vn[posxy]<0){
								east_val[t][e_id + 1] = -h_vn[posxy] *h_h[posxy]* dy;
							}
							else{
								east_val[t][e_id + 1] = 0.00;
							}
						}

					}
					// West                 posxy + 1
					if (mask[posxy] != mask[posxy + 1]){
						Qnqs[0] = Qnqs[0] + 1;
						w_id = w_id + 1;
						west_bc[w_id][0] = inj+2;
						west_bc[w_id][1] = ini+1;
						west_bc[w_id][2] = w_id + 1;
						west_val[t][0] = time;
						if (pressure_OpenBC == 1){
							west_val[t][w_id + 1] = h_h[posxy+1];
						}
						else if (volumeFlow_BC == 1){
							if (h_vn[posxy]>0){
								west_val[t][w_id + 1] = h_vn[posxy] *h_h[posxy]* dy;
							}
							else{
								west_val[t][w_id + 1] = 0.00;
							}
						}

					}

				}
				else if (inj == dimx - 1) { //dimx-1          

					// South                posxy - dimx
					if (mask[posxy] != mask[posxy - dimx]){
						Qnqs[0] = Qnqs[0] + 1;
						s_id = s_id + 1;
						south_bc[s_id][0] = inj+1;
						south_bc[s_id][1] = ini+2;
						south_bc[s_id][2] = s_id + 1;
						south_val[t][0] = time;
						if (pressure_OpenBC == 1){
							south_val[t][s_id + 1] = h_h[posxy - dimx];
						}
						else if (volumeFlow_BC == 1){
							if (h_um[posxy]<0){
								south_val[t][s_id + 1] = -h_um[posxy]*h_h[posxy]*dx;
							}
							else{
								south_val[t][s_id + 1] = 0.00;
							}
						}
					}
					// North
					if (mask[posxy] != mask[posxy + dimx]){		
						Qnqs[0] = Qnqs[0] + 1;
						n_id = n_id + 1;
						north_bc[n_id][0] = inj+1;
						north_bc[n_id][1] = ini;
						north_bc[n_id][2] = n_id + 1;
						north_val[t][0] = time;
						if (pressure_OpenBC == 1){
							north_val[t][n_id + 1] = h_h[posxy + dimx];
						}
						else if (volumeFlow_BC == 1){
							if (h_um[posxy]>0){
								north_val[t][n_id + 1] = h_um[posxy]*h_h[posxy]*dx;
							}
							else{
								north_val[t][n_id + 1] = 0.00;
							}
						}

					}
					// East
					if (mask[posxy] != mask[posxy - 1]){		
						Qnqs[0] = Qnqs[0] + 1;
						e_id = e_id + 1;
						east_bc[e_id][0] = inj;
						east_bc[e_id][1] = ini+1;
						east_bc[e_id][2] = e_id + 1;
						east_val[t][0] = time;
						if (pressure_OpenBC == 1){
							east_val[t][e_id + 1] = h_h[posxy-1];
						}
						else if (volumeFlow_BC == 1){
							if (h_vn[posxy]<0){
								east_val[t][e_id + 1] = -h_vn[posxy]*h_h[posxy]*dy;
							}
							else{
								east_val[t][e_id + 1] = 0.00;
							}

						}

					}

				}
				else if (inj == 0) {

					// South
					if (mask[posxy] != mask[posxy - dimx]){	
						Qnqs[0] = Qnqs[0] + 1;
						s_id = s_id + 1;
						south_bc[s_id][0] = inj+1;
						south_bc[s_id][1] = ini+2;
						south_bc[s_id][2] = s_id + 1;
						south_val[t][0] = time;
						if (pressure_OpenBC == 1){
							south_val[t][s_id + 1] = h_h[posxy - dimx];
						}
						else if (volumeFlow_BC == 1){
							if (h_um[posxy]<0){
								south_val[t][s_id + 1] = -h_um[posxy] *h_h[posxy]* dx;
							}
							else{
								south_val[t][s_id + 1] = 0.00;
							}
						}
					}
					// North
					if (mask[posxy] != mask[posxy + dimx]){		
						Qnqs[0] = Qnqs[0] + 1;
						n_id = n_id + 1;
						north_bc[n_id][0] = inj+1;
						north_bc[n_id][1] = ini;
						north_bc[n_id][2] = n_id + 1;
						north_val[t][0] = time;
						if (pressure_OpenBC == 1){
							north_val[t][n_id + 1] = h_h[posxy + dimx];
						}
						else if (volumeFlow_BC == 1){
							if (h_um[posxy]>0){
								north_val[t][n_id + 1] = h_um[posxy] *h_h[posxy]* dx;
							}
							else{
								north_val[t][n_id + 1] = 0.00;
							}
						}

					}
					// West
					if (mask[posxy] != mask[posxy + 1]){
						Qnqs[0] = Qnqs[0] + 1;
						w_id = w_id + 1;
						west_bc[w_id][0] = inj+2;
						west_bc[w_id][1] = ini+1;
						west_bc[w_id][2] = w_id + 1;
						west_val[t][0] = time;
						if (pressure_OpenBC == 1){
							west_val[t][w_id + 1] = h_h[posxy+1];
						}
						else if (volumeFlow_BC == 1){
							if (h_vn[posxy]>0){
								west_val[t][w_id + 1] = h_vn[posxy]*h_h[posxy]*dy;
							}
							else{
								west_val[t][w_id + 1] = 0.00;
							}
						}

					}
				}
			}
			if (inj > 0 && inj < dimx - 1){
				if (ini == 0){
					// South
					if (mask[posxy] != mask[posxy - dimx]){
					   Qnqs[0] = Qnqs[0] + 1;
					   s_id = s_id + 1;
					   south_bc[s_id][0] = inj+1;
					   south_bc[s_id][1] = ini+2;
					   south_bc[s_id][2] = s_id + 1;
					   south_val[t][0] = time;
					   if (pressure_OpenBC == 1){
						   south_val[t][s_id + 1] = h_h[posxy - dimx];
					   }
					   else if (volumeFlow_BC == 1){
						   if (h_um[posxy]<0){
							   south_val[t][s_id + 1] = -h_um[posxy]*h_h[posxy]*dx;
						   }
						   else{
							   south_val[t][s_id + 1] = 0.00;
						   }
					   }
					}

					// East
					if (mask[posxy] != mask[posxy - 1]){	
						Qnqs[0] = Qnqs[0] + 1;
						e_id = e_id + 1;
						east_bc[e_id][0] = inj;
						east_bc[e_id][1] = ini+1;
						east_bc[e_id][2] = e_id + 1;
						east_val[t][0] = time;
						if (pressure_OpenBC == 1){
							east_val[t][e_id + 1] = h_h[posxy-1];
						}
						else if (volumeFlow_BC == 1){
							if (h_vn[posxy]<0){
								east_val[t][e_id + 1] = -h_vn[posxy]*h_h[posxy]*dy;
							}
							else{
								east_val[t][e_id + 1] = 0.00;
							}
						}

					}
					// West
					if (mask[posxy] != mask[posxy + 1]){
						Qnqs[0] = Qnqs[0] + 1;
						w_id = w_id + 1;
						west_bc[w_id][0] = inj+2;
						west_bc[w_id][1] = ini+1;
						west_bc[w_id][2] = w_id + 1;
						west_val[t][0] = time;
						if (pressure_OpenBC == 1){
							west_val[t][w_id + 1] = h_h[posxy+1];
						}
						else if (volumeFlow_BC == 1){
							if (h_vn[posxy]>0){
								west_val[t][w_id + 1] = h_vn[posxy]*h_h[posxy]*dy;
							}
							else{
								west_val[t][w_id + 1] = 0.00;
							}
						}

					}
				}
				else if (ini == dimy - 1){

					// North
					if (mask[posxy] != mask[posxy + dimx]){
					   Qnqs[0] = Qnqs[0] + 1;
					   n_id = n_id + 1;
					   north_bc[n_id][0] = inj + 1;
					   north_bc[n_id][1] = ini;
					   north_bc[n_id][2] = n_id + 1;
					   north_val[t][0] = time;
					   if (pressure_OpenBC == 1){
						   north_val[t][n_id + 1] = h_h[posxy + dimx];
					   }
					   else if (volumeFlow_BC == 1){

						   if (h_um[posxy]>0){
							   north_val[t][n_id + 1] = h_um[posxy]*h_h[posxy]*dx;
						   }
						   else{
							   north_val[t][n_id + 1] = 0.00;
						   }
					   }
					}
					// East
					if (mask[posxy] != mask[posxy - 1]){
						Qnqs[0] = Qnqs[0] + 1;
						e_id = e_id + 1;
						east_bc[e_id][0] = inj;
						east_bc[e_id][1] = ini+1;
						east_bc[e_id][2] = e_id + 1;
						east_val[t][0] = time;
						if (pressure_OpenBC == 1){
							east_val[t][e_id + 1] = h_h[posxy-1];
						}
						else if (volumeFlow_BC == 1){

							if (h_vn[posxy]<0){
								east_val[t][e_id + 1] = -h_vn[posxy]*h_h[posxy]*dy;
							}
							else{
								east_val[t][e_id + 1] = 0.00;
							}

						}
					}
					// West
					if (mask[posxy] != mask[posxy + 1]){
						Qnqs[0] = Qnqs[0] + 1;
						w_id = w_id + 1;
						west_bc[w_id][0] = inj+2;
						west_bc[w_id][1] = ini+1;
						west_bc[w_id][2] = w_id + 1;
						west_val[t][0] = time;
						if (pressure_OpenBC == 1){
							west_val[t][w_id + 1] = h_h[posxy+1];
						}
						else if (volumeFlow_BC == 1){

							if (h_vn[posxy]>0){
								west_val[t][w_id + 1] = h_vn[posxy]*h_h[posxy]*dy;
							}
							else{
								west_val[t][w_id + 1] = 0.00;
							}

						}
					}
				}
			}
		}

	//	bc_id += gridDim.x * blockDim.x;
	//} // end while

		}
		//====================		
	}		
}

__host__ void write_efdc_CONFIG(const char *filepath, int *nqsij,\
	int nqser, int **north_bc, int **south_bc, int **east_bc,\
	int **west_bc,int nlayers, int ncols, int nrows, int nwcells,\
	double TCON, double TBEGIN, double TREF,int NTC,int NTSPTC,int NTCPP,int NTCVB,\
	int ISPPH, int NPPPH, int ISRPPH, int IPPHXY,int ISVPH,int NPVPH,int ISRVPH,int IVPHXY,\
	int ISTMSR, int MLTMSR,int NBTMSR,int NSTMSR,int NWTMSR,int NTSSTSP,double TCTMSR,\
	int nts_out, int **LTS,int NTSSSS,int MTSP,int MTSC,int MTSA,int MTSUE,int MTSUT,int MTSU,\
	double ZBRADJ,double   ZBRCVRT,double   HMIN,double  HADJ,double   HCVRT,\
	double   HDRY,double  HWET,double  BELADJ,double   BELCVRT,\
	int NWSER,int NASER,double DX, double DY){
	
	char s6[100], s7[100], s8[100], s9[100],s10[100],s11[100],s14[100];
	char s22[100], s23[100], s24[100], s25[100], s26[100],s71[100], s72[100],s73[100],s84[100],s87[100];

	long pos_C6,pos_C7, pos_C8, pos_C9, pos_C10,pos_C11,pos_C11A,pos_C14,pos_C15;
	long pos, pos_C22, pos_C23, pos_C24, pos_C25, pos_C26, pos_C27, pos_C71, pos_C71A,\
	     pos_C72, pos_C73, pos_C74, pos_C84, pos_C85,pos_C87,pos_C88, tam, tamcp;
		 
	char *copia = NULL;
	int i, bn, bs, be, bw, nqs,nl;
	nqs = 0;

    int dye_on = 1;
	int salt_on = 1;
	int temp_on = 1;
	int temp_opt = 1;
	
	//===========================================================
	//                         C6
	//===========================================================
	
	FILE *f6 = fopen(filepath, "rb+");
	fseek(f6, 0, SEEK_END);//SEEK_END           //Manda o cursor para o fim do arquivo
	
	tam = ftell(f6) - 4;              //A posição final é igual ao tamanho do arquivo
	rewind(f6);                       //Volta para o início do arquivo	
	pos_C7 = ftell(f6);
	
	while (fscanf(f6, "%s%*c\n", s6) == 1 && strcmp(s6, "C7") != 0){
		pos_C7 = ftell(f6);
	}
	tamcp = tam - pos_C7;                      //tamanho de tudo que vem após a palavra "C7"
	copia = (char*)malloc(tamcp*sizeof(char)); //Aloca um vetor para copiar tudo que vem depois de "C7"
	fread(copia, tamcp, 1, f6);                //Copia tudo que vem depois de "C7" até o fim do arquivo
	rewind(f6);                                //Volta para o início do arquivo
	//Salva a posição atual e lê a proxima palavra, até encontrar
	//o início do card C6.
	pos_C6 = ftell(f6);
	while (fscanf(f6, "%s%*c", s6) == 1 && strcmp(s6, "C6") != 0){
		pos_C6 = ftell(f6);
	}
	
	fseek(f6, pos_C6, SEEK_SET); //Volta para a posição onde a palavra "C6" começa	
	fprintf(f6, "%s\n", "");
	fprintf(f6, "%s\n", "C6 DISSOLVED AND SUSPENDED CONSTITUENT TRANSPORT SWITCHES");  
	fprintf(f6, "%s\n", "*  TURB INTENSITY=0,SAL=1,TEM=2,DYE=3,SFL=4,TOX=5,SED=6,SND=7,CWQ=8");  
	fprintf(f6, "%s\n", "*  ");
	fprintf(f6, "%s\n", "*  ISTRAN:  1 OR GREATER TO ACTIVATE TRANSPORT");  
	fprintf(f6, "%s\n", "*  ISTOPT:    NONZERO FOR TRANSPORT OPTIONS, SEE USERS MANUAL");  
	fprintf(f6, "%s\n", "*  ISCDCA:  0 FOR STANDARD DONOR CELL UPWIND DIFFERENCE ADVECTION (3TL ONLY)");  
	fprintf(f6, "%s\n", "*           1 FOR CENTRAL DIFFERENCE ADVECTION FOR THREE TIME LEVEL STEPS (3TL ONLY)");  
	fprintf(f6, "%s\n", "*           2 FOR EXPERIMENTAL UPWIND DIFFERENCE ADVECTION (FOR RESEARCH) (3TL ONLY)");  
	fprintf(f6, "%s\n", "*  ISADAC:  1 TO ACTIVATE ANTI-NUMERICAL DIFFUSION CORRECTION TO");  
	fprintf(f6, "%s\n", "*             STANDARD DONOR CELL SCHEME");  
	fprintf(f6, "%s\n", "*  ISFCT:   1 TO ADD FLUX LIMITING TO ANTI-NUMERICAL DIFFUSION CORRECTION");  
	fprintf(f6, "%s\n", "*  ISPLIT:  1 TO OPERATOR SPLIT HORIZONTAL AND VERTICAL ADVECTION");  
	fprintf(f6, "%s\n", "*             (FOR RESEARCH PURPOSES)");  
	fprintf(f6, "%s\n", "*  ISADAH:  1 TO ACTIVATE ANTI-NUM DIFFUSION CORRECTION TO HORIZONTAL");  
	fprintf(f6, "%s\n", "*             SPLIT ADVECTION STANDARD DONOR CELL SCHEME (FOR RESEARCH)");  
	fprintf(f6, "%s\n", "*  ISADAV:  1 TO ACTIVATE ANTI-NUM DIFFUSION CORRECTION TO VERTICAL");  
	fprintf(f6, "%s\n", "*             SPLIT ADVECTION STANDARD DONOR CELL SCHEME (FOR RESEARCH)");  
	fprintf(f6, "%s\n", "*  ISCI:    1 TO READ CONCENTRATION FROM FILE restart.inp");  
	fprintf(f6, "%s\n", "*  ISCO:    1 TO WRITE CONCENTRATION TO FILE restart.out");  
	fprintf(f6, "%s\n", "* "); 
	fprintf(f6, "%s\n", "C6  ISTRAN  ISTOPT  ISCDCA  ISADAC   ISFCT  ISPLIT  ISADAH  ISADAV   ISCI    ISCO");
	
	
	fprintf(f6, "    %d      %d      %d      %d      %d      %d      %d      %d      %d      %d     %s\n", \
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0,"!TURB 0");
	fprintf(f6, "    %d      %d      %d      %d      %d      %d      %d      %d      %d      %d     %s\n", \
		salt_on, salt_on, 0, 1, 1, 0, 0, 0, 0, 0, "!SAL 1");
	fprintf(f6, "    %d      %d      %d      %d      %d      %d      %d      %d      %d      %d     %s\n", \
		temp_on, temp_opt, 0, 1, 1, 0, 0, 0, 0, 0, "!TEM 2");
	fprintf(f6, "    %d      %d      %d      %d      %d      %d      %d      %d      %d      %d     %s\n", \
		dye_on, dye_on, 0, 1, 1, 0, 0, 0, 0, 0, "!DYE 3");
	fprintf(f6, "    %d      %d      %d      %d      %d      %d      %d      %d      %d      %d     %s\n", \
		0, 0, 0, 1, 1, 0, 0, 0, 0, 0, "!SFL 4");
	fprintf(f6, "    %d      %d      %d      %d      %d      %d      %d      %d      %d      %d     %s\n", \
		0, 0, 0, 1, 1, 0, 0, 0, 0, 0, "!TOX 5");
	fprintf(f6, "    %d      %d      %d      %d      %d      %d      %d      %d      %d      %d     %s\n", \
		0, 0, 0, 1, 1, 0, 0, 0, 0, 0, "!SED 6");
	fprintf(f6, "    %d      %d      %d      %d      %d      %d      %d      %d      %d      %d     %s\n", \
		0, 0, 0, 1, 1, 0, 0, 0, 0, 0, "!SND 7");
	fprintf(f6, "    %d      %d      %d      %d      %d      %d      %d      %d      %d      %d     %s\n", \
		0, 0, 0, 1, 1, 0, 0, 0, 0, 0, "!CWQ ");
		
		
	fprintf(f6, "%s\n", "-------------------------------------------------------------------------------");
	fprintf(f6, "%s ", "C7");
	fwrite(copia, tamcp, 1, f6); //Reescreve tudo que foi copiado
	free(copia);                 //Libera memória alocada dinamicamente pois não é mais necessária
	//Se o novo texto é menor que o original, sobrepor os carateres finais com espaços em branco:
	for (pos = ftell(f6); pos < tam; pos++){
		fputc(' ', f6);
	}
	fclose(f6); 
	
	//===========================================================
	//                         C7
	//===========================================================

	FILE *f7 = fopen(filepath, "rb+");
	fseek(f7, 0, SEEK_END);//SEEK_END           
	
	tam = ftell(f7) - 4;              
	rewind(f7);                       
	pos_C8 = ftell(f7);
	
	while (fscanf(f7, "%s%*c\n", s7) == 1 && strcmp(s7, "C8") != 0){
		pos_C8 = ftell(f7);
	}
	tamcp = tam - pos_C8;                      
	copia = (char*)malloc(tamcp*sizeof(char)); 
	fread(copia, tamcp, 1, f7);                
	rewind(f7);                                
	
	pos_C7 = ftell(f7);
	while (fscanf(f7, "%s%*c", s7) == 1 && strcmp(s7, "C7") != 0){
		pos_C7 = ftell(f7);
	}
	
	fseek(f7, pos_C7, SEEK_SET); 	
	fprintf(f7, "%s\n", "");
	fprintf(f7, "%s\n", "C7 TIME-RELATED INTEGER PARAMETERS");
	fprintf(f7, "%s\n", "*");
	fprintf(f7, "%s\n", "*  NTC:     NUMBER OF REFERENCE TIME PERIODS IN RUN");
	fprintf(f7, "%s\n", "*  NTSPTC : NUMBER OF TIME STEPS PER REFERENCE TIME PERIOD");
	fprintf(f7, "%s\n", "*  NLTC : NUMBER OF LINEARIZED REFERENCE TIME PERIODS");
	fprintf(f7, "%s\n", "*  NLTC : NUMBER OF TRANSITION REF TIME PERIODS TO FULLY NONLINEAR");
	fprintf(f7, "%s\n", "*  NTCPP : NUMBER OF REFERENCE TIME PERIODS BETWEEN FULL PRINTED OUTPUT");
	fprintf(f7, "%s\n", "*           TO FILE EFDC.OUT");
	fprintf(f7, "%s\n", "*  NTSTBC : NUMBER OF TIME STEPS BETWEEN USING A TWO TIME LEVEL TRAPEZOIDAL");
	fprintf(f7, "%s\n", "*           CORRECTION TIME STEP, ** MASS BALANCE PRINT INTERVAL **");
	fprintf(f7, "%s\n", "*  NTCNB : NUMBER OF REFERENCE TIME PERIODS WITH NO BUOYANCY FORCING(not used)");
	fprintf(f7, "%s\n", "*  NTCVB : NUMBER OF REF TIME PERIODS WITH VARIABLE BUOYANCY FORCING");
	fprintf(f7, "%s\n", "*  NTSMMT : NUMBER OF NUMBER OF REF TIME TO AVERAGE OVER TO OBTAIN");
	fprintf(f7, "%s\n", "*           RESIDUAL OR MEAN MASS TRANSPORT VARIABLES");
	fprintf(f7, "%s\n", "*  NFLTMT : USE 1 (FOR RESEARCH PURPOSES)");
	fprintf(f7, "%s\n", "*  NDRYSTP : MIN NO.OF TIME STEPS A CELL REMAINS DRY AFTER INTIAL DRYING");
	fprintf(f7, "%s\n", "*           -NDRYSTP FOR ISDRY = -99 TO ACTIVATE WASTING WATER IN DRY CELLS");
	fprintf(f7, "%s\n", "C7   NTC  NTSPTC    NLTC    NTTC   NTCPP  NTSTBC   NTCNB  NTCVB  NTSMMT  NFLTMT NDRYSTP");
	fprintf(f7, "    %d      %d      %d      %d      %d      %d      %d      %d      %d      %d     %d\n", \
		NTC, NTSPTC, 0, 0, NTCPP, 0, 0, 0, NTCVB, 1, 16);
	fprintf(f7, "%s\n", "-------------------------------------------------------------------------------");
	fprintf(f7, "%s ", "C8");
	fwrite(copia, tamcp, 1, f7); 
	free(copia);                 
	for (pos = ftell(f7); pos < tam; pos++){
		fputc(' ', f7);
	}
	fclose(f7); 
	
	// ====================================================
	//                    C8
	// ====================================================
	FILE *f8 = fopen(filepath, "rb+");
	fseek(f8, 0, SEEK_END);           
	tam = ftell(f8) - 4;              
	rewind(f8);                       	
	pos_C9 = ftell(f8);
	while (fscanf(f8, "%s%*c\n", s8) == 1 && strcmp(s8, "C9") != 0){
		pos_C9 = ftell(f8);
	}
	tamcp = tam - pos_C9;                      
	copia = (char*)malloc(tamcp*sizeof(char)); 
	fread(copia, tamcp, 1, f8);                
	rewind(f8);                           	
	pos_C8 = ftell(f8);
	while (fscanf(f8, "%s%*c", s8) == 1 && strcmp(s8, "C8") != 0){
		pos_C8 = ftell(f8);
	}
	fseek(f8, pos_C8, SEEK_SET); 	
	fprintf(f8, "%s\n", "");
	fprintf(f8, "%s\n", "C8 TIME-RELATED REAL PARAMETERS");
	fprintf(f8, "%s\n", "*");
	fprintf(f8, "%s\n", "*  TCON:     CONVERSION MULTIPLIER TO CHANGE TBEGIN TO SECONDS");
	fprintf(f8, "%s\n", "*  TBEGIN : TIME ORIGIN OF RUN");
	fprintf(f8, "%s\n", "*  TREF : REFERENCE TIME PERIOD IN sec(i.e. 44714.16S OR 86400S)");
	fprintf(f8, "%s\n", "*  CORIOLIS : CONSTANT CORIOLIS PARAMETER IN 1 / sec = 2 * 7.29E-5*SIN(LAT)");
	fprintf(f8, "%s\n", "*  ISCORV : 1 TO READ VARIABLE CORIOLIS COEFFICIENT FROM LXLY.INP FILE");
	fprintf(f8, "%s\n", "*  ISCCA : WRITE DIAGNOSTICS FOR MAX CORIOLIS - CURV ACCEL TO FILEEFDC.LOG");
	fprintf(f8, "%s\n", "*  ISCFL : 1 WRITE DIAGNOSTICS OF MAX THEORETICAL TIME STEP TO CFL.OUT");
	fprintf(f8, "%s\n", "*            GT 1  TIME STEP ONLY AT INTERVAL ISCFL FOR ENTIRE RUN");
	fprintf(f8, "%s\n", "*  ISCFLM : 1  TO MAP LOCATIONS OF MAX TIME STEPS OVER ENTIRE RUN");
	fprintf(f8, "%s\n", "*  DTSSFAC : DYNAMIC TIME STEPPING IF 0.0.LT.DTSSFAC.LT.1.0");
	fprintf(f8, "%s\n", "*");
	fprintf(f8, "%s\n", "C8  TCON  TBEGIN    TREF CORIOLIS ISCORV   ISCCA   ISCFL  ISCFLM DTSSFAC");		
	fprintf(f8, "    %lf      %lf      %lf      %lf      %d      %d      %d      %d      %lf\n", \
		TCON, TBEGIN, TREF, 0.0, 0, 0, 0, 0, 0.0);
	fprintf(f8, "%s\n", "-------------------------------------------------------------------------------");
	fprintf(f8, "%s ", "C9");
	fwrite(copia, tamcp, 1, f8); 
	free(copia);               	
	for (pos = ftell(f8); pos < tam; pos++){
		fputc(' ', f8);
	}
	fclose(f8);
	// ====================================================
	//                    C9
	// ====================================================

	FILE *f9 = fopen(filepath, "rb+");
	fseek(f9, 0, SEEK_END);
	tam = ftell(f9) - 4;
	rewind(f9);
	pos_C10 = ftell(f9);
	while (fscanf(f9, "%s%*c\n", s9) == 1 && strcmp(s9, "C10") != 0){
		pos_C10 = ftell(f9);
	}
	tamcp = tam - pos_C10;
	copia = (char*)malloc(tamcp*sizeof(char));
	fread(copia, tamcp, 1, f9);
	rewind(f9);
	pos_C9 = ftell(f9);
	while (fscanf(f9, "%s%*c", s9) == 1 && strcmp(s9, "C9") != 0){
		pos_C9 = ftell(f9);
	}
	fseek(f9, pos_C9, SEEK_SET);
	fprintf(f9, "%s\n", "");
	fprintf(f9, "%s\n", "C9 SPACE-RELATED AND SMOOTHING  PARAMETERS");
	fprintf(f9, "%s\n", "*");
	fprintf(f9, "%s\n", "*  KC:      NUMBER OF VERTICAL LAYERS");
	fprintf(f9, "%s\n", "*  IC : NUMBER OF CELLS IN I DIRECTION");
	fprintf(f9, "%s\n", "*  JC : NUMBER OF CELLS IN J DIRECTION");
	fprintf(f9, "%s\n", "*  LC : NUMBER OF ACTIVE CELLS IN HORIZONTAL + 2");
	fprintf(f9, "%s\n", "* LVC : NUMBER OF VARIABLE SIZE HORIZONTAL CELLS");
	fprintf(f9, "%s\n", "*  ISCO : 1 FOR CURVILINEAR - ORTHOGONAL GRID(LVC = LC - 2)");
	fprintf(f9, "%s\n", "*  NDM : NUMBER OF DOMAINS FOR HORIZONTAL DOMAIN DECOMPOSITION");
	fprintf(f9, "%s\n", "*           (NDM = 1, FOR MODEL EXECUTION ON A SINGLE PROCESSOR SYSTEM OR");
	fprintf(f9, "%s\n", "*             NDM = MM*NCPUS, WHERE MM IS AN INTEGER AND NCPUS IS THE NUMBER");
	fprintf(f9, "%s\n", "*             OF AVAILABLE CPU'S FOR MODEL EXECUTION ON A PARALLEL MULTIPLE PROCESSOR SYSTEM )");
	fprintf(f9, "%s\n", "*  LDM:     NUMBER OF WATER CELLS PER DOMAIN(LDM = (LC - 2) / NDM, FOR MULTIPE VECTOR PROCESSORS,");
	fprintf(f9, "%s\n", "*LDM MUST BE AN INTEGER MULTIPLE OF THE VECTOR LENGTH OR");
	fprintf(f9, "%s\n", "*             STRIDE NVEC THUS CONSTRAINING LC - 2 TO BE AN INTEGER MULTIPLE OF NVEC)");
	fprintf(f9, "%s\n", "*  ISMASK : 1 FOR MASKING WATER CELL TO LAND OR ADDING THIN BARRIERS");
	fprintf(f9, "%s\n", "*            USING INFORMATION IN FILE MASK.INP");
	fprintf(f9, "%s\n", "*  ISPGNS : 1 FOR IMPLEMENTING A PERIODIC GRID IN COMP N - S DIRECTION OR");
	fprintf(f9, "%s\n", "*            CONNECTING ARBITRATY CELLS USING INFO IN FILE MAPPGNS.INP");
	fprintf(f9, "%s\n", "*  NSHMAX : NUMBER OF DEPTH SMOOTHING PASSES");
	fprintf(f9, "%s\n", "*  NSBMAX : NUMBER OF INITIAL SALINITY FIELD SMOOTHING PASSES");
	fprintf(f9, "%s\n", "*  WSMH : DEPTH SMOOTHING WEIGHT");
	fprintf(f9, "%s\n", "*  WSMB : SALINITY SMOOTHING WEIGHT");
	fprintf(f9, "%s\n", "*");
	fprintf(f9, "%s\n", "*");
	fprintf(f9, "%s\n", "C");
	fprintf(f9, "%s\n", "C9   KC   IC   JC   LC  LVC ISCO  NDM  LDM  ISMASK  ISPGNS  NSHMAX  NSBMAX    WSMH    WSMB");
	fprintf(f9, "    %d      %d      %d      %d      %d      %d      %d      %d     %d     %d     %d     %d     %lf     %lf\n", \
	                 nlayers, ncols, nrows, nwcells+2, nwcells, 1, 1, nwcells, 0, 0, 0, 0, 0.03125, 0.06250);
	fprintf(f9, "%s\n", "-------------------------------------------------------------------------------");
	fprintf(f9, "%s ", "C10");
	fwrite(copia, tamcp, 1, f9);
	free(copia);
	for (pos = ftell(f9); pos < tam; pos++){
		fputc(' ', f9);
	}
	fclose(f9);

	// ====================================================
	//                    C10
	// ====================================================

	FILE *f10 = fopen(filepath, "rb+");
	fseek(f10, 0, SEEK_END);
	tam = ftell(f10) - 4;
	rewind(f10);
	pos_C11 = ftell(f10);
	while (fscanf(f10, "%s%*c\n", s10) == 1 && strcmp(s10, "C11") != 0){
		pos_C11 = ftell(f10);
	}
	tamcp = tam - pos_C11;
	copia = (char*)malloc(tamcp*sizeof(char));
	fread(copia, tamcp, 1, f10);
	rewind(f10);
	pos_C10 = ftell(f10);
	while (fscanf(f10, "%s%*c", s10) == 1 && strcmp(s10, "C10") != 0){
		pos_C10 = ftell(f10);
	}
	fseek(f10, pos_C10, SEEK_SET);
	//fprintf(f10, "%s\n", "");
	fprintf(f10, "%s\n", "C10 LAYER THICKNESS IN VERTICAL");
	fprintf(f10, "%s\n", "*");
	fprintf(f10, "%s\n", "*    K:  LAYER NUMBER, K = 1, KC");
	fprintf(f10, "%s\n", "*  DZC : DIMENSIONLESS LAYER THICKNESS(THICKNESSES MUST SUM TO 1.0)");
	fprintf(f10, "%s\n", "*");
	fprintf(f10, "%s\n", "C10  K   DZC");
	for (nl = 0; nl < nlayers; nl++){
		fprintf(f10, "    %d      %lf\n", nl + 1, (1.0000 / nlayers));
	}
	fprintf(f10, "%s\n", "-------------------------------------------------------------------------------");
	fprintf(f10, "%s ", "C11");
	fwrite(copia, tamcp, 1, f10);
	free(copia);
	for (pos = ftell(f10); pos < tam; pos++){
		fputc(' ', f10);
	}
	fclose(f10);
	
	// ====================================================
	//                    C11
	// ====================================================

	FILE *f11 = fopen(filepath, "rb+");
	fseek(f11, 0, SEEK_END);
	tam = ftell(f11) - 4;
	rewind(f11);
	pos_C11A = ftell(f11);
	while (fscanf(f11, "%s%*c\n", s11) == 1 && strcmp(s11, "C11A") != 0){
		pos_C11A = ftell(f11);
	}
	tamcp = tam - pos_C11A;
	copia = (char*)malloc(tamcp*sizeof(char));
	fread(copia, tamcp, 1, f11);
	rewind(f11);
	pos_C11 = ftell(f11);
	while (fscanf(f11, "%s%*c", s11) == 1 && strcmp(s11, "C11") != 0){
		pos_C11 = ftell(f11);
	}
	fseek(f11, pos_C11, SEEK_SET);
	//fprintf(f11, "%s\n", "");
	fprintf(f11, "%s\n", "C11 GRID, ROUGHNESS AND DEPTH PARAMETERS");  
	fprintf(f11, "%s\n", "*");
	fprintf(f11, "%s\n", "*  DX:       CARTESIAN CELL LENGTH IN X OR I DIRECTION");  
	fprintf(f11, "%s\n", "*  DY:       CARTESION CELL LENGHT IN Y OR J DIRECTION");  
	fprintf(f11, "%s\n", "*  DXYCVT:   MULTIPLY DX AND DY BY TO OBTAIN METERS");  
	fprintf(f11, "%s\n", "*  IMD:      GREATER THAN 0 TO READ MODDXDY.INP FILE");  
	fprintf(f11, "%s\n", "*  ZBRADJ:   LOG BDRY LAYER CONST OR VARIABLE ROUGH HEIGHT ADJ IN METERS");  
	fprintf(f11, "%s\n", "*  ZBRCVRT:  LOG BDRY LAYER VARIABLE ROUGHNESS HEIGHT CONVERT TO METERS");  
	fprintf(f11, "%s\n", "*  HMIN:     MINIMUM DEPTH OF INPUTS DEPTHS IN METERS");  
	fprintf(f11, "%s\n", "*  HADJ:     ADJUCTMENT TO DEPTH FIELD IN METERS");  
	fprintf(f11, "%s\n", "*  HCVRT:    CONVERTS INPUT DEPTH FIELD TO METERS");  
	fprintf(f11, "%s\n", "*  HDRY:     DEPTH AT WHICH CELL OR FLOW FACE BECOMES DRY");  
	fprintf(f11, "%s\n", "*  HWET:     DEPTH AT WHICH CELL OR FLOW FACE BECOMES WET");  
	fprintf(f11, "%s\n", "*  BELADJ:   ADJUCTMENT TO BOTTOM BED ELEVATION FIELD IN METERS");  
	fprintf(f11, "%s\n", "*  BELCVRT:  CONVERTS INPUT BOTTOM BED ELEVATION FIELD TO METERS");  
	fprintf(f11, "%s\n", "*");  
	fprintf(f11, "%s\n", "C11   DX      DY   DXYCVT    IMD  ZBRADJ ZBRCVRT    HMIN    HADJ   HCVRT    HDRY    HWET  BELADJ BELCVRT");  
 	fprintf(f11, "    %lf    %lf    %lf    %d    %lf    %lf    %lf    %lf    %lf    %lf    %lf    %lf    %lf\n",\
        	          DX,   DY,  1.00,  0,    ZBRADJ,   ZBRCVRT,   HMIN,  HADJ,   HCVRT,   HDRY,  HWET,  BELADJ,   BELCVRT);
	
	fprintf(f11, "%s\n", "-------------------------------------------------------------------------------");
	fprintf(f11, "%s ", "C11A");
	fwrite(copia, tamcp, 1, f11);
	free(copia);
	for (pos = ftell(f11); pos < tam; pos++){
		fputc(' ', f11);
	}
	fclose(f11);	
	
	// ====================================================
	//                    C14
	// ====================================================

	FILE *f14 = fopen(filepath, "rb+");
	fseek(f14, 0, SEEK_END);
	tam = ftell(f14) - 4;
	rewind(f14);
	pos_C15 = ftell(f14);
	while (fscanf(f14, "%s%*c\n", s14) == 1 && strcmp(s14, "C15") != 0){
		pos_C15 = ftell(f14);
	}
	tamcp = tam - pos_C15;
	copia = (char*)malloc(tamcp*sizeof(char));
	fread(copia, tamcp, 1, f14);
	rewind(f14);
	pos_C14 = ftell(f14);
	while (fscanf(f14, "%s%*c", s14) == 1 && strcmp(s14, "C14") != 0){
		pos_C14 = ftell(f14);
	}
	fseek(f14, pos_C14, SEEK_SET);
	fprintf(f14, "%s\n", "");
	fprintf(f14, "%s\n", "C14 TIDAL & ATMOSPHERIC FORCING, GROUND WATER AND SUBGRID CHANNEL PARAMETERS");  
	fprintf(f14, "%s\n", "*");  
	fprintf(f14, "%s\n", "*   MTIDE:     NUMBER OF PERIOD (TIDAL) FORCING CONSTITUENTS");  
	fprintf(f14, "%s\n", "*   NWSER:     NUMBER OF WIND TIME SERIES (0 SETS WIND TO ZERO)");  
	fprintf(f14, "%s\n", "*   NASER:     NUMBER OF ATMOSPHERIC CONDITION TIME SERIES (0 SETS ALL  ZERO)");  
	fprintf(f14, "%s\n", "*   ISGWI:     1 TO ACTIVATE SOIL MOISTURE BALANCE WITH DRYING AND WETTING");  
	fprintf(f14, "%s\n", "*              2 TO ACTIVATE GROUNDWATER INTERACTION WITH BED AND WATER COL");  
	fprintf(f14, "%s\n", "*  ISCHAN:    >0 ACTIVATE SUBGRID CHANNEL MODEL AND READ MODCHAN.INP");  
	fprintf(f14, "%s\n", "*  ISWAVE:     1-FOR BL IMPACTS (WAVEBL.INP), 2-FOR BL & CURRENT IMPACTS (WAVE.INP)");  
	fprintf(f14, "%s\n", "*              3-FOR INTERNALLY COMPUTED WIND WAVE BOUNDARY LAYER IMPACTS (DS)");  
	fprintf(f14, "%s\n", "* ITIDASM:     1 FOR TIDAL ELEVATION ASSIMILATION (NOT ACTIVE)");  
	fprintf(f14, "%s\n", "*  ISPERC:     1 TO PERCOLATE OR ELIMINATE EXCESS WATER IN DRY CELLS");  
	fprintf(f14, "%s\n", "* ISBODYF:     TO INCLUDE EXTERNAL MODE BODY FORCES FROM FBODY.INP");  
	fprintf(f14, "%s\n", "*              1 FOR UNIFORM OVER DEPTH, 2 FOR SURFACE LAYER ONLY");  
	fprintf(f14, "%s\n", "* ISPNHYDS: 1 FOR QUASI-NONHYDROSTATIC OPTION");  
	fprintf(f14, "%s\n", "*");  
	fprintf(f14, "%s\n", "C14 MTIDE  NWSER   NASER   ISGWI  ISCHAN  ISWAVE ITIDASM  ISPERC ISBODYF ISPNHYDS");  
	           fprintf(f14, "    %d    %d    %d    %d    %d    %d    %d    %d    %d    %d\n",\
        	                      0,    NWSER,NASER,0,    0,    0,    0,    0,    0,    0);	
	fprintf(f14, "%s\n", "-------------------------------------------------------------------------------");
	fprintf(f14, "%s ", "C15");
	fwrite(copia, tamcp, 1, f14);
	free(copia);
	for (pos = ftell(f14); pos < tam; pos++){
		fputc(' ', f14);
	}
	fclose(f14);	
	
	// ====================================================
	//                    C22
	// ====================================================
	FILE *f22 = fopen(filepath, "rb+");
	fseek(f22, 0, SEEK_END); 
	tam = ftell(f22) - 4; 
	rewind(f22); 	
	pos_C23 = ftell(f22);
	while (fscanf(f22, "%s%*c\n", s22) == 1 && strcmp(s22, "C23") != 0){
		pos_C23 = ftell(f22);
	}
	tamcp = tam - pos_C23; 
	copia = (char*)malloc(tamcp*sizeof(char)); 
	fread(copia, tamcp, 1, f22); 
	rewind(f22);
	pos_C22 = ftell(f22);
	while (fscanf(f22, "%s%*c", s22) == 1 && strcmp(s22, "C22") != 0){
		pos_C22 = ftell(f22);
	}
	fseek(f22, pos_C22, SEEK_SET); 
	fprintf(f22, "%s\n", "");
	fprintf(f22, "%s\n", "C22 SPECIFY NUM OF SEDIMENT AND TOXICS AND NUM OF CONCENTRATION TIME SERIES");  
	fprintf(f22, "%s\n", "* "); 
	fprintf(f22, "%s\n", "*  NTOX:   NUMBER OF TOXIC CONTAMINANTS (DEFAULT = 1)");  
	fprintf(f22, "%s\n", "*  NSED:   NUMBER OF COHESIVE SEDIMENT SIZE CLASSES (DEFAULT = 1)");  
	fprintf(f22, "%s\n", "*  NSND:   NUMBER OF NON-COHESIVE SEDIMENT SIZE CLASSES (DEFAULT = 1)");  
	fprintf(f22, "%s\n", "*  NCSER1: NUMBER OF SALINITY TIME SERIES");  
	fprintf(f22, "%s\n", "*  NCSER2: NUMBER OF TEMPERATURE TIME SERIES");  
	fprintf(f22, "%s\n", "*  NCSER3: NUMBER OF DYE CONCENTRATION TIME SERIES");  
	fprintf(f22, "%s\n", "*  NCSER4: NUMBER OF SHELLFISH LARVAE CONCENTRATION TIME SERIES");  
	fprintf(f22, "%s\n", "*  NCSER5: NUMBER OF TOXIC CONTAMINANT CONCENTRATION TIME SERIES");  
	fprintf(f22, "%s\n", "*          EACH TIME SERIES MUST HAVE DATA FOR NTOX TOXICICANTS");  
	fprintf(f22, "%s\n", "*  NCSER6: NUMBER OF COHESIVE SEDIMENT CONCENTRATION TIME SERIES");  
	fprintf(f22, "%s\n", "*          EACH TIME SERIES MUST HAVE DATA FOR NSED COHESIVE SEDIMENTS");  
	fprintf(f22, "%s\n", "*  NCSER7: NUMBER OF NON-COHESIVE SEDIMENT CONCENTRATION TIME SERIES");  
	fprintf(f22, "%s\n", "*          EACH TIME SERIES MUST HAVE DATA FOR NSND NON-COHESIVE SEDIMENTS");  
	fprintf(f22, "%s\n", "*  ISSBAL: SET TO 1 FOR SEDIENT MASS BALANCE           ! JOHN & JI, 4/25/97");  
	fprintf(f22, "%s\n", "*");  
	fprintf(f22, "%s\n", "C22 NTOX    NSED    NSND  NCSER1  NCSER2  NCSER3  NCSER4  NCSER5  NCSER6  NCSER7  ISSBAL");
	fprintf(f22, "    %d      %d      %d      %d      %d      %d      %d      %d      %d      %d      %d\n", \
		0, 0, 0, salt_on*nqser, temp_on*nqser, dye_on*nqser, 0, 0, 0, 0, 0);

	fprintf(f22, "%s\n", "-------------------------------------------------------------------------------");
	fprintf(f22, "%s ", "C23");
	fwrite(copia, tamcp, 1, f22); 
	free(copia); 

	for (pos = ftell(f22); pos < tam; pos++){
		fputc(' ', f22);
	}
	fclose(f22); 

	// ====================================================
	//                    C23
	// ====================================================
	FILE *f23 = fopen(filepath, "rb+");
	fseek(f23, 0, SEEK_END); 
	tam = ftell(f23) - 4; 
	rewind(f23); 	
	pos_C24 = ftell(f23);
	while (fscanf(f23, "%s%*c\n", s23) == 1 && strcmp(s23, "C24") != 0){
		pos_C24 = ftell(f23);
	}
	tamcp = tam - pos_C24; 
	copia = (char*)malloc(tamcp*sizeof(char)); 
	fread(copia, tamcp, 1, f23); 
	rewind(f23);
	pos_C23 = ftell(f23);
	while (fscanf(f23, "%s%*c", s23) == 1 && strcmp(s23, "C23") != 0){
		pos_C23 = ftell(f23);
	}
	fseek(f23, pos_C23, SEEK_SET); 
	fprintf(f23, "%s\n", "");
	fprintf(f23, "%s\n", "C23 VELOCITY, VOLUMN SOURCE / SINK, FLOW CONTROL, AND WITHDRAWAL / RETURN DATA");
	fprintf(f23, "%s\n", "*");
	fprintf(f23, "%s\n", "*  NVBS:   VEL BC(NOT USED)");
	fprintf(f23, "%s\n", "*  NUBW : VEL BC(NOT USED)");
	fprintf(f23, "%s\n", "*  NUBE : VEL BC(NOT USED)");
	fprintf(f23, "%s\n", "*  NVBN : VEL BC(NOT USED)");
	fprintf(f23, "%s\n", "*  NQSIJ : NUMBER OF CONSTANT AND / OR TIME SERIES SPECIFIED SOURCE / SINK");
	fprintf(f23, "%s\n", "*          LOCATIONS(RIVER INFLOWS, ETC)             .");
	fprintf(f23, "%s\n", "*  NQJPIJ : NUMBER OF CONSTANT AND / OR TIME SERIES SPECIFIED SOURCE");
	fprintf(f23, "%s\n", "*          LOCATIONS TREATED AS JETS / PLUMES          .");
	fprintf(f23, "%s\n", "*  NQSER : NUMBER OF VOLUME SOURCE / SINK TIME SERIES");
	fprintf(f23, "%s\n", "*  NQCTL : NUMBER OF PRESSURE CONTROLED WITHDRAWAL / RETURN PAIRS");
	fprintf(f23, "%s\n", "*  NQCTLT : NUMBER OF PRESSURE CONTROLED WITHDRAWAL / RETURN TABLES");
	fprintf(f23, "%s\n", "*  NQWR : NUMBER OF CONSTANT OR TIME SERIES SPECIFIED WITHDRAWL / RETURN");
	fprintf(f23, "%s\n", "*          PAIRS");
	fprintf(f23, "%s\n", "*  NQWRSR : NUMBER OF TIME SERIES SPECIFYING WITHDRAWL, RETURN AND");
	fprintf(f23, "%s\n", "*          CONCENTRATION RISE SERIES");
	fprintf(f23, "%s\n", "*  ISDIQ : SET TO 1 TO WRITE DIAGNOSTIC FILE, DIAQ.OUT");
	fprintf(f23, "%s\n", "*");
	fprintf(f23, "%s\n", "C23 NVBS    NUBW    NUBE    NVBN   NQSIJ  NQJPIJ   NQSER   NQCTL  NQCTLT    NQWR  NQWRSR   ISDIQ");

	fprintf(f23, "    %d      %d      %d      %d      %d      %d      %d      %d      %d      %d     %d      %d\n", \
		0, 0, 0, 0, nqsij[0], 0, nqser, 0, 0, 0, 0, 0);

	fprintf(f23, "%s\n", "-------------------------------------------------------------------------------");
	fprintf(f23, "%s ", "C24");
	fwrite(copia, tamcp, 1, f23); 
	free(copia); 

	for (pos = ftell(f23); pos < tam; pos++){
		fputc(' ', f23);
	}
	fclose(f23); 

	// ====================================================
	//                    C24
	// ====================================================
	FILE *f24 = fopen(filepath, "rb+");
	fseek(f24, 0, SEEK_END); 
	tam = ftell(f24) - 4; 
	rewind(f24); 	
	pos_C25 = ftell(f24);
	while (fscanf(f24, "%s%*c\n", s24) == 1 && strcmp(s24, "C25") != 0){
		pos_C25 = ftell(f24);
	}
	tamcp = tam - pos_C25; 
	copia = (char*)malloc(tamcp*sizeof(char)); 
	fread(copia, tamcp, 1, f24); 
	rewind(f24); 
	pos_C24 = ftell(f24);
	while (fscanf(f24, "%s%*c", s24) == 1 && strcmp(s24, "C24") != 0){
		pos_C24 = ftell(f24);
	}
	fseek(f24, pos_C24, SEEK_SET); 
	//fprintf(f24, "%s\n", "");
	fprintf(f24, "%s\n", "C24 VOLUMETRIC SOURCE / SINK LOCATIONS, MAGNITUDES, AND CONCENTRATION SERIES");
	fprintf(f24, "%s\n", "*");
	fprintf(f24, "%s\n", "*  IQS:      I CELL INDEX OF VOLUME SOURCE / SINK");
	fprintf(f24, "%s\n", "*  JQS : J CELL INDEX OF VOLUME SOURCE / SINK");
	fprintf(f24, "%s\n", "*  QSSE : CONSTANT INFLOW / OUTFLOW RATE IN M*m*m / s");
	fprintf(f24, "%s\n", "*  NQSMUL : MULTIPLIER SWITCH FOR CONSTANT AND TIME SERIES VOL S / S");
	fprintf(f24, "%s\n", "* = 0  MULT BY 1. FOR NORMAL IN / OUTFLOW(L*L*L / T)");
	fprintf(f24, "%s\n", "* = 1  MULT BY DY FOR LATERAL IN / OUTFLOW(L*L / T) ON U FACE");
	fprintf(f24, "%s\n", "* = 2  MULT BY DX FOR LATERAL IN / OUTFLOW(L*L / T) ON V FACE");
	fprintf(f24, "%s\n", "* = 3  MULT BY DX + DY FOR LATERAL IN / OUTFLOW(L*L / T) ON U&V FACES");
	fprintf(f24, "%s\n", "*  NQSMFF : IF NON ZERO ACCOUNT FOR VOL S / S MOMENTUM FLUX");
	fprintf(f24, "%s\n", "* = 1  MOMENTUM FLUX ON NEG U FACE");
	fprintf(f24, "%s\n", "* = 2  MOMENTUM FLUX ON NEG V FACE");
	fprintf(f24, "%s\n", "* = 3  MOMENTUM FLUX ON POS U FACE");
	fprintf(f24, "%s\n", "* = 4  MOMENTUM FLUX ON POS V FACE");
	fprintf(f24, "%s\n", "*  IQSERQ : ID NUMBER OF ASSOCIATED VOLUMN FLOW TIME SERIES");
	fprintf(f24, "%s\n", "*  ICSER1 : ID NUMBER OF ASSOCIATED SALINITY TIME SERIES");
	fprintf(f24, "%s\n", "*  ICSER2 : ID NUMBER OF ASSOCIATED TEMPERATURE TIME SERIES");
	fprintf(f24, "%s\n", "*  ICSER3 : ID NUMBER OF ASSOCIATED DYE CONC TIME SERIES");
	fprintf(f24, "%s\n", "*  ICSER4 : ID NUMBER OF ASSOCIATED SHELL FISH LARVAE RELEASE TIME SERIES");
	fprintf(f24, "%s\n", "*  ICSER5 : ID NUMBER OF ASSOCIATED TOXIC CONTAMINANT CONC TIME SERIES");
	fprintf(f24, "%s\n", "*  ICSER6 : ID NUMBER OF ASSOCIATED COHESIVE SEDIMENT CONC TIME SERIES");
	fprintf(f24, "%s\n", "*  ICSER7 : ID NUMBER OF ASSOCIATED NON - COHESIVE SED CONC TIME SERIES");
	fprintf(f24, "%s\n", "*  QSFACTOR : FRACTION OF TIME SERIES FLOW NQSERQ ASSIGNED TO THIS CELL");
	fprintf(f24, "%s\n", "*");
	fprintf(f24, "%s\n", "C24  IQS     JQS     QSSE     NQSMUL  NQSMFF  IQSERQ  ICSER1  ICSER2  ICSER3  ICSER4  ICSER5  ICSER6  ICSER7   QSFACTOR  !ID");
	bn = 0;
	while (north_bc[bn][0] != 0){
		nqs = nqs + 1;		

			fprintf(f24, "     %d      %d      %lf      %d      %d      %d      %d      %d      %d      %d      %d      %d      %d      %lf      %s\n", \
			north_bc[bn][0], north_bc[bn][1], 0.000, 0, 0, nqs, salt_on*nqs, temp_on*nqs, dye_on*nqs, 0, 0, 0, 0, 1.000000, "! North_bc");			
				
		
		bn = bn + 1;
	}
	bs = 0;
	while (south_bc[bs][0] != 0){
		nqs = nqs + 1;
		
		    fprintf(f24, "     %d      %d      %lf      %d      %d      %d      %d      %d      %d      %d      %d      %d      %d      %lf      %s\n", \
			south_bc[bs][0], south_bc[bs][1], 0.000, 0, 0, nqs, salt_on*nqs, temp_on*nqs, dye_on*nqs, 0, 0, 0, 0, 1.000000, "! South_bc");

		bs = bs + 1;
	}
	be = 0;
	while (east_bc[be][0] != 0){
		nqs = nqs + 1;

		    fprintf(f24, "     %d      %d      %lf      %d      %d      %d      %d      %d      %d      %d      %d      %d      %d      %lf      %s\n", \
			east_bc[be][0], east_bc[be][1], 0.000, 0, 0, nqs, salt_on*nqs, temp_on*nqs, dye_on*nqs, 0, 0, 0, 0, 1.000000, "! East_bc");


		be = be + 1;
	}
	bw = 0;
	while (west_bc[bw][0] != 0){
		nqs = nqs + 1;

		fprintf(f24, "     %d      %d      %lf      %d      %d      %d      %d      %d      %d      %d      %d      %d      %d      %lf      %s\n", \
			west_bc[bw][0], west_bc[bw][1], 0.000, 0, 0, nqs, salt_on*nqs, temp_on*nqs, dye_on*nqs, 0, 0, 0, 0, 1.000000, "! West_bc");
		

		bw = bw + 1;
	}

	fprintf(f24, "%s\n", "-------------------------------------------------------------------------------");
	fprintf(f24, "%s ", "C25");
	fwrite(copia, tamcp, 1, f24); 
	free(copia); 
	for (pos = ftell(f24); pos < tam; pos++){
		fputc(' ', f24);
	}
	fclose(f24); 

	// ====================================================
	//                    C25
	// ====================================================
	FILE *f25 = fopen(filepath, "rb+");
	fseek(f25, 0, SEEK_END);
	tam = ftell(f25) - 4; 
	rewind(f25); 	
	pos_C26 = ftell(f25);
	while (fscanf(f25, "%s%*c\n", s25) == 1 && strcmp(s25, "C26") != 0){
		pos_C26 = ftell(f25);
	}
	tamcp = tam - pos_C26; 
	copia = (char*)malloc(tamcp*sizeof(char)); 
	fread(copia, tamcp, 1, f25); 
	rewind(f25);
	pos_C25 = ftell(f25);
	while (fscanf(f25, "%s%*c", s25) == 1 && strcmp(s25, "C25") != 0){
		pos_C25 = ftell(f25);
	}
	fseek(f25, pos_C25, SEEK_SET); 
	//fprintf(f25, "%s\n", "");
	fprintf(f25, "%s\n", "C25 TIME CONSTANT INFLOW CONCENTRATIONS FOR TIME CONSTANT VOLUMETRIC SOURCES");
	fprintf(f25, "%s\n", "*");
	fprintf(f25, "%s\n", "*  SAL: SALT CONCENTRATION CORRESPONDING TO INFLOW ABOVE");
	fprintf(f25, "%s\n", "*  TEM : TEMPERATURE CORRESPONDING TO INFLOW ABOVE");
	fprintf(f25, "%s\n", "*  DYE : DYE CONCENTRATION CORRESPONDING TO INFLOW ABOVE");
	fprintf(f25, "%s\n", "*  SFL : SHELL FISH LARVAE CONCENTRATION CORRESPONDING TO INFLOW ABOVE");
	fprintf(f25, "%s\n", "*  TOX : NTOX TOXIC CONTAMINANT CONCENTRATIONS CORRESPONDING TO");
	fprintf(f25, "%s\n", "*       INFLOW ABOVE  WRITTEN AS TOXC(N), N = 1, NTOX A SINGLE DEFAULT");
	fprintf(f25, "%s\n", "*       VALUE IS REQUIRED EVEN IF TOXIC TRANSPORT IS NOT ACTIVE");
	fprintf(f25, "%s\n", "*");
	fprintf(f25, "%s\n", "C25    SAL       TEM       DYE       SFL  !ID");
	for (i = 0; i < nqs; i++){
		fprintf(f25, "     %lf      %lf      %lf      %lf  %s%d\n", \
			0.00, 0.00, 0.00, 0.00, "!IQSERQ_", i + 1);
	}
	fprintf(f25, "%s\n", "-------------------------------------------------------------------------------");
	fprintf(f25, "%s ", "C26");
	fwrite(copia, tamcp, 1, f25); 
	free(copia); 

	for (pos = ftell(f25); pos < tam; pos++){
		fputc(' ', f25);
	}
	fclose(f25); 

	// ====================================================
	//                    C26
	// ====================================================
	FILE *f26 = fopen(filepath, "rb+");
	fseek(f26, 0, SEEK_END); 
	tam = ftell(f26) - 4; 
	rewind(f26); 	
	pos_C27 = ftell(f26);
	while (fscanf(f26, "%s%*c\n", s26) == 1 && strcmp(s26, "C27") != 0){
		pos_C27 = ftell(f26);
	}
	tamcp = tam - pos_C27; 
	copia = (char*)malloc(tamcp*sizeof(char)); 
	fread(copia, tamcp, 1, f26);
	rewind(f26); 
	pos_C26 = ftell(f26);
	while (fscanf(f26, "%s%*c", s26) == 1 && strcmp(s26, "C26") != 0){
		pos_C26 = ftell(f26);
	}
	fseek(f26, pos_C26, SEEK_SET); 
	//fprintf(f26, "%s\n", "");
	fprintf(f26, "%s\n", "C26 TIME CONSTANT INFLOW CONCENTRATIONS FOR TIME CONSTANT VOLUMETRIC SOURCES");
	fprintf(f26, "%s\n", "*");
	fprintf(f26, "%s\n", "*  SED: NSED COHESIVE SEDIMENT CONCENTRATIONS CORRESPONDING TO");
	fprintf(f26, "%s\n", "*       INFLOW ABOVE  WRITTEN AS SEDC(N), N = 1, NSED.I.E., THE FIRST");
	fprintf(f26, "%s\n", "*       NSED VALUES ARE COHESIVE A SINGLE DEFAULT VALUE IS REQUIRED");
	fprintf(f26, "%s\n", "*       EVEN IF COHESIVE SEDIMENT TRANSPORT IS INACTIVE");
	fprintf(f26, "%s\n", "*  SND : NSND NON - COHESIVE SEDIMENT CONCENTRATIONS CORRESPONDING TO");
	fprintf(f26, "%s\n", "*       INFLOW ABOVE  WRITTEN AS SND(N), N = 1, NSND.I.E., THE LAST");
	fprintf(f26, "%s\n", "*       NSND VALUES ARE NON - COHESIVE.A SINGLE DEFAULT VALUE IS");
	fprintf(f26, "%s\n", "*       REQUIRED EVEN IF NON - COHESIVE SEDIMENT TRANSPORT IS INACTIVE");
	fprintf(f26, "%s\n", "*");
	fprintf(f26, "%s\n", "C26   SED1    SND1");
	for (i = 0; i < nqs; i++){
		fprintf(f26, "     %d      %d\n", \
			0, 0);
	}
	fprintf(f26, "%s\n", "-------------------------------------------------------------------------------");
	fprintf(f26, "%s ", "C27");
	fwrite(copia, tamcp, 1, f26); 
	free(copia); 
	for (pos = ftell(f26); pos < tam; pos++){
		fputc(' ', f26);
	}
	fclose(f26); 	
	
	// ====================================================
	//                    C71
	// ====================================================
	
	FILE *f71 = fopen(filepath, "rb+");
	fseek(f71, 0, SEEK_END);
	tam = ftell(f71) - 4;
	rewind(f71);
	pos_C71A = ftell(f71);
	while (fscanf(f71, "%s%*c\n", s71) == 1 && strcmp(s71, "C71A") != 0){
		pos_C71A = ftell(f71);
	}
	tamcp = tam - pos_C71A;
	copia = (char*)malloc(tamcp*sizeof(char));
	fread(copia, tamcp, 1, f71);
	rewind(f71);
	pos_C71 = ftell(f71);
	while (fscanf(f71, "%s%*c", s71) == 1 && strcmp(s71, "C71") != 0){
		pos_C71 = ftell(f71);
	}
	fseek(f71, pos_C71, SEEK_SET);
	fprintf(f71, "%s\n", "");
	fprintf(f71, "%s\n", "C71 CONTROLS FOR HORIZONTAL PLANE SCALAR FIELD CONTOURING");  
	fprintf(f71, "%s\n", "*");  
	fprintf(f71, "%s\n", "*  ISSPH:  1 TO WRITE FILE FOR SCALAR FIELD CONTOURING IN HORIZONTAL PLANE");  
	fprintf(f71, "%s\n", "*          2 WRITE ONLY DURING LAST REFERENCE TIME PERIOD");  
	fprintf(f71, "%s\n", "*  NPSPH:    NUMBER OF WRITES PER REFERENCE TIME PERIOD");  
	fprintf(f71, "%s\n", "*  ISRSPH: 1 TO WRITE FILE FOR RESIDUAL SALINITY PLOTTING IN");  
	fprintf(f71, "%s\n", "*            HORIZONTAL");  
	fprintf(f71, "%s\n", "*  ISPHXY: 0 DOES NOT WRITE I,J,X,Y IN ***CNH.OUT AND R***CNH.OUT FILES");  
	fprintf(f71, "%s\n", "*          1 WRITES I,J ONLY IN ***CNH.OUT AND R***CNH.OUT FILES");  
	fprintf(f71, "%s\n", "*          2 WRITES I,J,X,Y  IN ***CNH.OUT AND R***CNH.OUT FILES");  
	fprintf(f71, "%s\n", "*          3 WRITES EFDC_EXPLORER BINARY FORMAT FILES");  
	fprintf(f71, "%s\n", "*  DATA LINE REPEATS 7 TIMES FOR SAL,TEM,DYE,SFL,TOX,SED,SND");  
	fprintf(f71, "%s\n", "*");  
	fprintf(f71, "%s\n", "C71 ISSPH NPSPH ISRSPH ISPHXY"); 	
	
    fprintf(f71, "     %d    %d     %d     %d    %s\n", \
		salt_on, 1, 0, 3, "!SAL");
	fprintf(f71, "     %d    %d     %d     %d    %s\n", \
		temp_on, 1, 0, 3, "!TEM");
	fprintf(f71, "     %d    %d     %d     %d    %s\n", \
		dye_on, 1, 0, 3, "!DYE");
	fprintf(f71, "     %d    %d     %d     %d    %s\n", \
		0, 1, 0, 3, "!EE WC/Sediment Top Layer Flag");
	fprintf(f71, "     %d    %d     %d     %d    %s\n", \
		0, 1, 0, 3, "!TOX");
	fprintf(f71, "     %d    %d     %d     %d    %s\n", \
		0, 1, 0, 3, "!SED");
	fprintf(f71, "     %d    %d     %d     %d    %s\n", \
		0, 1, 0, 3, "!SND");
		
	fprintf(f71, "%s\n", "-------------------------------------------------------------------------------");
	fprintf(f71, "%s ", "C71A");
	fwrite(copia, tamcp, 1, f71);
	free(copia);
	for (pos = ftell(f71); pos < tam; pos++){
		fputc(' ', f71);
	}
	fclose(f71);

	

	// ====================================================
	//                    C72
	// ====================================================
	FILE *f72 = fopen(filepath, "rb+");
	fseek(f72, 0, SEEK_END);
	tam = ftell(f72) - 4;
	rewind(f72);
	pos_C73 = ftell(f72);
	while (fscanf(f72, "%s%*c\n", s72) == 1 && strcmp(s72, "C73") != 0){
		pos_C73 = ftell(f72);
	}
	tamcp = tam - pos_C73;
	copia = (char*)malloc(tamcp*sizeof(char));
	fread(copia, tamcp, 1, f72);
	rewind(f72);
	pos_C72 = ftell(f72);
	while (fscanf(f72, "%s%*c", s72) == 1 && strcmp(s72, "C72") != 0){
		pos_C72 = ftell(f72);
	}
	fseek(f72, pos_C72, SEEK_SET);
	fprintf(f72, "%s\n", "");
	fprintf(f72, "%s\n", "C72 CONTROLS FOR HORIZONTAL SURFACE ELEVATION OR PRESSURE CONTOURING");
	fprintf(f72, "%s\n", "*");
	fprintf(f72, "%s\n", "*  ISPPH:  1 TO WRITE FILE FOR SURFACE ELEVATION OR PRESSURE CONTOURING");
	fprintf(f72, "%s\n", "* 2 WRITE ONLY DURING LAST REFERENCE TIME PERIOD");
	fprintf(f72, "%s\n", "*  NPPPH : NUMBER OF WRITES PER REFERENCE TIME PERIOD");
	fprintf(f72, "%s\n", "*  ISRPPH : 1 TO WRITE FILE FOR RESIDUAL SURFACE ELEVATION  CONTOURNG IN");
	fprintf(f72, "%s\n", "*            HORIZONTAL PLANE");
	fprintf(f72, "%s\n", "*  IPPHXY : 0 DOES NOT WRITE I, J, X, Y IN surfplt.out and rsurfplt.out FILES");
	fprintf(f72, "%s\n", "* 1 WRITES I, J ONLY IN surfplt.out and rsurfplt.out FILES");
	fprintf(f72, "%s\n", "* 2 WRITES I, J, X, Y  IN surfplt.out and rsurfplt.out FILES");
	fprintf(f72, "%s\n", "* 3 WRITES EFDC EXPLORER BINARY FORMAT FILES");
	fprintf(f72, "%s\n", "*");
	fprintf(f72, "%s\n", "C72 ISPPH NPPPH ISRPPH IPPHXY");		
    fprintf(f72, "     %d    %d     %d     %d\n", \
		ISPPH, NPPPH, ISRPPH, IPPHXY);

	fprintf(f72, "%s\n", "-------------------------------------------------------------------------------");
	fprintf(f72, "%s ", "C73");
	fwrite(copia, tamcp, 1, f72);
	free(copia);
	for (pos = ftell(f72); pos < tam; pos++){
		fputc(' ', f72);
	}
	fclose(f72);

	// ====================================================
	//                    C73
	// ====================================================
	FILE *f73 = fopen(filepath, "rb+");
	fseek(f73, 0, SEEK_END);
	tam = ftell(f73) - 4;
	rewind(f73);
	pos_C74 = ftell(f73);
	while (fscanf(f73, "%s%*c\n", s73) == 1 && strcmp(s73, "C74") != 0){
		pos_C74 = ftell(f73);
	}
	tamcp = tam - pos_C74;
	copia = (char*)malloc(tamcp*sizeof(char));
	fread(copia, tamcp, 1, f73);
	rewind(f73);
	pos_C73 = ftell(f73);
	while (fscanf(f73, "%s%*c", s73) == 1 && strcmp(s73, "C73") != 0){
		pos_C73 = ftell(f73);
	}
	fseek(f73, pos_C73, SEEK_SET);
	//fprintf(f73, "%s\n", "");
	fprintf(f73, "%s\n", "C73 CONTROLS FOR HORIZONTAL PLANE VELOCITY VECTOR PLOTTING");
	fprintf(f73, "%s\n", "*");
	fprintf(f73, "%s\n", "*  ISVPH:  1 TO WRITE FILE FOR VELOCITY PLOTTING IN HORIZONTAL PLANE");
	fprintf(f73, "%s\n", "* 2 WRITE ONLY DURING LAST REFERENCE TIME PERIOD");
	fprintf(f73, "%s\n", "*  NPVPH : NUMBER OF WRITES PER REFERENCE TIME PERIOD");
	fprintf(f73, "%s\n", "*  ISRVPH : 1 TO WRITE FILE FOR RESIDUAL VELOCITY PLOTTIN IN");
	fprintf(f73, "%s\n", "*            HORIZONTAL PLANE");
	fprintf(f73, "%s\n", "*  IVPHXY : 0 DOES NOT WRITE I, J, X, Y IN velplth.out and rvelplth.out FILES");
	fprintf(f73, "%s\n", "* 1 WRITES I, J ONLY IN velplth.out and rvelplth.out FILES");
	fprintf(f73, "%s\n", "* 2 WRITES I, J, X, Y  IN velplth.out and rvelplth.out FILES");
	fprintf(f73, "%s\n", "* 3 WRITES EFDC EXPLORER BINARY FORMAT FILES");
	fprintf(f73, "%s\n", "*");
	fprintf(f73, "%s\n", "C73 ISVPH NPVPH ISRVPH IVPHXY");
    fprintf(f73, "     %d     %d     %d      %d\n", \
		ISVPH, NPVPH, ISRVPH, IVPHXY);

	fprintf(f73, "%s\n", "-------------------------------------------------------------------------------");
	fprintf(f73, "%s ", "C74");
	fwrite(copia, tamcp, 1, f73);
	free(copia);
	for (pos = ftell(f73); pos < tam; pos++){
		fputc(' ', f73);
	}
	fclose(f73);

	// ====================================================
	//                    C84
	// ====================================================
	FILE *f84 = fopen(filepath, "rb+");
	fseek(f84, 0, SEEK_END);
	tam = ftell(f84) - 4;
	rewind(f84);
	pos_C85 = ftell(f84);
	while (fscanf(f84, "%s%*c\n", s84) == 1 && strcmp(s84, "C85") != 0){
		pos_C85 = ftell(f84);
	}
	tamcp = tam - pos_C85;
	copia = (char*)malloc(tamcp*sizeof(char));
	fread(copia, tamcp, 1, f84);
	rewind(f84);
	pos_C84 = ftell(f84);
	while (fscanf(f84, "%s%*c", s84) == 1 && strcmp(s84, "C84") != 0){
		pos_C84 = ftell(f84);
	}
	fseek(f84, pos_C84, SEEK_SET);
	fprintf(f84, "%s\n", "");
	fprintf(f84, "%s\n", "C84 CONTROLS FOR WRITING TO TIME SERIES FILES");
	fprintf(f84, "%s\n", "*");
	fprintf(f84, "%s\n", "*  ISTMSR:  1 OR 2 TO WRITE TIME SERIES OF SURF ELEV, VELOCITY, NET");
	fprintf(f84, "%s\n", "*           INTERNAL AND EXTERNAL MODE VOLUME SOURCE - SINKS, AND");
	fprintf(f84, "%s\n", "*           CONCENTRATION VARIABLES, 2 APPENDS EXISTING TIME SERIES FILES");
	fprintf(f84, "%s\n", "*  MLTMSR : NUMBER HORIZONTAL LOCATIONS TO WRITE TIME SERIES OF SURF ELEV,");
	fprintf(f84, "%s\n", "*VELOCITY, AND CONCENTRATION VARIABLES");
	fprintf(f84, "%s\n", "*  NBTMSR : TIME STEP TO BEGIN WRITING TO TIME SERIES FILES(Inactive)");
	fprintf(f84, "%s\n", "*  NSTMSR : TIME STEP TO STOP WRITING TO TIME SERIES FILES(Inactive)");
	fprintf(f84, "%s\n", "*  NWTMSR : NUMBER OF TIME STEPS TO SKIP BETWEEN OUTPUT");
	fprintf(f84, "%s\n", "*  NTSSTSP : NUMBER OF TIME SERIES START - STOP SCENARIOS, 1 OR GREATER");
	fprintf(f84, "%s\n", "*  TCTMSR : UNIT CONVERSION FOR TIME SERIES TIME.FOR SECONDS, MINUTES,");
	fprintf(f84, "%s\n", "*HOURS, DAYS USE 1.0, 60.0, 3600.0, 86400.0 RESPECTIVELY");
	fprintf(f84, "%s\n", "*");
	fprintf(f84, "%s\n", "*");
	fprintf(f84, "%s\n", "C84 ISTMSR MLTMSR  NBTMSR  NSTMSR  NWTMSR NTSSTSP TCTMSR");
	fprintf(f84, "     %d    %d     %d    %d    %d    %d     %lf\n", \
	       ISTMSR, MLTMSR,  NBTMSR,  NSTMSR,  NWTMSR, NTSSTSP, TCTMSR);

	fprintf(f84, "%s\n", "-------------------------------------------------------------------------------");
	fprintf(f84, "%s ", "C85");
	fwrite(copia, tamcp, 1, f84);
	free(copia);
	for (pos = ftell(f84); pos < tam; pos++){
		fputc(' ', f84);
	}
	fclose(f84);

	// ====================================================
	//                    C87
	// ====================================================
	FILE *f87 = fopen(filepath, "rb+");
	fseek(f87, 0, SEEK_END);
	tam = ftell(f87) - 4;
	rewind(f87);
	pos_C88 = ftell(f87);
	while (fscanf(f87, "%s%*c\n", s87) == 1 && strcmp(s87, "C88") != 0){
		pos_C88 = ftell(f87);
	}
	tamcp = tam - pos_C88;
	copia = (char*)malloc(tamcp*sizeof(char));
	fread(copia, tamcp, 1, f87);
	rewind(f87);
	pos_C87 = ftell(f87);
	while (fscanf(f87, "%s%*c", s87) == 1 && strcmp(s87, "C87") != 0){
		pos_C87 = ftell(f87);
	}
	fseek(f87, pos_C87, SEEK_SET);
	fprintf(f87, "%s\n", "");
	fprintf(f87, "%s\n", "C87 CONTROLS FOR WRITING TO TIME SERIES FILES");
	fprintf(f87, "%s\n", "*");
	fprintf(f87, "%s\n", "*  ILTS:    I CELL INDEX");
	fprintf(f87, "%s\n", "*  JLTS : J CELL INDEX");
	fprintf(f87, "%s\n", "*  NTSSSS : WRITE SCENARIO FOR THIS LOCATION");
	fprintf(f87, "%s\n", "*  MTSP : 1 FOR TIME SERIES OF SURFACE ELEVATION");
	fprintf(f87, "%s\n", "*  MTSC : 1 FOR TIME SERIES OF TRANSPORTED CONCENTRATION VARIABLES");
	fprintf(f87, "%s\n", "*  MTSA : 1 FOR TIME SERIES OF EDDY VISCOSITY AND DIFFUSIVITY");
	fprintf(f87, "%s\n", "*  MTSUE : 1 FOR TIME SERIES OF EXTERNAL MODE HORIZONTAL VELOCITY");
	fprintf(f87, "%s\n", "*  MTSUT : 1 FOR TIME SERIES OF EXTERNAL MODE HORIZONTAL TRANSPORT");
	fprintf(f87, "%s\n", "*  MTSU : 1 FOR TIME SERIES OF HORIZONTAL VELOCITY IN EVERY LAYER");
	fprintf(f87, "%s\n", "*  MTSQE : 1 FOR TIME SERIES OF NET EXTERNAL MODE VOLUME SOURCE / SINK");
	fprintf(f87, "%s\n", "*  MTSQ : 1 FOR TIME SERIES OF NET EXTERNAL MODE VOLUME SOURCE / SINK");
	fprintf(f87, "%s\n", "*  CLTS : LOCATION AS A CHARACTER VARIALBLE");
	fprintf(f87, "%s\n", "*");
	fprintf(f87, "%s\n", "C87 ILTS JLTS NTSSSS MTSP MTSC MTSA MTSUE MTSUT MTSU MTSQE MTSQ CLTS");

	for (i = 0; i < nts_out; i++){
		fprintf(f87, "     %d    %d    %d    %d    %d    %d    %d    %d    %d    %d    %d  %s%d%s\n", \
			LTS[i][0], LTS[i][1], NTSSSS, MTSP, MTSC, MTSA, MTSUE, MTSUT, MTSU,0,0, "'P_",i,"'");
	}
	fprintf(f87, "%s\n", "-------------------------------------------------------------------------------");
	fprintf(f87, "%s ", "C88");
	fwrite(copia, tamcp, 1, f87);
	free(copia);
	for (pos = ftell(f87); pos < tam; pos++){
		//fputc(' ', f87);
		fprintf(f87, "%s ", "");
	}
	fclose(f87);
    
}

__host__ void write_efdc_SSER(const char *filepath, int type, int nlayers, int **north_bc, int **south_bc, int **east_bc, int **west_bc,double **north_val, double SALT, double efdc_basetime, int tq){

	FILE * outSSER = fopen(filepath, "w");

	//                      SSER.INP: LAYOUT  
	
	fprintf(outSSER, "%s\n", "C **, SSER.INP Time Series FILE, DDD  15 / 04 / 2021 15:28");
	fprintf(outSSER, "%s\n", "C **");
	fprintf(outSSER, "%s\n", "C **");
	fprintf(outSSER, "%s\n", "C **  InType MSSER(NS) TCSSER(NS) TASSER(NS) RMULADJ(NS) ADDADJ(NS)");
	fprintf(outSSER, "%s\n", "C **");
	fprintf(outSSER, "%s\n", "C **  IF InType.EQ.1 THEN READ DEPTH WEIGHTS AND SINGLE VALUE OF SSER");
	fprintf(outSSER, "%s\n", "C **                      ELSE READ A VALUE OF SSER FOR EACH LAYER");
	fprintf(outSSER, "%s\n", "C **");
	fprintf(outSSER, "%s\n", "C **  InType = 1 Structure");
	fprintf(outSSER, "%s\n", "C **  WKQ(K), K = 1, KC");
	fprintf(outSSER, "%s\n", "C **  TSSER(M, NS)    SSER(M, 1, NS) !(MSSER(NS) PAIRS FOR NS = 1, NSSER SERIES)");
	fprintf(outSSER, "%s\n", "C **");
	fprintf(outSSER, "%s\n", "C **  InType = 0 Structure");
	fprintf(outSSER, "%s\n", "C **  TSSER(M, NS)    (SSER(M, K, NS), K = 1, KC) !(MSSER(NS) PAIRS)");
	fprintf(outSSER, "%s\n", "C **");

	int ig1,il;
	int edg = 0;	
	// North
	int bn = 0;
	while (north_bc[bn][0] != 0){
		bn = bn + 1;
		edg = edg + 1;
		fprintf(outSSER, "      %d %d %lf %d %d %d %s%d\n",type, tq + 1, efdc_basetime, 0, 1, 0," ' *** North_edge_", edg);
		for (ig1 = 0; ig1 < tq + 1; ig1++){
			fprintf(outSSER, "      %lf ", (north_val[ig1][0]));
			for (il = 0; il < nlayers;il++){

				if (il==nlayers-1){
					fprintf(outSSER, "%lf\n", SALT);
				}
				else{
					fprintf(outSSER, "%lf ", SALT);
				}
			
			}
		}

	}

	// South
	int bs = 0;
	while (south_bc[bs][0] != 0){
		bs = bs + 1;
		edg = edg + 1;
		fprintf(outSSER, "      %d %d %lf %d %d %d %s%d\n", type, tq + 1, efdc_basetime, 0, 1, 0, " ' *** South_edge_", edg);
		for (ig1 = 0; ig1 < tq + 1; ig1++){
			fprintf(outSSER, "      %lf ", (north_val[ig1][0]));
			for (il = 0; il < nlayers; il++){
				if (il == nlayers - 1){
					fprintf(outSSER, "%lf\n", SALT);
				}
				else{
					fprintf(outSSER, "%lf ", SALT);
				}
			}
		}

	}

	// East
	int be = 0;
	while (east_bc[be][0] != 0){
		be = be + 1;
		edg = edg + 1;
		fprintf(outSSER, "      %d %d %lf %d %d %d %s%d\n", type, tq + 1, efdc_basetime, 0, 1, 0, " ' *** East_edge_", edg);
		for (ig1 = 0; ig1 < tq + 1; ig1++){
			fprintf(outSSER, "      %lf ", (north_val[ig1][0]));
			for (il = 0; il < nlayers; il++){
								
				if (il == nlayers - 1){
					fprintf(outSSER, "%lf\n", SALT);
				}
				else{
					fprintf(outSSER, "%lf ", SALT);
				}
								
			}
		}

	}

	// West
	int bw = 0;
	while (west_bc[bw][0] != 0){
		bw = bw + 1;
		edg = edg + 1;
		fprintf(outSSER, "      %d %d %lf %d %d %d %s%d\n", type, tq + 1, efdc_basetime, 0, 1, 0, " ' *** West_edge_", edg);
		for (ig1 = 0; ig1 < tq + 1; ig1++){
			fprintf(outSSER, "      %lf ", (north_val[ig1][0]));
			for (il = 0; il < nlayers; il++){
				
					
				if (il == nlayers - 1){
					fprintf(outSSER, "%lf\n", SALT);
				}
				else{
					fprintf(outSSER, "%lf ", SALT);
				}
							
				
			}
		}

	}
	fclose(outSSER);
}


__host__ void write_efdc_TSER(const char *filepath, int type, int nlayers, int **north_bc, int **south_bc, int **east_bc, int **west_bc,double **north_val, double *TEMP, double efdc_basetime, int tq){

	FILE * outTSER = fopen(filepath, "w");

	//                      TSER.INP: LAYOUT  
	
	fprintf(outTSER, "%s\n", "C **, TSER.INP Time Series FILE, DDD  15 / 04 / 2021 15:28");
	fprintf(outTSER, "%s\n", "C **");
	fprintf(outTSER, "%s\n", "C **");
	fprintf(outTSER, "%s\n", "C **  InType MTSER(NS) TCTSER(NS) TATSER(NS) RMULADJ(NS) ADDADJ(NS)");
	fprintf(outTSER, "%s\n", "C **");
	fprintf(outTSER, "%s\n", "C **  IF InType.EQ.1 THEN READ DEPTH WEIGHTS AND SINGLE VALUE OF TSER");
	fprintf(outTSER, "%s\n", "C **                      ELSE READ A VALUE OF TSER FOR EACH LAYER");
	fprintf(outTSER, "%s\n", "C **");
	fprintf(outTSER, "%s\n", "C **  InType = 1 Structure");
	fprintf(outTSER, "%s\n", "C **  WKQ(K), K = 1, KC");
	fprintf(outTSER, "%s\n", "C **  TTSER(M, NS)    TSER(M, 1, NS) !(MTSER(NS) PAIRS FOR NS = 1, NTSER SERIES)");
	fprintf(outTSER, "%s\n", "C **");
	fprintf(outTSER, "%s\n", "C **  InType = 0 Structure");
	fprintf(outTSER, "%s\n", "C **  TTSER(M, NS)    (TSER(M, K, NS), K = 1, KC) !(MTSER(NS) PAIRS)");
	fprintf(outTSER, "%s\n", "C **");

	int ig1,il;
	int edg = 0;
	// North
	int bn = 0;
	while (north_bc[bn][0] != 0){
		bn = bn + 1;
		edg = edg + 1;
		fprintf(outTSER, "      %d %d %lf %d %d %d %s%d\n",type, tq + 1, efdc_basetime, 0, 1, 0," ' *** North_edge_", edg);
		for (ig1 = 0; ig1 < tq + 1; ig1++){
			fprintf(outTSER, "      %lf ", (north_val[ig1][0]));
			for (il = 0; il < nlayers;il++){

				if (il==nlayers-1){
					fprintf(outTSER, "%lf\n", TEMP[ig1]);
				}
				else{
					fprintf(outTSER, "%lf ", TEMP[ig1]);
				}
			
			}
		}

	}

	// South
	int bs = 0;
	while (south_bc[bs][0] != 0){
		bs = bs + 1;
		edg = edg + 1;
		fprintf(outTSER, "      %d %d %lf %d %d %d %s%d\n", type, tq + 1, efdc_basetime, 0, 1, 0, " ' *** South_edge_", edg);
		for (ig1 = 0; ig1 < tq + 1; ig1++){
			fprintf(outTSER, "      %lf ", (north_val[ig1][0]));
			for (il = 0; il < nlayers; il++){
				if (il == nlayers - 1){
					fprintf(outTSER, "%lf\n", TEMP[ig1]);
				}
				else{
					fprintf(outTSER, "%lf ", TEMP[ig1]);
				}
			}
		}

	}

	// East
	int be = 0;
	while (east_bc[be][0] != 0){
		be = be + 1;
		edg = edg + 1;
		fprintf(outTSER, "      %d %d %lf %d %d %d %s%d\n", type, tq + 1, efdc_basetime, 0, 1, 0, " ' *** East_edge_", edg);
		for (ig1 = 0; ig1 < tq + 1; ig1++){
			fprintf(outTSER, "      %lf ", (north_val[ig1][0]));
			for (il = 0; il < nlayers; il++){
								
				if (il == nlayers - 1){
					fprintf(outTSER, "%lf\n", TEMP[ig1]);
				}
				else{
					fprintf(outTSER, "%lf ", TEMP[ig1]);
				}
								
			}
		}

	}

	// West
	int bw = 0;
	while (west_bc[bw][0] != 0){
		bw = bw + 1;
		edg = edg + 1;
		fprintf(outTSER, "      %d %d %lf %d %d %d %s%d\n", type, tq + 1, efdc_basetime, 0, 1, 0, " ' *** West_edge_", edg);
		for (ig1 = 0; ig1 < tq + 1; ig1++){
			fprintf(outTSER, "      %lf ", (north_val[ig1][0]));
			for (il = 0; il < nlayers; il++){
				
					
				if (il == nlayers - 1){
					fprintf(outTSER, "%lf\n", TEMP[ig1]);
				}
				else{
					fprintf(outTSER, "%lf ", TEMP[ig1]);
				}
							
				
			}
		}

	}
	fclose(outTSER);
}

__host__ void write_DYE_INP(const char *filepath, int cols, int rows, int nlayers,int max_water_cells, int *mask, double DYE_INIT){
	
	int watercell,j,i,l,posxy;
    watercell = 0;
	
	int width_number = round(1+log(max_water_cells)/log(10))-1;

	FILE * outDYE_INP = fopen(filepath, "w");	
	
	fprintf(outDYE_INP, "%s\n", "C");
	fprintf(outDYE_INP, "%s\n", "C");
	fprintf(outDYE_INP, "%s\n", "C");
	fprintf(outDYE_INP, "%s\n", "C");
	fprintf(outDYE_INP, "%s\n", "1");
	
	for (j=0;j<rows;j++){	
	
		posxy = rows*cols - (j + 1)*cols;
		posxy = posxy - 1;		
		for(i=0;i<cols;i++){						
			posxy=posxy+1;			
			if(mask[posxy] == 1){				
				watercell = watercell+1;				
				fprintf(outDYE_INP, "%*d     %d     %d     ",width_number, watercell, i+1, j+1);	
				
				for(l=0;l<nlayers;l++){					
					if(l<nlayers-1){
						fprintf(outDYE_INP, "%lf    ", DYE_INIT);
					}
					else{
						fprintf(outDYE_INP, "%lf\n", DYE_INIT);						
					}					
				}						
			}				
		}	
		
	}	
	fclose(outDYE_INP);		
}

__host__ void write_SALT_INP(const char *filepath, int cols, int rows, int nlayers,int max_water_cells, int *mask, double SALT_INIT){
	
	int watercell,j,i,l,posxy;
    watercell = 0;
	
	int width_number = round(1+log(max_water_cells)/log(10))-1;

	FILE * outSALT_INP = fopen(filepath, "w");	
	
	fprintf(outSALT_INP, "%s\n", "C");
	fprintf(outSALT_INP, "%s\n", "C");
	fprintf(outSALT_INP, "%s\n", "C");
	fprintf(outSALT_INP, "%s\n", "C");
	fprintf(outSALT_INP, "%s\n", "1");
	
	for (j=0;j<rows;j++){	
	
		posxy = rows*cols - (j + 1)*cols;
		posxy = posxy - 1;		
		for(i=0;i<cols;i++){						
			posxy=posxy+1;			
			if(mask[posxy] == 1){				
				watercell = watercell+1;				
				fprintf(outSALT_INP, "%*d     %d     %d     ",width_number, watercell, i+1, j+1);	
				
				for(l=0;l<nlayers;l++){					
					if(l<nlayers-1){
						fprintf(outSALT_INP, "%lf    ", SALT_INIT);
					}
					else{
						fprintf(outSALT_INP, "%lf\n", SALT_INIT);						
					}					
				}						
			}				
		}	
		
	}	
	fclose(outSALT_INP);		
}

__host__ void write_TEMP_INP(const char *filepath, int cols, int rows, int nlayers,int max_water_cells, int *mask, double *initial_depth, double WTEMP_MIN, double MAIR_TEMP){
	
	int watercell,j,i,l,posxy;
	double L_DEP,TEMP_INIT;
	
    watercell = 0;
	
	int width_number = round(1+log(max_water_cells)/log(10))-1;

	FILE * outTEMP_INP = fopen(filepath, "w");	
	
	fprintf(outTEMP_INP, "%s\n", "C");
	fprintf(outTEMP_INP, "%s\n", "C");
	fprintf(outTEMP_INP, "%s\n", "C");
	fprintf(outTEMP_INP, "%s\n", "C");
	fprintf(outTEMP_INP, "%s\n", "1");
	
	for (j=0;j<rows;j++){	
	
		posxy = rows*cols - (j + 1)*cols;
		posxy = posxy - 1;		
		for(i=0;i<cols;i++){						
			posxy=posxy+1;			
			if(mask[posxy] == 1){				
				watercell = watercell+1;				
				fprintf(outTEMP_INP, "%*d     %d     %d     ",width_number, watercell, i+1, j+1);				
				for(l=0;l<nlayers;l++){		
				    
					// Vertical disdtribution of the Temperarures
					
                    L_DEP =	(initial_depth[posxy]/nlayers)*	((nlayers-1)-l);				    
					TEMP_INIT = WTEMP_MIN + (MAIR_TEMP-WTEMP_MIN)/(1.00+L_DEP);
					
					if(l<nlayers-1){
						fprintf(outTEMP_INP, "%lf    ", TEMP_INIT);
					}
					else{
						fprintf(outTEMP_INP, "%lf\n", TEMP_INIT);						
					}					
				}						
			}				
		}	
		
	}	
	fclose(outTEMP_INP);		
}


__host__ void write_efdc_DSER(const char *filepath, int type, int nlayers, int **north_bc, int **south_bc, int **east_bc, int **west_bc,double **north_val, double DYE_CONC, double efdc_basetime, int tq){

	FILE * outDSER = fopen(filepath, "w");

	//                      DSER.INP: LAYOUT  
	
	fprintf(outDSER, "%s\n", "C **, DSER.INP Time Series FILE, DDD  15 / 04 / 2021 15:28");
	fprintf(outDSER, "%s\n", "C **");
	fprintf(outDSER, "%s\n", "C **");
	fprintf(outDSER, "%s\n", "C **  InType MDSER(NS) TCDSER(NS) TADSER(NS) RMULADJ(NS) ADDADJ(NS)");
	fprintf(outDSER, "%s\n", "C **");
	fprintf(outDSER, "%s\n", "C **  IF InType.EQ.1 THEN READ DEPTH WEIGHTS AND SINGLE VALUE OF DSER");
	fprintf(outDSER, "%s\n", "C **                      ELSE READ A VALUE OF DSER FOR EACH LAYER");
	fprintf(outDSER, "%s\n", "C **");
	fprintf(outDSER, "%s\n", "C **  InType = 1 Structure");
	fprintf(outDSER, "%s\n", "C **  WKQ(K), K = 1, KC");
	fprintf(outDSER, "%s\n", "C **  TDSER(M, NS)    DSER(M, 1, NS) !(MDSER(NS) PAIRS FOR NS = 1, NDSER SERIES)");
	fprintf(outDSER, "%s\n", "C **");
	fprintf(outDSER, "%s\n", "C **  InType = 0 Structure");
	fprintf(outDSER, "%s\n", "C **  TDSER(M, NS)    (DSER(M, K, NS), K = 1, KC) !(MDSER(NS) PAIRS)");
	fprintf(outDSER, "%s\n", "C **");

	int ig1,il;
	int edg = 0;
	// North
	int bn = 0;
	while (north_bc[bn][0] != 0){
		bn = bn + 1;
		edg = edg + 1;
		fprintf(outDSER, "      %d %d %lf %d %d %d %s%d\n",type, tq + 1, efdc_basetime, 0, 1, 0," ' *** North_edge_", edg);
		for (ig1 = 0; ig1 < tq + 1; ig1++){
			fprintf(outDSER, "      %lf ", (north_val[ig1][0]));
			for (il = 0; il < nlayers;il++){

				if (il==nlayers-1){
					fprintf(outDSER, "%lf\n", DYE_CONC);
				}
				else{
					fprintf(outDSER, "%lf ", DYE_CONC);
				}
			
			}
		}

	}

	// South
	int bs = 0;
	while (south_bc[bs][0] != 0){
		bs = bs + 1;
		edg = edg + 1;
		fprintf(outDSER, "      %d %d %lf %d %d %d %s%d\n", type, tq + 1, efdc_basetime, 0, 1, 0, " ' *** South_edge_", edg);
		for (ig1 = 0; ig1 < tq + 1; ig1++){
			fprintf(outDSER, "      %lf ", (north_val[ig1][0]));
			for (il = 0; il < nlayers; il++){
				if (il == nlayers - 1){
					fprintf(outDSER, "%lf\n", DYE_CONC);
				}
				else{
					fprintf(outDSER, "%lf ", DYE_CONC);
				}
			}
		}

	}

	// East
	int be = 0;
	while (east_bc[be][0] != 0){
		be = be + 1;
		edg = edg + 1;
		fprintf(outDSER, "      %d %d %lf %d %d %d %s%d\n", type, tq + 1, efdc_basetime, 0, 1, 0, " ' *** East_edge_", edg);
		for (ig1 = 0; ig1 < tq + 1; ig1++){
			fprintf(outDSER, "      %lf ", (north_val[ig1][0]));
			for (il = 0; il < nlayers; il++){
								
				if (il == nlayers - 1){
					fprintf(outDSER, "%lf\n", DYE_CONC);
				}
				else{
					fprintf(outDSER, "%lf ", DYE_CONC);
				}
								
			}
		}

	}

	// West
	int bw = 0;
	while (west_bc[bw][0] != 0){
		bw = bw + 1;
		edg = edg + 1;
		fprintf(outDSER, "      %d %d %lf %d %d %d %s%d\n", type, tq + 1, efdc_basetime, 0, 1, 0, " ' *** West_edge_", edg);
		for (ig1 = 0; ig1 < tq + 1; ig1++){
			fprintf(outDSER, "      %lf ", (north_val[ig1][0]));
			for (il = 0; il < nlayers; il++){
				
					
				if (il == nlayers - 1){
					fprintf(outDSER, "%lf\n", DYE_CONC);
				}
				else{
					fprintf(outDSER, "%lf ", DYE_CONC);
				}
							
				
			}
		}

	}
	fclose(outDSER);
}



__host__ void write_efdc_QSER(const char *filepath, int type, int nlayers, int **north_bc, int **south_bc, int **east_bc, int **west_bc, double **north_val, double **south_val, double **east_val, double **west_val, double efdc_basetime, int tq){

	FILE * outQSER = fopen(filepath, "w");

	//                      QSER.INP: LAYOUT  
	
	fprintf(outQSER, "%s\n", "C **, qser.inp Time Series FILE, DDD  15 / 04 / 2021 15:28");
	fprintf(outQSER, "%s\n", "C **");
	fprintf(outQSER, "%s\n", "C **  InType MQSER(NS) TCQSER(NS) TAQSER(NS) RMULADJ(NS) ADDADJ(NS) ICHGQS");
	fprintf(outQSER, "%s\n", "C **");
	fprintf(outQSER, "%s\n", "C **  IF InType.EQ.1 THEN READ DEPTH WEIGHTS AND SINGLE VALUE OF QSER");
	fprintf(outQSER, "%s\n", "C **                      ELSE READ A VALUE OF QSER FOR EACH LAYER");
	fprintf(outQSER, "%s\n", "C **");
	fprintf(outQSER, "%s\n", "C **  InType = 1 Structure");
	fprintf(outQSER, "%s\n", "C **  WKQ(K), K = 1, KC");
	fprintf(outQSER, "%s\n", "C **  TQSER(M, NS)    QSER(M, 1, NS)          !(MQSER(NS) PAIRS FOR NS = 1, NQSER SERIES)");
	fprintf(outQSER, "%s\n", "C **");
	fprintf(outQSER, "%s\n", "C **  InType = 0 Structure");
	fprintf(outQSER, "%s\n", "C **  TQSER(M, NS)    (QSER(M, K, NS), K = 1, KC) !(MQSER(NS) PAIRS)");
	fprintf(outQSER, "%s\n", "C **");

	int ig1,il;
	int edg = 0;
	// North
	int bn = 0;
	while (north_bc[bn][0] != 0){
		bn = bn + 1;
		edg = edg + 1;
		fprintf(outQSER, "      %d %d %lf %d %d %d %d %s%d\n",type, tq + 1, efdc_basetime, 0, 1, 0, 0," ' *** North_edge_", edg);
		for (ig1 = 0; ig1 < tq + 1; ig1++){
			fprintf(outQSER, "      %lf ", (north_val[ig1][0]));
			for (il = 0; il < nlayers;il++){

				if (il==nlayers-1){
					fprintf(outQSER, "%lf\n", north_val[ig1][bn] / nlayers);
				}
				else{
					fprintf(outQSER, "%lf ", north_val[ig1][bn] / nlayers);
				}
			
			}
		}

	}

	// South
	int bs = 0;
	while (south_bc[bs][0] != 0){
		bs = bs + 1;
		edg = edg + 1;
		fprintf(outQSER, "      %d %d %lf %d %d %d %d %s%d\n", type, tq + 1, efdc_basetime, 0, 1, 0, 0, " ' *** South_edge_", edg);
		for (ig1 = 0; ig1 < tq + 1; ig1++){
			fprintf(outQSER, "      %lf ", (south_val[ig1][0]));
			for (il = 0; il < nlayers; il++){
				if (il == nlayers - 1){
					fprintf(outQSER, "%lf\n", south_val[ig1][bs] / nlayers);
				}
				else{
					fprintf(outQSER, "%lf ", south_val[ig1][bs] / nlayers);
				}
			}
		}

	}

	// East
	int be = 0;
	while (east_bc[be][0] != 0){
		be = be + 1;
		edg = edg + 1;
		fprintf(outQSER, "      %d %d %lf %d %d %d %d %s%d\n", type, tq + 1, efdc_basetime, 0, 1, 0, 0, " ' *** East_edge_", edg);
		for (ig1 = 0; ig1 < tq + 1; ig1++){
			fprintf(outQSER, "      %lf ", (east_val[ig1][0]));
			for (il = 0; il < nlayers; il++){
								
				if (il == nlayers - 1){
					fprintf(outQSER, "%lf\n", east_val[ig1][be] /nlayers);
				}
				else{
					fprintf(outQSER, "%lf ", east_val[ig1][be] /nlayers);
				}
								
			}
		}

	}

	// West
	int bw = 0;
	while (west_bc[bw][0] != 0){
		bw = bw + 1;
		edg = edg + 1;
		fprintf(outQSER, "      %d %d %lf %d %d %d %d %s%d\n", type, tq + 1, efdc_basetime, 0, 1, 0, 0, " ' *** West_edge_", edg);
		for (ig1 = 0; ig1 < tq + 1; ig1++){
			fprintf(outQSER, "      %lf ", (west_val[ig1][0]));
			for (il = 0; il < nlayers; il++){
				
					
				if (il == nlayers - 1){
					fprintf(outQSER, "%lf\n", west_val[ig1][bw] /nlayers);
				}
				else{
					fprintf(outQSER, "%lf ", west_val[ig1][bw] / nlayers);
				}
							
				
			}
		}

	}
	fclose(outQSER);
}

__host__ void write_efdc_WSER(const char *filepath, double **wind_vel, double data_basetime, double efdc_basetime, int ntime){

	// ntime = number of time data points
	// data_time in seconds
	// efdc_basetime in seconds

	FILE * outWSER = fopen(filepath, "w");

	//                      WSER.INP: LAYOUT  

	fprintf(outWSER, "%s\n", "C **, wser.inp Time Series FILE, DDD 11 / 12 / 2008 00:31 : 51");
	fprintf(outWSER, "%s\n", "C **");
	fprintf(outWSER, "%s\n", "C **  WIND FORCING FILE, USE WITH 7 APRIL 97 AND LATER VERSIONS OF EFDC");
	fprintf(outWSER, "%s\n", "C **");
	fprintf(outWSER, "%s\n", "C **  MASER(NW) = NUMBER OF TIME DATA POINTS");
	fprintf(outWSER, "%s\n", "C **  TCASER(NW) = DATA TIME UNIT CONVERSION TO SECONDS");
	fprintf(outWSER, "%s\n", "C **  TAASER(NW) = ADDITIVE ADJUSTMENT OF TIME VALUES SAME UNITS AS INPUT TIMES");
	fprintf(outWSER, "%s\n", "C **  WINDSCT(NW) = WIND SPEED CONVERSION TO M / SEC");
	fprintf(outWSER, "%s\n", "C **  ISWDINT(NW) = DIRECTION CONVENTION");
	fprintf(outWSER, "%s\n", "C **               0 DIRECTION TO");
	fprintf(outWSER, "%s\n", "C **               1 DIRECTION FROM");
	fprintf(outWSER, "%s\n", "C **               2 WINDS IS EAST VELOCITY, WINDD IS NORTH VELOCITY");
	fprintf(outWSER, "%s\n", "#  EFDC_DS_WIND_HEIGHT: 10.0");
	fprintf(outWSER, "%s\n", "#  EFDC_DS_ICE_PERIOD_START : 0.");
	fprintf(outWSER, "%s\n", "#  EFDC_DS_ICE_PERIOD_END : 0.");
	fprintf(outWSER, "%s\n", "C **  MASER  TCASER   TAASER  WINDSCT  ISWDINT");
	fprintf(outWSER, "%s\n", "C **  TASER(M) WINDS(M) WINDD(M)");

	// if ISWDINT = 0 or 1, then WINDS(M) = velocity and WINDD(M) = ang

	int ig1, il, wn;	
	double stime = -1.000;
	fprintf(outWSER, "      %d %lf %lf %lf %d\n", ntime, efdc_basetime, 0.0, 1.0, 1);
	for (ig1 = 0; ig1 < ntime; ig1++){
		wn = -1;
		stime = stime + 1.000;
		fprintf(outWSER, "      %lf ", (stime*data_basetime)/efdc_basetime);
		for (il = 0; il < 2; il++){
			wn = wn + 1;
			if (il == 1){
				fprintf(outWSER, "%lf\n", wind_vel[ig1][wn]);
			}
			else{
				fprintf(outWSER, "%lf ", wind_vel[ig1][wn]);
			}
		}

	}
	
	fclose(outWSER);
}

__host__ void write_efdc_ASER(const char *filepath, double **ASER_val, double data_basetime, double efdc_basetime, int ntime){

	// ntime = number of time data points
	// data_time in seconds
	// efdc_basetime in seconds

	FILE * outASER = fopen(filepath, "w");

	//                      ASER.INP: LAYOUT  
	fprintf(outASER, "%s\n", "# ASER.INP: 1 / 1 / 2007 to 11 / 27 / 2009");
	fprintf(outASER, "%s\n", "# WBAN : 22521 - Honolulu Inter Airport");
	fprintf(outASER, "%s\n", "#	First time point = 733042 HST(derived from the fact that days 0.5 have max temp)");
	fprintf(outASER, "%s\n", "#  ATMOSPHERIC FORCING FILE, USE WITH 28 JULY 96 AND LATER VERSIONS OF EFDC");
	fprintf(outASER, "%s\n", "#  MASER = NUMBER OF TIME DATA POINTS");
	fprintf(outASER, "%s\n", "#  TCASER = DATA TIME UNIT CONVERSION TO SECONDS");
	fprintf(outASER, "%s\n", "#  TAASER = ADDITIVE ADJUSTMENT OF TIME VALUES SAME UNITS AS INPUT TIMES");
	fprintf(outASER, "%s\n", "#  IRELH = 0 VALUE TWET COLUMN VALUE IS TWET, = 1 VALUE IS RELATIVE HUMIDITY");
	fprintf(outASER, "%s\n", "#  RAINCVT = CONVERTS RAIN TO UNITS OF M / SEC.in / hr * 0.0254 m / 1 in * 1 hr / 3600 s = 7.05556E-06 (m / s)");
	fprintf(outASER, "%s\n", "#  EVAPCVT = CONVERTS EVAP TO UNITS OF M / SEC, IF EVAPCVT<0 EVAP IS INTERNALLY COMPUTED");
	fprintf(outASER, "%s\n", "#  SOLRCVT = CONVERTS SOLAR SW RADIATION TO JOULES / SQ METER");
	fprintf(outASER, "%s\n", "#  CLDCVT = MULTIPLIER FOR ADJUSTING CLOUD COVER");
	fprintf(outASER, "%s\n", "#  IASWRAD = O DISTRIBUTE SW SOL RAD OVER WATER COL AND INTO BED, = 1 ALL TO SURF LAYER");
	fprintf(outASER, "%s\n", "#  REVC = 1000 * EVAPORATIVE TRANSFER COEF, REVC<0 USE WIND SPD DEPD DRAG COEF");
	fprintf(outASER, "%s\n", "#  RCHC = 1000 * CONVECTIVE HEAT TRANSFER COEF, REVC<0 USE WIND SPD DEPD DRAG COEF");
	fprintf(outASER, "%s\n", "#  SWRATNF = FAST SCALE SOLAR SW RADIATION ATTENUATION COEFFCIENT 1. / METERS");
	fprintf(outASER, "%s\n", "#  SWRATNS = SLOW SCALE SOLAR SW RADIATION ATTENUATION COEFFCIENT 1. / METERS");
	fprintf(outASER, "%s\n", "#  FSWRATF = FRACTION OF SOLSR SW RADIATION ATTENUATED FAST  0<FSWRATF<1");
	fprintf(outASER, "%s\n", "#  DABEDT = DEPTH OR THICKNESS OF ACTIVE BED TEMPERATURE LAYER, METERS");
	fprintf(outASER, "%s\n", "#  TBEDIT = INITIAL BED TEMPERATURE");
	fprintf(outASER, "%s\n", "#  HTBED1 = CONVECTIVE HT COEFFCIENT BETWEEN BED AND BOTTOM WATER LAYER  NO DIM");
	fprintf(outASER, "%s\n", "#  HTBED2 = HEAT TRANS COEFFCIENT BETWEEN BED AND BOTTOM WATER LAYER  M / SEC");
	fprintf(outASER, "%s\n", "#  PATM = ATM PRESS MILLIBAR");
	fprintf(outASER, "%s\n", "#  TDRY / TEQ = DRY ATM TEMP ISOPT(2) = 1 OR EQUIL TEMP ISOPT(2) = 2");
	fprintf(outASER, "%s\n", "#  TWET / RELH = WET BULB ATM TEMP IRELH = 0, RELATIVE HUMIDITY IRELH = 1");
	fprintf(outASER, "%s\n", "#  RAIN = RAIN FALL RATE LENGTH / TIME");
	fprintf(outASER, "%s\n", "#  EVAP = EVAPORATION RATE IF EVAPCVT>0.");
	fprintf(outASER, "%s\n", "#  SOLSWR = SOLAR SHORT WAVE RADIATION AT WATER SURFACE  ENERGY FLUX / UNIT AREA");
	fprintf(outASER, "%s\n", "#  CLOUD = FRATIONAL CLOUD COVER");
	fprintf(outASER, "%s\n", "#");
	fprintf(outASER, "%s\n", "#  MASER    TCASER  TAASER  IRELH    RAINCVT  EVAPCVT  SOLRCVT   CLDCVT");
	fprintf(outASER, "%s\n", "#");
	fprintf(outASER, "%s\n", "#  IASWRAD  REVC    RCHC    SWRATNF  SWRATNS  FSWRATF  DABEDT    TBEDIT    HTBED1    HTBED2");
	fprintf(outASER, "%s\n", "#");
	fprintf(outASER, "%s\n", "#  TASER(M) PATM(M) TDRY(M) TWET(M)  RAIN(M)  EVAP(M)  SOLSWR(M) CLOUD(M)");
	fprintf(outASER, "%s\n", "#                    /TEQ   /RELH                      /HTCOEF");

	//                                                        MASER  TCASER         TAASER  IRELH    RAINCVT                 EVAPCVT                  SOLRCVT   CLDCVT
	fprintf(outASER, "      %d %lf %lf %d %E %E %lf %lf\n", ntime, efdc_basetime, 1.0, 1, 1 / (data_basetime*1000.0), 1 / (data_basetime*1000.0), 1.0, 1.0);
	//                                                                IASWRAD  REVC    RCHC    SWRATNF  SWRATNS  FSWRATF  DABEDT    TBEDIT    HTBED1    HTBED2  
	fprintf(outASER, "      %d %lf %lf %lf %lf %lf %lf %lf %lf %lf\n", 1,     -1.0,   -1.0,    0.0,     0.0,     0.0,     10.0,     5.0,      0.1,      1.0E-7);

	int ig1, il, an;
	double stime = -1.000;

	for (ig1 = 0; ig1 < ntime; ig1++){
		an = -1;
		stime = stime + 1.000;

		fprintf(outASER, "      %lf ", (stime*data_basetime) / efdc_basetime);
		for (il = 0; il < 7; il++){
			an = an + 1;
			if (il == 6){

				fprintf(outASER, "%lf\n", ASER_val[ig1][an]);

			}
			else{

				fprintf(outASER, "%lf ", ASER_val[ig1][an]);

			}
		}

	}

	fclose(outASER);
}

__host__ void write_efdc_SHOW(const char *filepath,int NSTYPE,int NSHOWR,int ISHOWC,int JSHOWC,int NSHFREQ,int ZSSMIN,int ZSSMAX,int SSALMAX){

	FILE * outSHOW = fopen(filepath, "w");

	//                      SHOW.INP: LAYOUT  

	fprintf(outSHOW, "%s\n", "*  SHOW.INP FILE, IN FREE FORMAT ACROSS LINE");
	fprintf(outSHOW, "%s\n", "*");
	fprintf(outSHOW, "%s\n", "*  NSTYPE    NSHOWR    ISHOWC   JSHOWC    NSHFREQ");
	fprintf(outSHOW, "%s\n", "*");
	fprintf(outSHOW, "%s\n", "*  ZSSMIN    ZSSMAX    SSALMAX");
	fprintf(outSHOW, "%s\n", "*");

	fprintf(outSHOW, "    %d    %d    %d    %d    %d\n", \
		NSTYPE, NSHOWR, ISHOWC, JSHOWC, NSHFREQ);

	fprintf(outSHOW, "    %d    %d    %d\n", \
		ZSSMIN,ZSSMAX,SSALMAX);
	fclose(outSHOW);
}

__host__ void lake_catchment_smooth(int dimx, int dimy, int *mask, double *baseo){

	int ini, inj;
	//int bc_id=0;	
	
	int posxy;

	for (ini = 0; ini<dimy; ini++){
		posxy = dimy*dimx - (ini + 1)*dimx;
		posxy = posxy - 1;
		for (inj = 0; inj<dimx; inj++) {
			posxy = posxy + 1;
			//int bc_id = blockDim.x*blockIdx.x + threadIdx.x;
			//while (bc_id<N){
			//	int inj = bc_id % dimx;  // 0 to dimx  (cols)     
			//	int ini = bc_id / dimx;  // 0 to dimy  (rows)

			if (mask[posxy] != 1){
				if ((ini > 0) && (ini < dimy - 1)){
					//======================
					if (inj>0 && inj < dimx - 1){ //dimx-1	

						// South                posxy - dimx
						if (mask[posxy] != mask[posxy - dimx]){
							if(baseo[posxy + dimx] != -9999){						
								baseo[posxy] = (baseo[posxy - dimx]  + baseo[posxy + dimx])/2.0;
							}
						}
						
						// North                posxy + dimx
						if (mask[posxy] != mask[posxy + dimx]){
							if(baseo[posxy - dimx] != -9999){
								baseo[posxy] = (baseo[posxy - dimx] + baseo[posxy + dimx])/2.0;
							}
						}

						// East                 posxy - 1
						if (mask[posxy] != mask[posxy - 1]){
							if(baseo[posxy + 1] != -9999){
								baseo[posxy] = (baseo[posxy - 1] + baseo[posxy + 1])/2.0;
							}
						}
						
						// West                 posxy + 1
						if (mask[posxy] != mask[posxy + 1]){
							if(baseo[posxy - 1] != -9999){
								baseo[posxy] = (baseo[posxy - 1] + baseo[posxy + 1])/2.0;								
							}
						}
						
					}				
				}				
			}

			//bc_id = bc_id + 1;
			//	bc_id += gridDim.x * blockDim.x;
			//} // end while
			
		}
		//====================
	}
}

//__host__ void make_gefdc_inputs(const char *domain_filepath, const char *bathymetry_filepath,\
                                const char *grid_inputs_filepath){
__host__ void make_gefdc_inputs(const char *domain_filepath, const char *grid_inputs_filepath, double *bathy){									
									
   int er,np,sq,j_in,i,j,dimx,dimy,dec,min_dec,rows,cols,NDEPDAT,ISGG,IGM,JGM,ITRXM,ITRHM,ITRKM,ITRGM,NDEPSM,NDEPSMF,DDATADJ,ISIRKI,JSIRKI,ISIHIHJ,JSIHIHJ,ISIDEP,ISIDPTYP,ISVEG,NVEGDAT,NVEGTYP,ILT,JLT;
   float CDLON1,CDLON2,CDLON3,CDLAT1,CDLAT2,CDLAT3,DEPMIN,RPX,RPK,RPH,XSHIFT,YSHIFT,HSCALE,RKJDKI,ANGORO,CDEP,RADM,SURFELV,XLB,YLB;
   double res,RSQXM,RSQKM,RSQKIM,RSQHM,RSQHIM,RSQHJM,ZROUGH,xcoor,ycoor,resolution,NaN;
 
 
// Read input files

   FILE *matrix_file = fopen(domain_filepath,"r");              // domain

// FILE *bathy_file = fopen(bathymetry_filepath,"r");        // bathymetry

   FILE *grid_parameters = fopen(grid_inputs_filepath,"r"); // grid_make_inputs


//char c;
//   FILE * len;
//   len = fopen("matrix.txt","r");
//   rows = 0;
//   cols = 0;
   // Extract characters from file and store in character c 
//   for (c = getc(len); c != EOF; c = getc(len)){ 
//       cols = cols+1;
//       if (c == '\n'){ // Increment count if this character is newline 
//          rows = rows + 1;
//       }       
//   }  
//   cols = (cols/2)/rows; 
   /*
   fscanf(matrix_file," north: %f", &north);
   fscanf(matrix_file," south: %f", &south);
   fscanf(matrix_file," east: %f", &east);
   fscanf(matrix_file," west: %f", &west);
   fscanf(matrix_file," rows: %d", &rows);
   fscanf(matrix_file," cols: %d", &cols);
   */
   
    fscanf(matrix_file, " ncols %d\n", &cols);
	fscanf(matrix_file, " nrows %d\n", &rows);
	fscanf(matrix_file, " xllcorner %lf\n", &xcoor);
	fscanf(matrix_file, " yllcorner %lf\n", &ycoor);
    fscanf(matrix_file, " cellsize %lf\n", &resolution);
    fscanf(matrix_file, " NODATA_value %lf\n", &NaN);
	/*
	fscanf(bathy_file, " ncols %d\n", &cols);
	fscanf(bathy_file, " nrows %d\n", &rows);
	fscanf(bathy_file, " xllcorner %lf\n", &xcoor);
	fscanf(bathy_file, " yllcorner %lf\n", &ycoor);
    fscanf(bathy_file, " cellsize %lf\n", &resolution);
    fscanf(bathy_file, " NODATA_value %lf\n", &NaN);
	*/
	
   /*
   fscanf(bathy_file," north: %f", &north);
   fscanf(bathy_file," south: %f", &south);
   fscanf(bathy_file," east: %f", &east);
   fscanf(bathy_file," west: %f", &west);
   fscanf(bathy_file," rows: %d", &rows);
   fscanf(bathy_file," cols: %d", &cols);
   */
   dimy = rows;
   dimx = cols;


//   printf("Número de linhas: %d\n", rows);   
//   printf("Número de colunas: %d\n", cols);

    int **values, **M_cell_contours;
	float **bathymetry;
	
    values = (int**)malloc(rows*sizeof(int*));
	M_cell_contours = (int**)malloc(rows*sizeof(int*));
	bathymetry = (float**)malloc(rows*sizeof(float*));

		for (int iu = 0; iu < rows; iu++){
			values[iu] = (int*)malloc(cols * sizeof(int));
			M_cell_contours[iu] = (int*)malloc(cols * sizeof(int));
			bathymetry[iu] = (float*)malloc(cols * sizeof(float));
        }
   int nb = -1;
// Read domain_txt file   
   for (i=0; i<dimy; i++){
      for(j=0; j<dimx; j++){
                   
         if (j<dimx-1){
            fscanf(matrix_file,"%d", &values[i][j]);
         //   fscanf(bathy_file,"%f", &bathymetry[i][j]);
         //   printf("%d",values[i][j]);  
                   
         }
         else if (j=dimx-1) {            
            fscanf(matrix_file,"%d\n", &values[i][j]);  
         //   fscanf(bathy_file,"%f\n", &bathymetry[i][j]);          
         //   printf("%d\n",values[i][j]);                

         }
         
         nb = nb+1;		 
         bathymetry[i][j] = bathy[nb];		 
       
      }
   }

res = resolution;

// Read grid_make_input.txt file
//fscanf(grid_parameters," SPATIAL RESOLUTION");
//fscanf(grid_parameters," RESOLUTION: %lf",&res);
fscanf(grid_parameters," GRAPHICS GRID INFORMATION");
fscanf(grid_parameters," ISGG: %d",&ISGG);
fscanf(grid_parameters," IGM: %d",&IGM);
fscanf(grid_parameters," JGM: %d",&JGM);
fscanf(grid_parameters," CARTESIAN AND GRAPHICS GRID COORDINATE DATA");
fscanf(grid_parameters," CDLON1: %f",&CDLON1);
fscanf(grid_parameters," CDLON2: %f",&CDLON2);
fscanf(grid_parameters," CDLON3: %f",&CDLON3);
fscanf(grid_parameters," CDLAT1: %f",&CDLAT1);
fscanf(grid_parameters," CDLAT2: %f",&CDLAT2);
fscanf(grid_parameters," CDLAT3: %f",&CDLAT3);
fscanf(grid_parameters," SOLUTION ITERATIONS AND SMOOTHING INFORMATION");
fscanf(grid_parameters," ITRXM: %d",&ITRXM);
fscanf(grid_parameters," ITRHM: %d",&ITRHM);
fscanf(grid_parameters," ITRKM: %d",&ITRKM);
fscanf(grid_parameters," ITRGM: %d",&ITRGM);
fscanf(grid_parameters," NDEPSM: %d",&NDEPSM);
fscanf(grid_parameters," NDEPSMF: %d",&NDEPSMF);
fscanf(grid_parameters," DEPMIN: %f",&DEPMIN);
fscanf(grid_parameters," DDATADJ: %d",&DDATADJ);
fscanf(grid_parameters," PARAMETERS AND CONVERGENCE CRITERIA");
fscanf(grid_parameters," RPX: %f",&RPX);
fscanf(grid_parameters," RPK: %f",&RPK);
fscanf(grid_parameters," RPH: %f",&RPH);
fscanf(grid_parameters," RSQXM: %le",&RSQXM);
fscanf(grid_parameters," RSQKM: %le",&RSQKM);
fscanf(grid_parameters," RSQKIM: %le",&RSQKIM);
fscanf(grid_parameters," RSQHM: %le",&RSQHM);
fscanf(grid_parameters," RSQHIM: %le",&RSQHIM);
fscanf(grid_parameters," RSQHJM: %le",&RSQHJM);
fscanf(grid_parameters," COORDINATE SHIFT PARAMETERS");
fscanf(grid_parameters," XSHIFT: %f",&XSHIFT);
fscanf(grid_parameters," YSHIFT: %f",&YSHIFT);
fscanf(grid_parameters," HSCALE: %f",&HSCALE);
fscanf(grid_parameters," RKJDKI: %f",&RKJDKI);
fscanf(grid_parameters," ANGORO: %f",&ANGORO);
fscanf(grid_parameters," INTERPOLATION SWITCHES");
fscanf(grid_parameters," ISIRKI: %d",&ISIRKI);
fscanf(grid_parameters," JSIRKI: %d",&JSIRKI);
fscanf(grid_parameters," ISIHIHJ: %d",&ISIHIHJ);
fscanf(grid_parameters," JSIHIHJ: %d",&JSIHIHJ);
fscanf(grid_parameters," DEPTH INTERPOLATION SWITCHES");
fscanf(grid_parameters," ISIDEP: %d",&ISIDEP);
fscanf(grid_parameters," CDEP: %f",&CDEP);
fscanf(grid_parameters," RADM: %f",&RADM);
fscanf(grid_parameters," ISIDPTYP: %d",&ISIDPTYP);
fscanf(grid_parameters," SURFELV: %f",&SURFELV);
fscanf(grid_parameters," ISVEG: %d",&ISVEG);
fscanf(grid_parameters," NVEGDAT: %d",&NVEGDAT);
fscanf(grid_parameters," NVEGTYP: %d",&NVEGTYP);
fscanf(grid_parameters," ZROUGH: %lf",&ZROUGH);
fscanf(grid_parameters," LAST BOUNDARY POINT INFORMATION");
fscanf(grid_parameters," ILT: %d",&ILT);
fscanf(grid_parameters," JLT: %d",&JLT);
fscanf(grid_parameters," XLB: %f",&XLB);
fscanf(grid_parameters," YLB: %f",&YLB);



// Initi write CELL.INP
   FILE * wf_cell;
   wf_cell = fopen("CELL.INP","w");          //CELL.INP

// Initi write gridext.inp 
   FILE * wf_gridext;
   wf_gridext = fopen("gridext.inp","w");    //gridext.inp

// INiti write depdat.inp 
   FILE * wf_depdat;
   wf_depdat = fopen("depdat.inp","w");      //depdat.inp

// Initi write gefdc.inp
   FILE * wf_gefdc;
   wf_gefdc = fopen("gefdc.inp","w");        //gefdc.inp


//============================================
// Build CELL.INP
//============================================
//============================================
// Add cell contous 
   for (i=0;i<dimy;i++){
//======================  
    if ((i>0) && (i<dimy-1)){
      for (j=0;j<dimx;j++) {
//======================
         if (j>0 && j<dimx-1){          
             if (values[i][j] != 1){
               if ((values[i][j] != values[i+1][j]) || (values[i][j] != values[i-1][j]) || (values[i][j] != values[i][j+1]) || (values[i][j] != values[i][j-1])){
               M_cell_contours[i][j] = 1;            
               }
               else{
               M_cell_contours[i][j] = 0; 
               }
             } 
             else{
               M_cell_contours[i][j] = 1; 
             }        
         }
         else if (j>=dimx-1) {          
             if (values[i][j] != 1){
               if ((values[i][j] != values[i+1][j]) || (values[i][j] != values[i-1][j])|| (values[i][j] != values[i][j-1])){
               M_cell_contours[i][j] = 1;                
               }
               else{
               M_cell_contours[i][j] = 0; 
               }
             }   
             else{
               M_cell_contours[i][j] = 1; 
             }                  
         }     
         else if (j<=0) {          
             if (values[i][j] != 1){
               if ((values[i][j] != values[i+1][j]) || (values[i][j] != values[i-1][j])|| (values[i][j] != values[i][j+1])){
               M_cell_contours[i][j] = 1;                
               }
               else{
               M_cell_contours[i][j] = 0; 
               }
             }      
             else{
               M_cell_contours[i][j] = 1; 
             }               
         }                       
//=================    
      }
    }
//====================
    else if (i<=0){
//====================    
      for (j=0;j<dimx;j++) {
//======================
         if (j>0 && j<dimx-1){          
             if (values[i][j] != 1){
               if ((values[i][j] != values[i+1][j]) || (values[i][j] != values[i][j+1]) || (values[i][j] != values[i][j-1])){
               M_cell_contours[i][j] = 1;                
               }
               else{
               M_cell_contours[i][j] = 0; 
               }
             } 
             else{
               M_cell_contours[i][j] = 1; 
             }        
         }
         else if (j>=dimx-1) {          
             if (values[i][j] != 1){
               if ((values[i][j] != values[i+1][j]) || (values[i][j] != values[i][j-1])){
               M_cell_contours[i][j] = 1;                
               }
               else{
               M_cell_contours[i][j] = 0; 
               }
             }                 
         }     
         else if (j<=0) {          
             if (values[i][j] != 1){
               if ((values[i][j] != values[i+1][j]) || (values[i][j] != values[i][j+1])){
               M_cell_contours[i][j] = 1;                
               }
               else{
               M_cell_contours[i][j] = 0; 
               }
             }                 
         }                    
//====================    
      }
//====================
    }
//====================
//====================
    else if (i>=dimy-1){
//====================    
      for (j=0;j<dimx;j++) {
//======================
        if (j>0 && j<dimx-1){          
             if (values[i][j] != 1){
               if ((values[i][j] != values[i-1][j]) || (values[i][j] != values[i][j+1]) || (values[i][j] != values[i][j-1])){
               M_cell_contours[i][j] = 1;                
               }
               else{
               M_cell_contours[i][j] = 0; 
               }
             } 
             else{
               M_cell_contours[i][j] = 1; 
             }        
         }
         else if (j>=dimx-1) {          
             if (values[i][j] != 1){
               if ((values[i][j] != values[i-1][j])|| (values[i][j] != values[i][j-1])){
               M_cell_contours[i][j] = 1;               
               }
               else{
               M_cell_contours[i][j] = 0; 
               }
             }                 
         }     
         else if (j<=0) {          
             if (values[i][j] != 1){
               if ((values[i][j] != values[i-1][j])|| (values[i][j] != values[i][j+1])){
               M_cell_contours[i][j] = 1;                
               }
               else{
               M_cell_contours[i][j] = 0; 
               }
             }                 
         }                   
//====================    
      }
//====================
    }
//====================
   }
//====================================
//====================================



// Build layout CELL.INP

//C cell.inp file, i columns and j rows
//C    0        1
//C    1234567890
//C
//  6  999999999
//  5  955559999
//  4  955555559
//  3  955555559
//  2  955559999
//  1  999999999
//C
//C    1234567890
//C    0        1

fprintf(wf_cell,"%s %d %s %d %s\n","C cell.inp file,",cols,"columns and",rows,"rows @");

int ncl = 120;

np = (cols/ncl);
min_dec = 11;

np = np+1;

if (np==1){
  // np=1;
  if (dimx/10<=10){
     min_dec = (dimx/10)+1;
  }
}
else if(np>1){
dimx = ncl;//cols/np;
}
er = (cols - (np-1)*ncl);//np*(cols/np));


dec = ((dimx)/10)+1;
//printf("%d\n",(cols - (np-1)*ncl));//np*(cols/np)));

NDEPDAT=0;


// Write CELL.INP column ID ======
fprintf(wf_cell,"%s","C");
fprintf(wf_cell,"    %d",0);
if (min_dec<=3){ // <=2
fprintf(wf_cell,"        %d\n",1);
}
else{
fprintf(wf_cell,"        %d",1);
}
for (i=2;i<min_dec-1;i++){
   if (i==min_dec-2){
     fprintf(wf_cell,"         %d\n",i);
   }
   else{
     fprintf(wf_cell,"         %d",i);
   }
}
//
fprintf(wf_cell,"%s","C");
fprintf(wf_cell,"   %s"," ");
for (i=1;i<dec;i++){
   for (j=1;j<10;j++){
      fprintf(wf_cell,"%d",j);
   }
   if (i==dec-1){
      fprintf(wf_cell,"%d\n",0);
   }
   else{
      fprintf(wf_cell,"%d",0);
   }
}
//
fprintf(wf_cell,"%s\n","C");


j_in = -ncl;//(cols/np);
//==================================
for (sq = 0;sq<np;sq++){
   
   j_in = j_in+ncl;//(cols/np);

   if (sq == np-1){
      dimx = j_in+er;//(cols/np) + er;
   }
   else{
      dimx = j_in+ncl;//(cols/np);
   }



// ================================================================

// Definition of domain limits 
   for (i=0;i<dimy;i++){

     // Write CELL.INP line ID ==========
     //                        1000000
     //                         100000  
     //                          10000
     //                           1000 
     //                            100
     //                             10
     //                              1
         //if ((dimy)>=1000000){    
           //fprintf(wf_cell," %d  ",dimy-i); 
         //}
         //else if (((dimy)>=100000 && (dimy)<1000000)){
           //fprintf(wf_cell," %d  ",dimy-i);
         //}
         if (dimy>=10000){
           //fprintf(wf_cell," %d  ",dimy-i);
           if (((dimy-i)>=10000) && ((dimy-i)<100000)){
           fprintf(wf_cell,"%d  ",dimy-i);
           }
           else if (((dimy-i)>=1000) && ((dimy-i)<10000)){
           fprintf(wf_cell," %d  ",dimy-i);
           }
           else if (((dimy-i)>=100) && ((dimy-i)<1000)){
           fprintf(wf_cell,"  %d  ",dimy-i);
           }
           else if (((dimy-i)>=10) && ((dimy-i)<100)){
           fprintf(wf_cell,"   %d  ",dimy-i);
           }
           else if ((dimy-i)<10){
           fprintf(wf_cell,"    %d  ",dimy-i);
           }
         }
         else if (dimy>=1000){
           if (((dimy-i)>=1000) && ((dimy-i)<10000)){
           fprintf(wf_cell,"%d  ",dimy-i);
           }
           else if (((dimy-i)>=100) && ((dimy-i)<1000)){
           fprintf(wf_cell," %d  ",dimy-i);
           }
           else if (((dimy-i)>=10) && ((dimy-i)<100)){
           fprintf(wf_cell,"  %d  ",dimy-i);
           }
           else if ((dimy-i)<10){
           fprintf(wf_cell,"   %d  ",dimy-i);
           }
         }
         else if (dimy>=100){
           if (((dimy-i)>=100) && ((dimy-i)<1000)){
           fprintf(wf_cell,"%d  ",dimy-i);
           }
           else if (((dimy-i)>=10) && ((dimy-i)<100)){
           fprintf(wf_cell," %d  ",dimy-i);
           }
           else if ((dimy-i)<10){
           fprintf(wf_cell,"  %d  ",dimy-i);
           }
         }
         else if (dimy>=10){                   
           if (((dimy-i)>=10) && ((dimy-i)<100)){
           fprintf(wf_cell," %d  ",dimy-i);
           }
           else if ((dimy-i)<10){
           fprintf(wf_cell,"  %d  ",dimy-i);
           }
         }
         else if (dimy<10){
           fprintf(wf_cell," %d  ",dimy-i); 
         }
     // ==================================
//======================  
    for (j=j_in;j<dimx;j++) {      
        if ((i>0) && (i<dimy-1)){
             fprintf(wf_gridext,"%d %d %lf %lf\n",j+1,(rows-i),j*res,(rows-i-1)*res);   // Gridext file;
//======================
         if (j>j_in && j<dimx-1){ //dimx-1        
             if (values[i][j] != 1){
               if ((M_cell_contours[i][j] != M_cell_contours[i+1][j]) || (M_cell_contours[i][j] != M_cell_contours[i-1][j]) || (M_cell_contours[i][j] != M_cell_contours[i][j+1]) || (M_cell_contours[i][j] != M_cell_contours[i][j-1])){
               //printf("%d",9);
               fprintf(wf_cell,"%d",9);    
               }
               else{
                   if(M_cell_contours[i][j] == 1){
                      //printf("%d",9);
                      fprintf(wf_cell,"%d",9);                     
                   }
                   else{
                      //printf("%d",0);
                      fprintf(wf_cell,"%d",0);
                   }
               }
             } 
             else{
               //printf("%d",5);
               fprintf(wf_cell,"%d",5);
               NDEPDAT = NDEPDAT + 1; 
               fprintf(wf_depdat,"%lf %lf %f\n",(j+(j+1))*res/2,((rows-i)+((rows-i)-1))*res/2, bathymetry[i][j]);   // depdat file;
             }        
         }
         else if (j>=dimx-1) { //dimx-1          

             if (values[i][j] != 1){
               if ((M_cell_contours[i][j] != M_cell_contours[i+1][j]) || (M_cell_contours[i][j] != M_cell_contours[i-1][j])|| (M_cell_contours[i][j] != M_cell_contours[i][j-1])){
               //printf("%d\n",9);   
               fprintf(wf_cell,"%d\n",9);    
     
               }
               else{
                   if(M_cell_contours[i][j] == 1){
                      //printf("%d\n",9);
                      fprintf(wf_cell,"%d\n",9);                     
                   }
                   else{
                      //printf("%d\n",0);
                      fprintf(wf_cell,"%d\n",0);
                   }
               }
             }  
             else{
               //printf("%d\n",5);
               fprintf(wf_cell,"%d\n",5);      
               NDEPDAT = NDEPDAT + 1;    
               fprintf(wf_depdat,"%lf %lf %f\n",(j+(j+1))*res/2,((rows-i)+((rows-i)-1))*res/2, bathymetry[i][j]);   // depdat file;
             }               
         }     
         else if (j<=j_in) {          
             if (values[i][j] != 1){
               if ((M_cell_contours[i][j] != M_cell_contours[i+1][j]) || (M_cell_contours[i][j] != M_cell_contours[i-1][j])|| (M_cell_contours[i][j] != M_cell_contours[i][j+1])){
               //printf("%d",9);  
               fprintf(wf_cell,"%d",9);            
               }
               else{
                   if(M_cell_contours[i][j] == 1){
                      //printf("%d",9);
                      fprintf(wf_cell,"%d",9);                     
                   }
                   else{
                      //printf("%d",0);
                      fprintf(wf_cell,"%d",0);
                   }
               }
             }   
             else{
               //printf("%d",5);
               fprintf(wf_cell,"%d",5);    
               NDEPDAT = NDEPDAT + 1;    
               fprintf(wf_depdat,"%lf %lf %f\n",(j+(j+1))*res/2,((rows-i)+((rows-i)-1))*res/2, bathymetry[i][j]);   // depdat file; 
             }               
         }                       
//=================    
      }
//====================
      else if (i<=0){
//====================     
         fprintf(wf_gridext,"%d %d %lf %lf\n",j+1,(rows-i),j*res,(rows-i-1)*res);   // Gridext file;
//======================
         if (j>j_in && j<dimx-1){  //dimx-1   
      
             if (values[i][j] != 1){
               if ((M_cell_contours[i][j] != M_cell_contours[i+1][j]) || (M_cell_contours[i][j] != M_cell_contours[i][j+1]) || (M_cell_contours[i][j] != M_cell_contours[i][j-1])){
               //printf("%d",9);        
               fprintf(wf_cell,"%d",9);        
               
               }
               else{
                   if(M_cell_contours[i][j] == 1){
                      //printf("%d",9);
                      fprintf(wf_cell,"%d",9);                     
                   }
                   else{
                      //printf("%d",0);
                      fprintf(wf_cell,"%d",0);
                   }
               }
             } 
             else{
               //printf("%d",5);
               fprintf(wf_cell,"%d",5);
               NDEPDAT = NDEPDAT + 1; 
               fprintf(wf_depdat,"%lf %lf %f\n",(j+(j+1))*res/2,((rows-i)+((rows-i)-1))*res/2, bathymetry[i][j]);   // depdat file;
             }        
         }
         else if (j>=dimx-1) {    //dimx-1       
             if (values[i][j] != 1){
               if ((M_cell_contours[i][j] != M_cell_contours[i+1][j]) || (M_cell_contours[i][j] != M_cell_contours[i][j-1])){
               //printf("%d\n",9);       
               fprintf(wf_cell,"%d\n",9);         

               }
               else{
                   if(M_cell_contours[i][j] == 1){
                      //printf("%d\n",9);
                      fprintf(wf_cell,"%d\n",9);                     
                   }
                   else{
                      //printf("%d\n",0);
                      fprintf(wf_cell,"%d\n",0);
                   }
               }
             }           
             else{
               //printf("%d\n",5);
               fprintf(wf_cell,"%d\n",5);    
               NDEPDAT = NDEPDAT + 1;    
               fprintf(wf_depdat,"%lf %lf %f\n",(j+(j+1))*res/2,((rows-i)+((rows-i)-1))*res/2, bathymetry[i][j]);   // depdat file;     
             }       
         }     
         else if (j<=j_in) {          
             if (values[i][j] != 1){
               if ((M_cell_contours[i][j] != M_cell_contours[i+1][j]) || (M_cell_contours[i][j] != M_cell_contours[i][j+1])){
               //printf("%d",9);   
               fprintf(wf_cell,"%d",9);      
      
               }
               else{
                   if(M_cell_contours[i][j] == 1){
                      //printf("%d",9);
                      fprintf(wf_cell,"%d",9);                     
                   }
                   else{
                      //printf("%d",0);
                      fprintf(wf_cell,"%d",0);
                   }
               }
             }  
             else{
               //printf("%d",5);
               fprintf(wf_cell,"%d",5);   
               NDEPDAT = NDEPDAT + 1;      
               fprintf(wf_depdat,"%lf %lf %f\n",(j+(j+1))*res/2,((rows-i)+((rows-i)-1))*res/2, bathymetry[i][j]);   // depdat file;
             }                
         }                    
//====================
      }
//====================
//====================
      else if (i>=dimy-1){
//====================    
                   
        fprintf(wf_gridext,"%d %d %lf %lf\n",j+1,(rows-i),j*res,(rows-i-1)*res);   // Gridext file;
//======================
        if (j>j_in && j<dimx-1){   //dimx-1 
             if (values[i][j] != 1){
               if ((M_cell_contours[i][j] != M_cell_contours[i-1][j]) || (M_cell_contours[i][j] != M_cell_contours[i][j+1]) || (M_cell_contours[i][j] != M_cell_contours[i][j-1])){
               //printf("%d",9);      
               fprintf(wf_cell,"%d",9);         

               }
               else{
                   if(M_cell_contours[i][j] == 1){
                      //printf("%d",9);
                      fprintf(wf_cell,"%d",9);                     
                   }
                   else{
                      //printf("%d",0);
                      fprintf(wf_cell,"%d",0);
                   }
               }
             } 
             else{
               //printf("%d",5);
               fprintf(wf_cell,"%d",5);  
               NDEPDAT = NDEPDAT + 1;    
               fprintf(wf_depdat,"%lf %lf %f\n",(j+(j+1))*res/2,((rows-i)+((rows-i)-1))*res/2, bathymetry[i][j]);   // depdat file;    
             }        
         }
         else if (j>=dimx-1) {     //dimx-1      
             if (values[i][j] != 1){
               if ((M_cell_contours[i][j] != M_cell_contours[i-1][j])|| (M_cell_contours[i][j] != M_cell_contours[i][j-1])){
               //printf("%d\n",9);   
               fprintf(wf_cell,"%d\n",9);          

               }
               else{
                   if(M_cell_contours[i][j] == 1){
                      //printf("%d\n",9);
                      fprintf(wf_cell,"%d\n",9);                     
                   }
                   else{
                      //printf("%d\n",0);
                      fprintf(wf_cell,"%d\n",0);
                   }
               }
             }         
             else{
               //printf("%d\n",5);
               fprintf(wf_cell,"%d\n",5);    
               NDEPDAT = NDEPDAT + 1;       
               fprintf(wf_depdat,"%lf %lf %f\n",(j+(j+1))*res/2,((rows-i)+((rows-i)-1))*res/2, bathymetry[i][j]);   // depdat file;
             }         
         }     
         else if (j<=j_in) {          
             if (values[i][j] != 1){
               if ((M_cell_contours[i][j] != M_cell_contours[i-1][j])|| (M_cell_contours[i][j] != M_cell_contours[i][j+1])){
               //printf("%d",9); 
               fprintf(wf_cell,"%d",9);            
               }
               else{
                   if(M_cell_contours[i][j] == 1){
                      //printf("%d",9);
                      fprintf(wf_cell,"%d",9);                     
                   }
                   else{
                      //printf("%d",0);
                      fprintf(wf_cell,"%d",0);
                   }
               }
             }  
             else{
               //printf("%d",5);
               fprintf(wf_cell,"%d",5); 
               NDEPDAT = NDEPDAT + 1;       
               fprintf(wf_depdat,"%lf %lf %f\n",(j+(j+1))*res/2,((rows-i)+((rows-i)-1))*res/2, bathymetry[i][j]);   // depdat file;
             }                
         }                   
//====================
      }  
//====================
    }
//====================
   }
//====================
} // for sq end
//====================


fprintf(wf_cell,"%s\n","C");
//
fprintf(wf_cell,"%s","C");
fprintf(wf_cell,"   %s"," ");
for (i=1;i<dec;i++){
   for (j=1;j<10;j++){
      fprintf(wf_cell,"%d",j);
   }
   if (i== dec-1){
      fprintf(wf_cell,"%d\n",0);
   }
   else{
      fprintf(wf_cell,"%d",0);
   }
}
//
fprintf(wf_cell,"%s","C");
fprintf(wf_cell,"    %d",0);
if (dec<=2){
fprintf(wf_cell,"        %d\n",1);
}
else{
fprintf(wf_cell,"        %d",1);
}
for (i=2;i<min_dec-1;i++){
   if (i == min_dec-2){
     fprintf(wf_cell,"         %d\n",i);
   }
   else{
     fprintf(wf_cell,"         %d",i);
   }
}

//printf("%d\n",NDEPDAT);



//==========================================================
//                Build gefdc.inp
//==========================================================

// Write gefdc.inp

fprintf(wf_gefdc,"%s\n","C1 TITLE");
fprintf(wf_gefdc,"%s\n","C1 (LIMITED TO 80 CHARACTERS)");
fprintf(wf_gefdc,"%s\n","'gefdc.inp DHARA-EFDC'");
fprintf(wf_gefdc,"%s\n","C2 INTEGER INPUT");
fprintf(wf_gefdc,"%s\n","C2 NTYPE  NBPP  IMIN  IMAX  JMIN  JMAX  IC   JC");
       fprintf(wf_gefdc,"   %d     %d    %d    %d    %d    %d    %d   %d\n",
                             0,    0,    1,    cols, 1,    rows, cols,rows);
fprintf(wf_gefdc,"%s\n","C3 GRAPHICS GRID INFORMATION");
fprintf(wf_gefdc,"%s\n","C3 ISGG IGM JGM DXCG DYCG NWTGG");
       fprintf(wf_gefdc,"   %d   %d  %d  %f   %f   %d\n",
                            ISGG,IGM,JGM,res,res,1);							
fprintf(wf_gefdc,"%s\n","C4 CARTESIAN AND GRAPHICS GRID COORDINATE DATA");
fprintf(wf_gefdc,"%s\n","C4 CDLON1 CDLON2 CDLON3 CDLAT1 CDLAT2 CDLAT3");
       fprintf(wf_gefdc,"   %f     %f     %f     %f     %f     %f\n",
                            CDLON1,CDLON2,CDLON3,CDLAT1,CDLAT2,CDLAT3);
fprintf(wf_gefdc,"%s\n","C5 INTEGER INPUT");
fprintf(wf_gefdc,"%s\n","C5 ITRXM ITRHM ITRKM ITRGM NDEPSM NDEPSMF DEPMIN DDATADJ");
       fprintf(wf_gefdc,"   %d    %d    %d    %d    %d     %d      %f     %d\n",
                            ITRXM,ITRHM,ITRKM,ITRGM,NDEPSM,NDEPSMF,DEPMIN,DDATADJ);
fprintf(wf_gefdc,"%s\n","C6 REAL INPUT");
fprintf(wf_gefdc,"%s\n","C6 RPX RPK RPH RSQXM RSQKM RSQKIM RSQHM RSQHIM RSQHJM");
       fprintf(wf_gefdc,"   %f  %f  %f  %E    %E    %E     %E    %E     %E\n",
                            RPX,RPK,RPH,RSQXM,RSQKM,RSQKIM,RSQHM,RSQHIM,RSQHJM);
fprintf(wf_gefdc,"%s\n","C7 COORDINATE SHIFT PARAMETERS");
fprintf(wf_gefdc,"%s\n","C7 XSHIFT YSHIFT HSCALE RKJDKI ANGORO");
       fprintf(wf_gefdc,"   %f     %f     %f     %f     %f\n",
                            XSHIFT,YSHIFT,HSCALE,RKJDKI,ANGORO);
fprintf(wf_gefdc,"%s\n","C8 INTERPOLATION SWITCHES");
fprintf(wf_gefdc,"%s\n","C8 ISIRKI JSIRKI ISIHIHJ JSIHIHJ");
       fprintf(wf_gefdc,"   %d     %d     %d      %d\n",
                            ISIRKI,JSIRKI,ISIHIHJ,JSIHIHJ);
fprintf(wf_gefdc,"%s\n","C9 NTYPE = 7 SPECIFID INPUT");
fprintf(wf_gefdc,"%s\n","C9 IB IE JB JE N7RLX NXYIT ITN7M IJSMD ISMD JSMD RP7 SERRMAX");
fprintf(wf_gefdc,"%s\n","C10 NTYPE = 7 SPECIFID INPUT");
fprintf(wf_gefdc,"%s\n","C10 X Y IN ORDER (IB,JB) (IE,JB) (IE,JE) (IB,JE)");
fprintf(wf_gefdc,"%s\n","C11 DEPTH INTERPOLATION SWITCHES");
fprintf(wf_gefdc,"%s\n","C11 ISIDEP NDEPDAT CDEP RADM ISIDPTYP SURFELV ISVEG NVEGDAT NVEGTYP");
       fprintf(wf_gefdc,"    %d     %d      %f   %f   %d       %f      %d    %d      %d     %lf\n",
                             ISIDEP,NDEPDAT,CDEP,RADM,ISIDPTYP,SURFELV,ISVEG,NVEGDAT,NVEGTYP,ZROUGH);
fprintf(wf_gefdc,"%s\n","C12 LAST BOUNDARY POINT INFORMATION");
fprintf(wf_gefdc,"%s\n","C12 ILT JLT X(ILT,JLT) Y(ILT,JLT)");
       fprintf(wf_gefdc,"    %d  %d       %f         %f\n",
                             ILT,JLT,     XLB,         YLB);
fprintf(wf_gefdc,"%s\n","C13 BOUNDARY POINT INFORMATION");
fprintf(wf_gefdc,"%s\n","C13 I J X(I,J) Y(I,J)");
    

   fclose(matrix_file);
  // fclose(bathy_file);
   fclose(grid_parameters);
   fclose(wf_cell);
   fclose(wf_gridext);
   fclose(wf_depdat);
   fclose(wf_gefdc);
   
}

int main()
{

	// Definition of integer type variables - scalar
	int dir_number, out_velocity_x, out_velocity_y, out_elevation,\
	out_depth, out_outlet_on, dir_it, N, cols, rows, i, j, k, lpout,\
	lkout, nst,n_out,outx,outy,efdc;

	// Definition of variables of type double - scalar
	double resolution, tday0, thour0, tmin0, tsec0, tday, thour, tmin, tsec, dkout, dpout, time0, \
		timmax, dt, dtoq, dt2, gg, manning_coef, dtrain, NaN, hmn, rr;// north, south, east, west;

	// Definition of string variables
	std::string dir_parameters, dirfile_setup, dir_DEM, dir_overQ,\
	dir_rain, initi_cond, outlet_file, dir_temperature,\
	dir_solar_radiation, dir_mask, dir_wind_velocities, dir_meteo_data;
	
	std::string tempo;

	std::stringstream out;

	// Definition of processing variables in CPU - vector / array
	double *h_baseo, *h_h, *h_um, *h_hm, *h_uu1, *h_umo, \
		*h_vv1, *h_vva, *h_vn, *h_hn, *h_vno, *h_uua, **h_qq, \
		*h_rain, *h_ho, *h_uu, *h_vv, *h_dx, *h_dy,\
		*h_ql, *h_rr, *h_th, *h_initial_condition,\
		**h_wind_vel, **h_efdc_ASER;

	double *h_brx, *h_bry,*h_outx,*h_outy, xcoor, ycoor;
	int *h_inf, *h_infx, *h_infy, *h_infsw, *h_outlet,*h_mask;

	// Evaporation variables definition
	double albedo, INT, INF, LWL, EV_WL_min,*h_T, *h_Rg, *h_Evapo, *h_Ev;
	int evaporation_on;

	// Definition of processing variables in GPU - vector / array
	double *d_baseo, *d_h, *d_um, *d_hm, *d_uu1, *d_umo, \
		*d_vv1, *d_vva, *d_vn, *d_hn, *d_vno, *d_uua, *d_qq, *d_rain, *d_ho, *d_uu, *d_vv, *d_dx, *d_dy, *d_ql, *d_rr, *d_th;
	int *d_inf, *d_infx, *d_infy, *d_infsw;// , *d_outlet;

	// Evaporation variables definition
	double *d_T, *d_Rg, *d_Rs, *d_pw, *d_lv, *d_Evapo, *d_Ev;

	double duration;

	int numBlocks;                     //Number of blocks
	int threadsPerBlock;               //Number of threads
	int maxThreadsPerBlock;

	char dirfile[4000];//, fdem[50], fqq[50], frain[50], fdesignrain[50];
	//*************************************************
	FILE *dir = fopen("db\/dir.txt", "r");
	fscanf(dir, " dir_number: %d/n", &dir_number);

	for (dir_it = 0; dir_it < dir_number; dir_it++){
		fscanf(dir, " %s/n", dirfile);
		dir_parameters = dirfile;

		FILE *file_setup;
		dirfile_setup = "db\/" + dir_parameters + "\/SW2D\/input\/setup.dat";
		file_setup = fopen(dirfile_setup.c_str(), "r");
		if (file_setup == NULL) {
			printf("unknown file - setup.dat\n");
			system("pause");
			return 0;
		}

		fscanf(file_setup, " tday0_thour0_tmin0_tsec0: %lf %lf %lf %lf\n ", &tday0, &thour0, &tmin0, &tsec0);
		fscanf(file_setup, " tday_thour_tmin_tsec: %lf %lf %lf %lf\n ", &tday, &thour, &tmin, &tsec);
		fscanf(file_setup, " dt: %lf\n ", &dt);
		fscanf(file_setup, " dpout: %lf\n ", &dpout);
		fscanf(file_setup, " dkout: %lf\n ", &dkout);
		fscanf(file_setup, " dtoq: %lf\n ", &dtoq);
		fscanf(file_setup, " evaporation_on: %d\n", &evaporation_on);
		fscanf(file_setup, " EV_WL_min: %lf\n ", &EV_WL_min);
		fscanf(file_setup, " INT: %lf\n ", &INT);
		fscanf(file_setup, " INF: %lf\n ", &INF);
		fscanf(file_setup, " LWL: %lf\n ", &LWL);
		fscanf(file_setup, " manning_coef: %lf\n ", &manning_coef);
		fscanf(file_setup, " out_velocity_x: %d\n ", &out_velocity_x);
		fscanf(file_setup, " out_velocity_y: %d\n ", &out_velocity_y);
		fscanf(file_setup, " out_elevation: %d\n ", &out_elevation);
		fscanf(file_setup, " out_depth: %d\n ", &out_depth);
		fscanf(file_setup, " out_outlet_on: %d\n ", &out_outlet_on);
		fscanf(file_setup, " efdc: %d\n ", &efdc);
		fclose(file_setup);

	
		// ******************************************************************
		dt2 = 2.00*dt;
		gg = 9.80;
		// ******************************************************************
		// ******************************************************************
		//                            Input MDT
		// ******************************************************************

		dir_DEM = "db\/" + dir_parameters + "\/SW2D\/input\/dem.txt";
		FILE *V_DEM = fopen(dir_DEM.c_str(), "r");
		if (V_DEM == NULL) {
			printf("unknown file - dem.txt\n");
			system("pause");
			return 0;
		}
		fscanf(V_DEM, " ncols %d\n", &cols);
		fscanf(V_DEM, " nrows %d\n", &rows);
		fscanf(V_DEM, " xllcorner %lf\n", &xcoor);
		fscanf(V_DEM, " yllcorner %lf\n", &ycoor);
		fscanf(V_DEM, " cellsize %lf\n", &resolution);
		fscanf(V_DEM, " NODATA_value %lf\n", &NaN);
		
		N = (rows)*(cols)+cols/2;
		
		h_baseo = (double*)malloc(N*sizeof(double));
		for (i = 0; i < N; i++){
			fscanf(V_DEM, "%lf\n", &h_baseo[i]);
		}
		fclose(V_DEM);
		// ****************************************************************
		//              Input coord source or sink
		// ****************************************************************

		FILE *file_coord_source;
		std::string dirfile_coord_source = "db\/" + dir_parameters + "\/SW2D\/input\/coord_source_sink.dat";
		file_coord_source = fopen(dirfile_coord_source.c_str(), "r");
		if (file_coord_source == NULL) {
			printf("unknown file - coord_source_sink.dat\n");
			system("pause");
			return 0;
		}
		fscanf(file_coord_source, "nst  %d\n ", &nst);
		h_brx = (double*)malloc(nst*sizeof(double));
		h_bry = (double*)malloc(nst*sizeof(double));
		for (int inst = 0; inst < nst; inst++){
			fscanf(file_coord_source, " %lf %lf\n ", &h_brx[inst], &h_bry[inst]);
			h_brx[inst] = h_brx[inst] / resolution;
			h_bry[inst] = h_bry[inst] / resolution;
		}
		fclose(file_coord_source);


		// ******************************************************************
		//                            Input outlet
		// ******************************************************************	
		/*
		if (out_outlet_on == 1){
			outlet_file = "db\/" + dir_parameters + "\/input\/outlet.dat";
			FILE *V_outlet = fopen(outlet_file.c_str(), "r");
			if (V_outlet == NULL) {
				printf("unknown file - outlet.dat\n");
				system("pause");
				return 0;
			}
			h_outlet = (int*)malloc(N*sizeof(int));
			for (i = 0; i < N; i++){
				fscanf(V_outlet, "%d\n", &h_outlet[i]);
			}
			fclose(V_outlet);
		}
		*/
		// ******************************************************************
		//                    Input source or sink
		// ******************************************************************
		int cont_qq = 0;
		int ch = 0;
		dir_overQ = "db\/" + dir_parameters + "\/SW2D\/input\/Q_source_sink.dat";
		FILE *V_dir_overQ = fopen(dir_overQ.c_str(), "r");
		if (V_dir_overQ == NULL) {
			printf("unknown file - Q_source_sink.dat\n");
			system("pause");
			return 0;
		}

		//Count the number of elements in the Q_source_sink.dat file
		while (!feof(V_dir_overQ))
		{
			ch = fgetc(V_dir_overQ);
			if (ch == '\n')
			{
				cont_qq++;
			}
		}
		fclose(V_dir_overQ);
		h_qq = (double**)malloc((cont_qq)*sizeof(double*));
		for (int iu = 0; iu < cont_qq; iu++){
			h_qq[iu] = (double*)malloc(nst*sizeof(double));
		}
		FILE *V1_dir_overQ = fopen(dir_overQ.c_str(), "r");
		char skip[10];
		fscanf(V1_dir_overQ, "flow %s\n", skip);
		for (int jq = 0; jq < cont_qq; jq++){
			for (int iq = 0; iq < nst; iq++){
				fscanf(V1_dir_overQ, " %lf\n", &h_qq[jq][iq]);	
				/*
					if (iq==(nst-1)){
						printf(" %lf\n", h_qq[jq][iq]);
					}
					else{
						printf(" %lf", h_qq[jq][iq]);
					}
                */
			}
		}
		fclose(V1_dir_overQ);

		//system("pause");
		// ****************************************************************
		//                  Input meteorological data
		// ****************************************************************

		//>>>>>>>>>>>>>>>>>>> Input Mensurement Rain <<<<<<<<<<<<<<<<<<<<<<
		int cont_rain = 0;
		int chr = 0;
		dir_rain = "db\/" + dir_parameters + "\/SW2D\/input\/rain.dat";
		FILE *V_dir_rain = fopen(dir_rain.c_str(), "r");
		if (V_dir_rain == NULL) {
			printf("unknown file - rain.dat\n");
			system("pause");
			return 0;
		}
		// Count the number of elements in the rain.dat file
		while (!feof(V_dir_rain))
		{
			chr = fgetc(V_dir_rain);
			if (chr == '\n')
			{
				cont_rain++;
			}
		}
		fclose(V_dir_rain);
		FILE *V1_dir_rain = fopen(dir_rain.c_str(), "r");
		char head1_rain[10], head2_rain[10], head3_rain[10], head4_rain[10];//, head5_rain[10], head6_rain[10], head7_rain[10];

		fscanf(V1_dir_rain, " %lf\n", &dtrain);
		fscanf(V1_dir_rain, " %s %s %s %s\n", head1_rain, head2_rain, head3_rain, head4_rain);// , head5_rain, head6_rain, head7_rain);
		h_rain = (double*)malloc((cont_rain - 1)*sizeof(double));
		//std::string rtime;
		char rtime[10];
		for (i = 0; i < (cont_rain - 1); i++){
			fscanf(V1_dir_rain, " %s %lf\n", rtime, &h_rain[i]);
		}
		fclose(V1_dir_rain);
		
		//>>>>>>>>>> Input Temperatures and solar radiation data <<<<<<<<<<<<<<<<<
		h_Ev = (double*)malloc(sizeof(double));
		cudaMalloc((void**)&d_Ev, sizeof(double));
		h_Ev[0] = 0.00;
		cudaMemcpy(d_Ev, h_Ev, sizeof(double), cudaMemcpyHostToDevice);
		
		if (evaporation_on == 1){

			char head_solar_rad[30],head_temperature[20];

			h_Evapo = (double*)malloc((cont_rain - 1)*sizeof(double));
			h_T = (double*)malloc((cont_rain - 1)*sizeof(double));
			h_Rg = (double*)malloc((cont_rain - 1)*sizeof(double));

			cudaMalloc((void**)&d_T, (cont_rain - 1)*sizeof(double));
			cudaMalloc((void**)&d_Rg, (cont_rain - 1)*sizeof(double));
			cudaMalloc((void**)&d_Rs, (cont_rain - 1)*sizeof(double));
			cudaMalloc((void**)&d_pw, (cont_rain - 1)*sizeof(double));
			cudaMalloc((void**)&d_lv, (cont_rain - 1)*sizeof(double));
			cudaMalloc((void**)&d_Evapo, (cont_rain - 1)*sizeof(double));

			dir_temperature = "db\/" + dir_parameters + "\/SW2D\/input\/temperatures.txt";
			FILE *V_dir_temperature = fopen(dir_temperature.c_str(), "r");
			if (V_dir_temperature == NULL) {
				printf("unknown file - evaporation.dat\n");
				system("pause");
				return 0;
			}
			dir_solar_radiation = "db\/" + dir_parameters + "\/SW2D\/input\/solar_radiation.txt";
			FILE *V_dir_solar_radiation = fopen(dir_solar_radiation.c_str(), "r");
			if (V_dir_solar_radiation == NULL) {
				printf("unknown file - evaporation.dat\n");
				system("pause");
				return 0;
			}

			fscanf(V_dir_solar_radiation, " albedo: %lf\n", &albedo);
			fscanf(V_dir_solar_radiation, " %s\n", head_solar_rad);
			fscanf(V_dir_temperature, " %s\n", head_temperature);

		    for (i = 0; i < (cont_rain - 1); i++){	
				//>>>>>>>>>>>>>>>>>>>> Input Temperatures data <<<<<<<<<<<<<<<<<
				fscanf(V_dir_temperature, " %lf\n", &h_T[i]);
				//>>>>>>>>>>>>>>>>>>>> Input Solar Radiation data <<<<<<<<<<<<<<
				fscanf(V_dir_solar_radiation, " %lf\n", &h_Rg[i]);
			}
			cudaMemcpy(d_T, h_T, (cont_rain - 1)*sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(d_Rg, h_Rg, (cont_rain - 1)*sizeof(double), cudaMemcpyHostToDevice);

			fclose(V_dir_temperature);
			fclose(V_dir_solar_radiation);

			//system("pause");

		}

		// *************************************************************
		//                    Input initial condition
		// *************************************************************

		initi_cond = "db\/" + dir_parameters + "\/SW2D\/input\/initial_condition.txt";
		FILE *V_initial_condition = fopen(initi_cond.c_str(), "r");
		if (V_initial_condition == NULL) {
			printf("unknown file - initial_condition.txt\n");
			system("pause");
			return 0;
		}
		h_initial_condition = (double*)malloc(N*sizeof(double));
		for (i = 0; i < N; i++){
			fscanf(V_initial_condition, "%lf\n", &h_initial_condition[i]);
		}
		fclose(V_initial_condition);


		// *************************************************************
		//                   output settings
		// *************************************************************
		lpout = round(dpout / dt);
		lkout = round(dkout / dt);

		// Save Results
		std::string dirRes_Output;
		FILE *WL_Output;
		std::string dirWaterLevel = "db\/" + dir_parameters + "\/SW2D\/output\/WaterLevel.txt";
		WL_Output = fopen(dirWaterLevel.c_str(), "w");
		if (out_outlet_on == 1){		    
		//    std::string dirWaterLevel = "db\/" + dir_parameters + "\/output\/WaterLevel.txt";
		//    WL_Output = fopen(dirWaterLevel.c_str(), "w");

		
			FILE *file_coord_out;
			std::string dirfile_coord_out = "db\/" + dir_parameters + "\/SW2D\/input\/coord_out.dat";
			file_coord_out = fopen(dirfile_coord_out.c_str(), "r");
			if (file_coord_out == NULL) {
				printf("unknown file - coord_out.dat\n");
				system("pause");
				return 0;
			}
			fscanf(file_coord_out, "n_out  %d/n ", &n_out);
			h_outx = (double*)malloc(n_out*sizeof(double));
			h_outy = (double*)malloc(n_out*sizeof(double));
			for (int in_out = 0; in_out < n_out; in_out++){
				fscanf(file_coord_out, " %lf %lf/n ", &h_outx[in_out], &h_outy[in_out]);
				h_outx[in_out] = h_outx[in_out] / resolution;
				h_outy[in_out] = h_outy[in_out] / resolution;
			}
			fclose(file_coord_out);
		}
		else{
		fprintf(WL_Output, "%s\n", "Outlet off");
		}
		//h_out = (double**)malloc((cont_rain - 1)*sizeof(double*));
		//for (int iu = 0; iu < cont_qq; iu++){
		//	h_out[iu] = (double*)malloc(n_out*sizeof(double));
		//}

		
		// *************************************************************
		//                  Time settings
		// *************************************************************
		time0 = 3600.0*24.0*tday0 + 3600.0*thour0 + 60.0*tmin0 + tsec0; // Initial time
		timmax = 3600.0*24.0*tday + 3600.0*thour + 60.0*tmin + tsec;    // Final time

		if ((tday == 0) & (thour == 0) & (tmin == 0) & (tsec == 0)){
			timmax = (cont_rain - 2)*dtrain;
		}
		
		// ***************************************************************
		//                  EFDC COUPLE
		// ***************************************************************
		
		int efdc_ntimes = ((timmax - time0) / dt)/(lkout); //
		int **north_bc;
		int **south_bc;
		int **east_bc;
		int **west_bc;
		int **LTS;
		int *nqsij, nqser, QSER_type,ius,vis;
		double **north_val;
		double **south_val;
		double **east_val;
		double **west_val;
		
		double efdc_time, efdc_basetime, TCON, TBEGIN, TREF, TCTMSR,\
		       ZBRADJ,ZBRCVRT,HMIN,HADJ,HCVRT,HDRY,HWET,BELADJ,BELCVRT;
			   
		int P2, NTC, NTSPTC, NTCPP, NTCVB, ISPPH, NPPPH, ISRPPH, IPPHXY,\
			ISVPH, NPVPH, ISRVPH, IVPHXY, ISTMSR, MLTMSR, NBTMSR, NSTMSR, NWTMSR, NTSSTSP,\
			nts_out, NTSSSS, MTSP, MTSC, MTSA, MTSUE, MTSUT, MTSU,\
			NSTYPE, NSHOWR, ISHOWC, JSHOWC, NSHFREQ, ZSSMIN, ZSSMAX, SSALMAX,efdc_count,\
			NWSER,NASER,N_processors,Lorp_type, ISGG;
			
		std::stringstream n_proc, c_name;

		int water_cells = 0;
		nqsij = (int*)malloc(1 * sizeof(int));		

		QSER_type = 0; // flows will be distributed in the vertical layers. If QSER_type = 1, Volume flows for a single layer.
		nqsij[0] = 0;
		nqser = 0;
		
		int pressure_OpenBC = 0;
		int volumeFlow_BC = 1;
		int nlayers = 10;
		
		
		if (efdc == 1){
			
			char ch;
			FILE *EFDC_template, *paste_EFDC_template;
			std::string dirfile_template = "db\/" + dir_parameters + "\/EFDC\/input\/template\/EFDC.INP";
			EFDC_template = fopen(dirfile_template.c_str(), "r");			
			std::string dirfile_paste_template = "db\/" + dir_parameters + "\/EFDC\/input\/EFDC.INP";
			paste_EFDC_template = fopen(dirfile_paste_template.c_str(), "w");				
			while ((ch = fgetc(EFDC_template)) != EOF){
                 fputc(ch, paste_EFDC_template);
            }			
			fclose(EFDC_template);
			fclose(paste_EFDC_template);
			
			efdc_basetime = 3600;  // seconds	
 
			TCON = efdc_basetime;  //CONVERSION MULTIPLIER TO CHANGE TBEGIN TO SECONDS 
			TBEGIN = 0.0;          //TIME ORIGIN OF RUN
			TREF = efdc_basetime;  //REFERENCE TIME PERIOD IN sec(i.e. 44714.16S OR 86400S)
			
			NTC = timmax/efdc_basetime;//NUMBER OF REFERENCE TIME PERIODS IN RUN
			NTSPTC = 2*efdc_basetime;//NUMBER OF TIME STEPS PER REFERENCE TIME PERIOD
			NTCPP = NTC;            //NUMBER OF REFERENCE TIME PERIODS BETWEEN FULL PRINTED OUTPUT TO FILE EFDC.OUT
			NTCVB = NTC;           //NUMBER OF REF TIME PERIODS WITH VARIABLE BUOYANCY FORCING
			
			//C11 GRID, ROUGHNESS AND DEPTH PARAMETERS
			ZBRADJ = 0.0;          //LOG BDRY LAYER CONST OR VARIABLE ROUGH HEIGHT ADJ IN METERS
			ZBRCVRT = 1.0;         //LOG BDRY LAYER VARIABLE ROUGHNESS HEIGHT CONVERT TO METERS
			HMIN = 0.0;            //MINIMUM DEPTH OF INPUTS DEPTHS IN METERS
			HADJ = 0.0;            //ADJUCTMENT TO DEPTH FIELD IN METERS
			HCVRT = 1.0;           //CONVERTS INPUT DEPTH FIELD TO METERS
			HDRY = 0.05;            //DEPTH AT WHICH CELL OR FLOW FACE BECOMES DRY
			HWET = 0.025;           //DEPTH AT WHICH CELL OR FLOW FACE BECOMES WET
			BELADJ = 0.00;          //ADJUCTMENT TO BOTTOM BED ELEVATION FIELD IN METERS
			BELCVRT = 1.0;         //CONVERTS INPUT BOTTOM BED ELEVATION FIELD TO METERS
			
			// C14 TIDAL & ATMOSPHERIC FORCING, GROUND WATER AND SUBGRID CHANNEL PARAMETERS
	        NWSER = 1;             //NUMBER OF WIND TIME SERIES (0 SETS WIND TO ZERO)  
			NASER = 1;             //NUMBER OF ATMOSPHERIC CONDITION TIME SERIES (0 SETS ALL  ZERO)

            //C72 CONTROLS FOR HORIZONTAL SURFACE ELEVATION OR PRESSURE CONTOURING
			ISPPH = 1;            //1 TO WRITE FILE FOR SURFACE ELEVATION OR PRESSURE CONTOURING
			                      //2 WRITE ONLY DURING LAST REFERENCE TIME PERIOD
			NPPPH = 1;            //NUMBER OF WRITES PER REFERENCE TIME PERIOD
			ISRPPH = 0;           //1 TO WRITE FILE FOR RESIDUAL SURFACE ELEVATION  CONTOURNG IN HORIZONTAL PLANE
			IPPHXY = 1;           //0 DOES NOT WRITE I,J,X,Y IN surfplt.out and rsurfplt.out FILES
			                      //1 WRITES I,J ONLY IN surfplt.out and rsurfplt.out FILES
			                      //2 WRITES I,J,X,Y  IN surfplt.out and rsurfplt.out FILES
			                      //3 WRITES EFDC EXPLORER BINARY FORMAT FILES

			//C73 CONTROLS FOR HORIZONTAL PLANE VELOCITY VECTOR PLOTTING
			ISVPH = 1;            //1 TO WRITE FILE FOR VELOCITY PLOTTING IN HORIZONTAL PLANE
			                      //2 WRITE ONLY DURING LAST REFERENCE TIME PERIOD
			NPVPH = 1;            //NUMBER OF WRITES PER REFERENCE TIME PERIOD
			ISRVPH = 0;           //1 TO WRITE FILE FOR RESIDUAL VELOCITY PLOTTIN IN HORIZONTAL PLANE
			IVPHXY = 1;           //0 DOES NOT WRITE I,J,X,Y IN velplth.out and rvelplth.out FILES
			                      //1 WRITES I,J ONLY IN velplth.out and rvelplth.out FILES
			                      //2 WRITES I,J,X,Y  IN velplth.out and rvelplth.out FILES
			                      //3 WRITES EFDC EXPLORER BINARY FORMAT FILES

			//C84 CONTROLS FOR WRITING TO TIME SERIES FILES
			ISTMSR = 1;           //1 OR 2 TO WRITE TIME SERIES OF SURF ELEV, VELOCITY, NET
				                  //INTERNAL AND EXTERNAL MODE VOLUME SOURCE - SINKS, AND
				                  //CONCENTRATION VARIABLES, 2 APPENDS EXISTING TIME SERIES FILES
			MLTMSR = 2;           //NUMBER HORIZONTAL LOCATIONS TO WRITE TIME SERIES OF SURF ELEV,
				                  //VELOCITY, AND CONCENTRATION VARIABLES
			NBTMSR = 1;           //TIME STEP TO BEGIN WRITING TO TIME SERIES FILES (Inactive)
			NSTMSR = NTC*NTSPTC;        //TIME STEP TO STOP WRITING TO TIME SERIES FILES (Inactive)
			NWTMSR = NTSPTC;      //NUMBER OF TIME STEPS TO SKIP BETWEEN OUTPUT
			NTSSTSP = 1;          //NUMBER OF TIME SERIES START-STOP SCENARIOS,  1 OR GREATER
			TCTMSR = 1.0;         //UNIT CONVERSION FOR TIME SERIES TIME.  FOR SECONDS, MINUTES,  
			                      //HOURS, DAYS USE 1.0, 60.0, 3600.0, 86400.0 RESPECTIVELY
								  
			
			// C87 CONTROLS FOR WRITING TO TIME SERIES FILES
			nts_out = MLTMSR;
			LTS = (int**)malloc((nts_out)*sizeof(int*));
			for (int iu = 0; iu < nts_out; iu++){
				LTS[iu] = (int*)malloc(2 * sizeof(int));
			}

			LTS[0][0] = 492;
			LTS[0][1] = 534;

			LTS[1][0] = 493;
			LTS[1][1] = 535;

			NTSSSS = 1;           // WRITE SCENARIO FOR THIS LOCATION
			MTSP = 1;             // 1 FOR TIME SERIES OF SURFACE ELEVATION
			MTSC = 1;             // 1 FOR TIME SERIES OF TRANSPORTED CONCENTRATION VARIABLES
			MTSA = 1;             // 1 FOR TIME SERIES OF EDDY VISCOSITY AND DIFFUSIVITY
			MTSUE = 1;            // 1 FOR TIME SERIES OF EXTERNAL MODE HORIZONTAL VELOCITY
			MTSUT = 0;            // 1 FOR TIME SERIES OF EXTERNAL MODE HORIZONTAL TRANSPORT
			MTSU = 1;             // 1 FOR TIME SERIES OF HORIZONTAL VELOCITY IN EVERY LAYER

            // gefdc.f - GRAPHICS GRID INFORMATION
            ISGG = 0;


			// SHOW.INP PARAMETERS
			NSTYPE = 3;           
			NSHOWR = NTSPTC;
			ISHOWC = 13; //492
			JSHOWC = 13; //534
			NSHFREQ = NTSPTC;
			ZSSMIN = -1;
			ZSSMAX = 13;
			SSALMAX = 1000;
			
			// EFDC DOMAIN DECOMPOSITION
			N_processors = 6;
			Lorp_type = 3 ;             // 1 type h
			                            // 2 type v
										// 3 type r
												
			dir_mask = "db\/" + dir_parameters + "\/SW2D\/input\/mask.txt";
			FILE *V_mask = fopen(dir_mask.c_str(), "r");
			if (V_mask == NULL) {
				printf("unknown file - mask.txt\n");
				system("pause");
				return 0;
			}
			fscanf(V_mask, " ncols %d\n", &cols);
			fscanf(V_mask, " nrows %d\n", &rows);
			fscanf(V_mask, " xllcorner %lf\n", &xcoor);
			fscanf(V_mask, " yllcorner %lf\n", &ycoor);
			fscanf(V_mask, " cellsize %lf\n", &resolution);
			fscanf(V_mask, " NODATA_value %lf\n", &NaN);
			//N = (rows)*(cols);
			h_mask = (int*)malloc(N*sizeof(int));

			for (i = 0; i < N; i++){
				fscanf(V_mask, "%d\n", &h_mask[i]);
				if (h_mask[i] == 1){
					water_cells = water_cells + 1;
				}
			}
			fclose(V_mask);

			north_bc = (int**)malloc((water_cells)*sizeof(int*));
			south_bc = (int**)malloc((water_cells)*sizeof(int*));
			east_bc = (int**)malloc((water_cells)*sizeof(int*));
			west_bc = (int**)malloc((water_cells)*sizeof(int*));

			for (int iu = 0; iu < water_cells; iu++){
				north_bc[iu] = (int*)malloc(3 * sizeof(int));
				south_bc[iu] = (int*)malloc(3 * sizeof(int));
				east_bc[iu] = (int*)malloc(3 * sizeof(int));
				west_bc[iu] = (int*)malloc(3 * sizeof(int));
			}			

			for (ius = 0; ius < water_cells; ius++){
				for (vis = 0; vis < 3; vis++){
					north_bc[ius][vis] = 0;
					south_bc[ius][vis] = 0;
					east_bc[ius][vis] = 0;
					west_bc[ius][vis] = 0;
				}
			}

			north_val = (double**)malloc((efdc_ntimes + 2)*sizeof(double*));
			south_val = (double**)malloc((efdc_ntimes + 2)*sizeof(double*));
			east_val = (double**)malloc((efdc_ntimes + 2)*sizeof(double*));
			west_val = (double**)malloc((efdc_ntimes + 2)*sizeof(double*));
			for (int iu = 0; iu < efdc_ntimes + 2; iu++){
				north_val[iu] = (double*)malloc((water_cells + 1) * sizeof(double));
				south_val[iu] = (double*)malloc((water_cells + 1) * sizeof(double));
				east_val[iu] = (double*)malloc((water_cells + 1) * sizeof(double));
				west_val[iu] = (double*)malloc((water_cells + 1) * sizeof(double));
			}

			for (ius = 0; ius < efdc_ntimes + 2; ius++){
				for (vis = 0; vis < (water_cells + 1); vis++){
					north_val[ius][vis] = 0.0;
					south_val[ius][vis] = 0.0;
					east_val[ius][vis] = 0.0;
					west_val[ius][vis] = 0.0;
				}
			}

			// Wind velocities

			dir_wind_velocities = "db\/" + dir_parameters + "\/SW2D\/input\/wind_velocities.txt";
			FILE *V_wind_velocities = fopen(dir_wind_velocities.c_str(), "r");
			if (V_wind_velocities == NULL) {
				printf("unknown file - wind_velocities.txt\n");
				system("pause");
				return 0;
			}
			h_wind_vel = (double**)malloc((efdc_ntimes+2)*sizeof(double*));
			for (int iu = 0; iu < (efdc_ntimes + 2); iu++){
				h_wind_vel[iu] = (double*)malloc(2*sizeof(double));
				
				fscanf(V_wind_velocities, " %lf %lf\n ", &h_wind_vel[iu][0], &h_wind_vel[iu][1]);

			}
			fclose(V_wind_velocities);

			std::string efdc_WSER_filepath = "db\/" + dir_parameters + "\/EFDC\/input\/WSER.INP";
			write_efdc_WSER(efdc_WSER_filepath.c_str(), h_wind_vel, dtrain, efdc_basetime, efdc_ntimes);


			// EFDC Meteorological data

			dir_meteo_data = "db\/" + dir_parameters + "\/SW2D\/input\/efdc_meteo_data.txt";

			FILE *V_dir_meteo_data = fopen(dir_meteo_data.c_str(), "r");

			if (V_dir_meteo_data == NULL) {
				printf("unknown file - efdc_meteo_data.txt\n");
				system("pause");
				return 0;
			}
			h_efdc_ASER = (double**)malloc((efdc_ntimes + 2)*sizeof(double*));
			for (int iu = 0; iu < (efdc_ntimes + 2); iu++){
				h_efdc_ASER[iu] = (double*)malloc(7 * sizeof(double));

				fscanf(V_dir_meteo_data, " %lf %lf %lf %lf %lf %lf %lf\n ", &h_efdc_ASER[iu][0], &h_efdc_ASER[iu][1], &h_efdc_ASER[iu][2], &h_efdc_ASER[iu][3], &h_efdc_ASER[iu][4], &h_efdc_ASER[iu][5], &h_efdc_ASER[iu][6]);

			}
			fclose(V_dir_meteo_data);		

			lake_catchment_smooth(cols, rows, h_mask, h_baseo);			
						
			std::string domain_filepath,bathymetry_filepath,grid_inputs_filepath,out_cell_filepath,\
			            out_gridext_filepath,out_depdat_filepath,out_gefdc_filepath;
			
			domain_filepath = "db\/" + dir_parameters + "\/SW2D\/input\/mask.txt";
			bathymetry_filepath="db\/" + dir_parameters + "\/SW2D\/input\/dem.txt";
			grid_inputs_filepath="db\/" + dir_parameters + "\/SW2D\/input\/grid_make_input.txt";
			
			//out_cell_filepath="db\/" + dir_parameters + "\/EFDC\/input\/CELL.INP";
			//out_gridext_filepath="db\/" + dir_parameters + "\/EFDC\/input\/gefdc\/gridext.inp";
			//out_depdat_filepath="db\/" + dir_parameters + "\/EFDC\/input\/gefdc\/depdat.inp";
			//out_gefdc_filepath="db\/" + dir_parameters + "\/EFDC\/input\/gefdc\/gefdc.inp";
			
			//make_gefdc_inputs(domain_filepath.c_str(),bathymetry_filepath.c_str(),\
                            grid_inputs_filepath.c_str(),out_cell_filepath.c_str(),\
							out_gridext_filepath.c_str(),out_depdat_filepath.c_str(),\
						    out_gefdc_filepath.c_str());	
							
							
            //make_gefdc_inputs(domain_filepath.c_str(),bathymetry_filepath.c_str(),\
                            grid_inputs_filepath.c_str());		
							
			make_gefdc_inputs(domain_filepath.c_str(),grid_inputs_filepath.c_str(),h_baseo);
							
            
			// Build EFDC EXE
			std::string dirfile_input_EFDCexe = "db\/" + dir_parameters + "\/EFDC\/input\/EFDC";
			rename("EFDC_src/EFDC", dirfile_input_EFDCexe.c_str());			
			
			// RUN Gorp (domain decomposition)			
						
            n_proc << N_processors;	             // NUMBER OF PROCESSORS EFDC DOMAIN DECOMPOSITION
			c_name << "sc";
			
			std::string gorpparam = ".\/Gorp CELL.INP " + c_name.str() + " " + n_proc.str()+"";
			int Gorp_RUN = system(gorpparam.c_str());	
			
			//=========================================
			
			if(Lorp_type == 1){
				std::string LORP_name1 = "LORP_" + c_name.str() + "_" +n_proc.str()+ "_h.INP";
				std::string dirfile_input_LORP1 = "db\/" + dir_parameters + "\/EFDC\/input\/LORP.INP";
				rename(LORP_name1.c_str(), dirfile_input_LORP1.c_str());				
			}		
			else if(Lorp_type == 2){
				std::string LORP_name2 = "LORP_" + c_name.str() + "_" +n_proc.str()+ "_v.INP";
				std::string dirfile_input_LORP2 = "db\/" + dir_parameters + "\/EFDC\/input\/LORP.INP";
				rename(LORP_name2.c_str(), dirfile_input_LORP2.c_str());				
			}
			else{
				std::string LORP_name3 = "LORP_" + c_name.str() + "_" +n_proc.str()+ "_r.INP";
				std::string dirfile_input_LORP3 = "db\/" + dir_parameters + "\/EFDC\/input\/LORP.INP";
				rename(LORP_name3.c_str(), dirfile_input_LORP3.c_str());				
			}			
			//n_proc.str("");
			c_name.str("");		

            
            if (ISGG == 1){				
			// Copy CELL.INP to gcell.inp 			
				char buf_cell;
				FILE *copy_cell, *paste_gcell;
				
				std::string dir_copy_cell = "CELL.INP";
				copy_cell = fopen(dir_copy_cell.c_str(), "r");	
				
				std::string dir_paste_gcell = "gcell.inp";
				paste_gcell = fopen(dir_paste_gcell.c_str(), "w");			
			
				while ((buf_cell = fgetc(copy_cell)) != EOF){				
					fputc(buf_cell, paste_gcell);				 
				}			
				fclose(copy_cell);							
				fclose(paste_gcell);				
			}
			
						
			// RUN Grid_efdc
			int F_RUN = system(".\/Grid_efdc");	
											
			
			// Move gefdc outputs to EFDC input folder	
            std::string dirfile_input_CELL = "db\/" + dir_parameters + "\/EFDC\/input\/CELL.INP";
            std::string dirfile_input_DXDY = "db\/" + dir_parameters + "\/EFDC\/input\/DXDY.INP";
            std::string dirfile_input_LXLY = "db\/" + dir_parameters + "\/EFDC\/input\/LXLY.INP";           
					
			rename("CELL.INP", dirfile_input_CELL.c_str());
			rename("DXDY.INP", dirfile_input_DXDY.c_str());
			rename("LXLY.INP", dirfile_input_LXLY.c_str());
			
			if (ISGG == 1){				
				std::string dirfile_input_GCELLMAP = "db\/" + dir_parameters + "\/EFDC\/input\/GCELLMP.INP";
				rename("gcellmap.out", dirfile_input_GCELLMAP.c_str());
			}
			
			int RM1 = system("rm -rf grid.ixy");
	        int RM2 = system("rm -rf grid.jxy");
	        int RM3 = system("rm -rf grid.cord");
	        int RM4 = system("rm -rf grid.dxf");
	        int RM5 = system("rm -rf grid.init");
	        int RM6 = system("rm -rf grid.mask");
	        int RM7 = system("rm -rf gridext.out");
	        int RM8 = system("rm -rf init.dxf");
	        int RM9 = system("rm -rf gefdc.out");
			
			int RM10 = system("rm -rf salt.inp");
			int RM11 = system("rm -rf gridext.inp");
			int RM12 = system("rm -rf gefdc.inp");
			int RM13 = system("rm -rf dxdy.diag");
			int RM14 = system("rm -rf depspc.out");
			int RM16 = system("rm -rf depdat.inp");
			int RM17 = system("rm -rf data.plt");
			int RM18 = system("rm -rf gcell.inp");
			
		    int RMlog = system("rm -rf *.log*");
			int RMtxt = system("rm -rf *.txt*");
			int RMinp = system("rm -rf *.INP*");
			
		    // Copy CELL.INP to CELLLT.INP 
			
			char ch_cell;
			FILE *copy_EFDC_cell, *paste_EFDC_celllt;
			std::string dirfile_cell = "db\/" + dir_parameters + "\/EFDC\/input\/CELL.INP";
			copy_EFDC_cell = fopen(dirfile_cell.c_str(), "r");				
            std::string dirfile_paste_celllt = "db\/" + dir_parameters + "\/EFDC\/input\/CELLLT.INP";
			paste_EFDC_celllt = fopen(dirfile_paste_celllt.c_str(), "w");			
			
			while ((ch_cell = fgetc(copy_EFDC_cell)) != EOF){				
				 fputc(ch_cell, paste_EFDC_celllt);				 
            }			
			fclose(copy_EFDC_cell);							
			fclose(paste_EFDC_celllt);
			
			// EFDC Ininialization DYE, SALT and TEMP
			double DYE_INIT = 0.00;
			double SALT_INIT = 25.00;
			
 			std::string efdc_DYE_INP = "db\/" + dir_parameters + "\/EFDC\/input\/DYE.INP";
			write_DYE_INP(efdc_DYE_INP.c_str(), cols, rows, nlayers,water_cells, h_mask, DYE_INIT);
			
			std::string efdc_SALT_INP = "db\/" + dir_parameters + "\/EFDC\/input\/SALT.INP";
			write_SALT_INP(efdc_SALT_INP.c_str(), cols, rows, nlayers,water_cells, h_mask, SALT_INIT);

			std::string efdc_TEMP_INP = "db\/" + dir_parameters + "\/EFDC\/input\/TEMP.INP";
			write_TEMP_INP(efdc_TEMP_INP.c_str(),cols,rows,nlayers,water_cells, h_mask, h_initial_condition,5.00, 28.00);
			
		}				
		
		
		// ***************************************************************
		// ***************************************************************
		printf("%s\n"," ******************************************************************** ");
		printf("%s\n"," Two-dimensional shallow water model accelerated by GPGPU (SW2D-GPU)  ");
		printf("%s\n"," ******************************************************************** ");
		printf("%s\n"," Month/Year - 11/2020 ");
		printf("%s\n"," Developer of parallel code in GPGPU: ");
		printf("%s\n","     Tomas Carlotto         |   Code written in CUDA C/C++ ");
		printf("%s\n"," ******************************************************************** ");
		printf("%s\n"," ******************************************************************** ");
		printf("%s\n", dirfile);
		// *************************************************************
		//              Memory Allocation - CPU
		// *************************************************************

		h_inf = (int*)malloc(N*sizeof(int));
		h_infx = (int*)malloc(N*sizeof(int));
		h_infy = (int*)malloc(N*sizeof(int));
		h_infsw = (int*)malloc(N*sizeof(int));

		h_h = (double*)malloc(N*sizeof(double));
		h_ho = (double*)malloc(N*sizeof(double));
		h_hm = (double*)malloc(N*sizeof(double));
		h_hn = (double*)malloc(N*sizeof(double));

		h_um = (double*)malloc(N*sizeof(double));
		h_umo = (double*)malloc(N*sizeof(double));
		h_uu = (double*)malloc(N*sizeof(double));
		h_uua = (double*)malloc(N*sizeof(double));
		h_uu1 = (double*)malloc(N*sizeof(double));

		h_vn = (double*)malloc(N*sizeof(double));
		h_vno = (double*)malloc(N*sizeof(double));
		h_vv = (double*)malloc(N*sizeof(double));
		h_vva = (double*)malloc(N*sizeof(double));
		h_vv1 = (double*)malloc(N*sizeof(double));

		h_ql = (double*)malloc(N*sizeof(double));
		h_rr = (double*)malloc(sizeof(double));
		h_th = (double*)malloc(sizeof(double));

		h_th[0] = 1.0e-4;                          

		int km = -1;
		for (int im = 0; im <= rows - 1; im++) {
			for (int jm = 0; jm <= cols - 1; jm++) {
				km = km + 1;
				
				h_um[km] = 0.00;
				h_umo[km] = 0.00;
				h_uu[km] = 0.00;
				h_uua[km] = 0.00;
				h_uu1[km] = 0.00;

				h_vn[km] = 0.00;
				h_vno[km] = 0.00;
				h_vv[km] = 0.00;
				h_vva[km] = 0.00;
				h_vv1[km] = 0.00;

				h_ql[km] = 0.000;

				h_h[km] = h_initial_condition[km];
				
				h_ho[km] = 0.00;
				h_hm[km] = 0.00;
				h_hn[km] = 0.00;

			}
		}

		// *************************************************************
		//                   Memory Allocation - GPU
		// *************************************************************
		cudaMalloc((void**)&d_inf, N*sizeof(int));
		cudaMalloc((void**)&d_infx, N*sizeof(int));
		cudaMalloc((void**)&d_infy, N*sizeof(int));
		cudaMalloc((void**)&d_infsw, N*sizeof(int));

		cudaMalloc((void**)&d_h, N*sizeof(double));
		cudaMalloc((void**)&d_ho, N*sizeof(double));
		cudaMalloc((void**)&d_hm, N*sizeof(double));
		cudaMalloc((void**)&d_hn, N*sizeof(double));

		cudaMalloc((void**)&d_um, N*sizeof(double));
		cudaMalloc((void**)&d_umo, N*sizeof(double));
		cudaMalloc((void**)&d_uu, N*sizeof(double));
		cudaMalloc((void**)&d_uua, N*sizeof(double));
		cudaMalloc((void**)&d_uu1, N*sizeof(double));

		cudaMalloc((void**)&d_vn, N*sizeof(double));
		cudaMalloc((void**)&d_vno, N*sizeof(double));
		cudaMalloc((void**)&d_vv, N*sizeof(double));
		cudaMalloc((void**)&d_vva, N*sizeof(double));
		cudaMalloc((void**)&d_vv1, N*sizeof(double));

		cudaMalloc((void**)&d_baseo, N*sizeof(double));
		//cudaMalloc((void**)&d_qq, (cont_qq)*sizeof(double));
		//cudaMalloc((void**)&d_rain, (cont_rain - 1)*sizeof(double));
		cudaMalloc((void**)&d_ql, N*sizeof(double));
		cudaMalloc((void**)&d_rr, sizeof(double));
		cudaMalloc((void**)&d_th, sizeof(double));
		//cudaMalloc((void**)&d_outlet, N*sizeof(int));
		//cudaMemcpy(d_outlet, h_outlet, N*sizeof(int), cudaMemcpyHostToDevice);
		
	    cudaMemcpy(d_um, h_um, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_umo, h_umo, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_uu, h_uu, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_uua, h_uua, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_uu1, h_uu1, N*sizeof(double), cudaMemcpyHostToDevice);

		cudaMemcpy(d_vn, h_vn, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_vno, h_vno, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_vv, h_vv, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_vva, h_vva, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_vv1, h_vv1, N*sizeof(double), cudaMemcpyHostToDevice);
		
		cudaMemcpy(d_th, h_th, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_h, h_h, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_hn, h_hn, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_ho, h_ho, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_hm, h_hm, N*sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_infx, h_infx, N*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_infy, h_infy, N*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_baseo, h_baseo, N*sizeof(double), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_qq, h_qq, (cont_qq)*sizeof(double), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_rain, h_rain, (cont_rain - 1)*sizeof(double), cudaMemcpyHostToDevice);

		// *******************************************************************
		// Definition of the number of blocks and threads for mesh (N cells)
		// *******************************************************************

		cudaDeviceProp prop;
		int count;	cudaGetDeviceCount(&count);
		for (int i = 0; i < count; i++){
			cudaGetDeviceProperties(&prop, i);
			maxThreadsPerBlock = prop.maxThreadsPerBlock;
		}
		if (N < maxThreadsPerBlock){
			threadsPerBlock = N;
			numBlocks = (N + N - 1) / N;
		}
		else{
			threadsPerBlock = maxThreadsPerBlock;
			numBlocks = (N + round(maxThreadsPerBlock) - 1) / round(maxThreadsPerBlock);
		}
		
		double dx = resolution;
		double dy = resolution;
		
		init_inf << < numBlocks, threadsPerBlock >> >(rows, cols, d_ho, d_h, d_inf, d_baseo, N, NaN);
		cudaDeviceSynchronize();
		initiald << < numBlocks, threadsPerBlock >> >(rows, cols, d_h, d_infx, d_infy, d_inf, d_hm, d_hn, d_baseo, N, NaN);
		cudaDeviceSynchronize();		
		if (evaporation_on == 1){
			gpu_evaporation_calc << <numBlocks, threadsPerBlock >> >(albedo, d_T, d_Rg, d_Rs, d_pw, d_lv, d_Evapo, dtrain, (cont_rain-1));
			cudaMemcpy(h_Evapo, d_Evapo, (cont_rain - 1)*sizeof(double), cudaMemcpyDeviceToHost);
			
			if (efdc==1){
				// efdc_Metorological_vars from SW2D-GPU
				int iev = 0;
				for (iev = 0; iev < (efdc_ntimes+2); iev++){
					h_efdc_ASER[iev][1] = h_T[iev];
					h_efdc_ASER[iev][3] = (1-LWL)*h_rain[iev];
					h_efdc_ASER[iev][4] = h_Evapo[iev];
					h_efdc_ASER[iev][5] = h_Rg[iev];
				}
				std::string efdc_ASER_filepath = "db\/" + dir_parameters + "\/EFDC\/input\/ASER.INP";
				write_efdc_ASER(efdc_ASER_filepath.c_str(), h_efdc_ASER, dtrain, efdc_basetime, efdc_ntimes);				
			}
			
		}				
			
		double time = time0;
		int mstep = 0;

		//Start time
		std::clock_t start;
		start = std::clock();

		int tq = -1;
		int out0 = -1;
		while (time + dt <= timmax){

			out0 = out0 + 1;

			if (mstep % lpout == 0){
				printf(" %d\n", int(time));
			}

			// ************************************************
			//           2D FLOW CALCULATION
			// ************************************************

			flux << < numBlocks, threadsPerBlock >> >(d_th, gg, manning_coef, d_inf, d_h, d_infx, d_infy, d_baseo, d_um, d_hm, d_uu1, \
			                                       	  d_umo, d_vv1, d_vva, d_vn, d_hn, d_vno, d_uua, d_ho, N, \
				                                      cols, rows, dx, dy, dt2);

			cudaDeviceSynchronize();

			// ************************************************
			//              CONTINUITY EQUATION
			// ************************************************
			
			stream_flow(cols, rows, (xcoor/resolution), (ycoor/resolution), time, dtrain, h_rain, h_qq, h_ql, dtoq, h_brx, h_bry, dx, dy, nst, h_rr);			
			cudaMemcpy(d_ql, h_ql, N*sizeof(double), cudaMemcpyHostToDevice);
			cudaMemcpy(d_rr, h_rr, sizeof(double), cudaMemcpyHostToDevice);
			
			if (evaporation_on == 1){
				evaporation_load(time, dtrain, h_Evapo, h_Ev);
				cudaMemcpy(d_Ev, h_Ev, sizeof(double), cudaMemcpyHostToDevice);
			}
			
			continuity << < numBlocks, threadsPerBlock >> >(dt2, cols,rows, dx, dy, d_rr, d_Ev, d_ql, d_h, d_ho, d_um, d_vn,INT,INF,LWL,EV_WL_min, d_inf, N);	
			cudaDeviceSynchronize();			

			// ************************************************
			//                 ERROR TREATMENT
			// ************************************************
			treat_error << < numBlocks, threadsPerBlock >> >(cols, rows, d_th, d_inf, d_um, d_vn, d_h, N);
			cudaDeviceSynchronize();
			// time step **************************************
			time = time + dt;
			mstep = mstep + 1;
			//*************************************************
			//              CONTINUITY EQUATION
			//*************************************************
			stream_flow(cols, rows, (xcoor/resolution), (ycoor/resolution), time, dtrain, h_rain, h_qq, h_ql, dtoq, h_brx, h_bry, dx, dy, nst, h_rr);

			cudaMemcpy(d_ql, h_ql, N*sizeof(double), cudaMemcpyHostToDevice); //h_rr[0] = 0; //if (time+dt<=1000){h_rr[0] = 0.005;}
			cudaMemcpy(d_rr, h_rr, sizeof(double), cudaMemcpyHostToDevice);
			
			if (evaporation_on == 1){
				evaporation_load(time, dtrain, h_Evapo, h_Ev);
				cudaMemcpy(d_Ev, h_Ev, sizeof(double), cudaMemcpyHostToDevice);
			}
			
			//system("pause");
			continuity << < numBlocks, threadsPerBlock >> >(dt2, cols,rows, dx, dy, d_rr, d_Ev, d_ql, d_h, d_ho, d_um, d_vn,INT,INF,LWL,EV_WL_min, d_inf, N);
			cudaDeviceSynchronize();

			//*************************************************
			//           PREPARING NEXT CALCULATION
			//*************************************************
			hm_hn << < numBlocks, threadsPerBlock >> >(d_hm, d_hn, d_h, N, cols, rows);
			cudaDeviceSynchronize();
			uu1_vv1 << < numBlocks, threadsPerBlock >> >(d_th, d_hm, d_hn, d_uu1, d_um, d_vv1, d_vn, N, cols, rows);
			cudaDeviceSynchronize();
			uu_vv << < numBlocks, threadsPerBlock >> >(d_th, d_h, d_uu1, d_vv1, d_uu, d_vv, N, cols);
			cudaDeviceSynchronize();
			uua_vva << < numBlocks, threadsPerBlock >> >(d_uu1, d_vv1, d_uua, d_vva, N, cols, rows);
			cudaDeviceSynchronize();

			//**************************************************
			//                   FORWARD
			//**************************************************
			forward << < numBlocks, threadsPerBlock >> >(cols, rows, d_umo, d_um, d_vno, d_vn, d_ho, d_h, N);
			cudaDeviceSynchronize();
			//**************************************************

			time = time + dt;
			mstep = mstep + 1;

			//output
			if ((mstep % (lkout) == 0) || (out0 == 0)){
				tq = tq + 1;

				if (out0 > 0){
					out << round(mstep*dt / dkout);
					tempo = out.str();
					out.str("");
				}
				else{
					out << 0;
					tempo = out.str();
					out.str("");
				}
								
                if (efdc == 1){		

					efdc_time = mstep*dt / efdc_basetime;
					//printf("%lf\n", efdc_time);

					efdc_count = efdc_count + 1;
					cudaMemcpy(h_h, d_h, N*sizeof(double), cudaMemcpyDeviceToHost);

					if (tq==0){//(out0==0){
						couple_BC_efdc(pressure_OpenBC, volumeFlow_BC, cols, rows, dx, dy, nqsij, north_bc, south_bc, east_bc, west_bc, north_val, south_val, east_val, west_val, h_mask, h_initial_condition, h_vva, h_uua, tq, 0.0000);
						nqser = nqsij[0];
						std::string efdc_QSER_filepath = "db\/" + dir_parameters + "\/EFDC\/input\/QSER.INP";
						write_efdc_QSER(efdc_QSER_filepath.c_str(), QSER_type, nlayers, north_bc, south_bc, east_bc, west_bc, north_val, south_val, east_val, west_val, efdc_basetime, tq);
					}
					else{
						couple_BC_efdc(pressure_OpenBC, volumeFlow_BC, cols, rows, dx, dy, nqsij, north_bc, south_bc, east_bc, west_bc, north_val, south_val, east_val, west_val, h_mask, h_h, h_vva, h_uua, tq , efdc_time);
						nqser = nqsij[0];
					}

					if (volumeFlow_BC == 1){
						cudaMemcpy(h_uua, d_uua, N*sizeof(double), cudaMemcpyDeviceToHost);
						cudaMemcpy(h_vva, d_vva, N*sizeof(double), cudaMemcpyDeviceToHost);
					}	

					
					//couple_BC_efdc(pressure_OpenBC, volumeFlow_BC, cols, rows, dx, dy, nqsij, north_bc, south_bc, east_bc, west_bc, north_val, south_val, east_val, west_val, h_mask, h_h, h_vn, h_um, tq+1, efdc_time);
					//nqser = nqsij[0];

					//}

					if (pressure_OpenBC == 1){
						int bn, bs, be, bw, ig, ig1, ef;

						if (tq == 0){							

							FILE *north_bc_efdcOut;
							std::string efdc_north_bc = "db\/" + dir_parameters + "\/SW2D\/output\/efdc_north_bc.txt";
							north_bc_efdcOut = fopen(efdc_north_bc.c_str(), "w");

							FILE *south_bc_efdcOut;
							std::string efdc_south_bc = "db\/" + dir_parameters + "\/SW2D\/output\/efdc_south_bc.txt";
							south_bc_efdcOut = fopen(efdc_south_bc.c_str(), "w");

							FILE *east_bc_efdcOut;
							std::string efdc_east_bc = "db\/" + dir_parameters + "\/SW2D\/output\/efdc_east_bc.txt";
							east_bc_efdcOut = fopen(efdc_east_bc.c_str(), "w");

							FILE *west_bc_efdcOut;
							std::string efdc_west_bc = "db\/" + dir_parameters + "\/SW2D\/output\/efdc_west_bc.txt";
							west_bc_efdcOut = fopen(efdc_west_bc.c_str(), "w");

							int npsers = 0;
							bn = 0;
							while (north_bc[bn][0] != 0){
								npsers = npsers + 1;
								fprintf(north_bc_efdcOut, "%d ", north_bc[bn][0]);
								fprintf(north_bc_efdcOut, "%d ", north_bc[bn][1]);
								fprintf(north_bc_efdcOut, "%d ", 0);
								fprintf(north_bc_efdcOut, "%d ", 0);
								fprintf(north_bc_efdcOut, "%d\n", npsers);
								bn = bn + 1;
							}
							bs = 0;
							while (south_bc[bs][0] != 0){
								npsers = npsers + 1;
								fprintf(south_bc_efdcOut, "%d ", south_bc[bs][0]);
								fprintf(south_bc_efdcOut, "%d ", south_bc[bs][1]);
								fprintf(south_bc_efdcOut, "%d ", 0);
								fprintf(south_bc_efdcOut, "%d ", 0);
								fprintf(south_bc_efdcOut, "%d\n", npsers);
								bs = bs + 1;
							}
							be = 0;
							while (east_bc[be][0] != 0){
								npsers = npsers + 1;
								fprintf(east_bc_efdcOut, "%d ", east_bc[be][0]);
								fprintf(east_bc_efdcOut, "%d ", east_bc[be][1]);
								fprintf(east_bc_efdcOut, "%d ", 0);
								fprintf(east_bc_efdcOut, "%d ", 0);
								fprintf(east_bc_efdcOut, "%d\n", npsers);
								be = be + 1;
							}
							bw = 0;
							while (west_bc[bw][0] != 0){
								npsers = npsers + 1;
								fprintf(west_bc_efdcOut, "%d ", west_bc[bw][0]);
								fprintf(west_bc_efdcOut, "%d ", west_bc[bw][1]);
								fprintf(west_bc_efdcOut, "%d ", 0);
								fprintf(west_bc_efdcOut, "%d ", 0);
								fprintf(west_bc_efdcOut, "%d\n", npsers);
								bw = bw + 1;
							}

							fclose(north_bc_efdcOut);
							fclose(south_bc_efdcOut);
							fclose(east_bc_efdcOut);
							fclose(west_bc_efdcOut);
						}

						FILE *wl_efdcOut;
						std::string efdc_wl = "db\/" + dir_parameters + "\/SW2D\/output\/efdc_wl.txt";
						wl_efdcOut = fopen(efdc_wl.c_str(), "w");
						int edg = 0;
						// North
						for (ig = 0; ig < north_bc[bn - 1][2]; ig++){
							edg = edg + 1;
							fprintf(wl_efdcOut, "%d %lf %d %d %d %s%d\n", tq + 1, efdc_basetime, 0, 1, 0, " ' *** North_edge_", edg);

							for (ig1 = 0; ig1 < tq + 1; ig1++){
								fprintf(wl_efdcOut, "%lf ", (north_val[ig1][0] - north_val[0][0]));
								fprintf(wl_efdcOut, "%lf\n", north_val[ig1][ig + 1]);
							}

						}

						// South
						for (ig = 0; ig < south_bc[bs - 1][2]; ig++){
							edg = edg + 1;
							fprintf(wl_efdcOut, "%d %lf %d %d %d %s%d\n", tq + 1, efdc_basetime, 0, 1, 0, " ' *** South_edge_", edg);

							for (ig1 = 0; ig1 < tq + 1; ig1++){
								fprintf(wl_efdcOut, "%lf ", (south_val[ig1][0] - south_val[0][0]));
								fprintf(wl_efdcOut, "%lf\n", south_val[ig1][ig + 1]);
							}

						}

						// East
						for (ig = 0; ig < east_bc[be - 1][2]; ig++){
							edg = edg + 1;
							fprintf(wl_efdcOut, "%d %lf %d %d %d %s%d\n", tq + 1, efdc_basetime, 0, 1, 0, " ' *** East_edge_", edg);

							for (ig1 = 0; ig1 < tq + 1; ig1++){
								fprintf(wl_efdcOut, "%lf ", (east_val[ig1][0] - east_val[0][0]));
								fprintf(wl_efdcOut, "%lf\n", east_val[ig1][ig + 1]);
							}

						}

						// West
						for (ig = 0; ig < west_bc[bw - 1][2]; ig++){
							edg = edg + 1;
							fprintf(wl_efdcOut, "%d %lf %d %d %d %s%d\n", tq + 1, efdc_basetime, 0, 1, 0, " ' *** West_edge_", edg);

							for (ig1 = 0; ig1 < tq + 1; ig1++){
								fprintf(wl_efdcOut, "%lf ", (west_val[ig1][0] - west_val[0][0]));
								fprintf(wl_efdcOut, "%lf\n", west_val[ig1][ig + 1]);
							}

						}

						fclose(wl_efdcOut);
					}
					else{
						// Volumetric BC: EFDC.INP write
						if (tq == 0){
							std::string efdc_filepath = "db\/" + dir_parameters + "\/EFDC\/input\/EFDC.INP";
							write_efdc_CONFIG(efdc_filepath.c_str(), nqsij, nqser, north_bc, south_bc, east_bc, west_bc, nlayers, \
								cols, rows, water_cells, TCON, TBEGIN, TREF, NTC, NTSPTC, NTCPP, NTCVB,\
								ISPPH, NPPPH, ISRPPH, IPPHXY, ISVPH, NPVPH, ISRVPH, IVPHXY,\
								ISTMSR,MLTMSR,NBTMSR,NSTMSR,NWTMSR,NTSSTSP,TCTMSR,\
								nts_out, LTS, NTSSSS, MTSP, MTSC, MTSA, MTSUE, MTSUT, MTSU,\
								ZBRADJ,ZBRCVRT,HMIN,HADJ,HCVRT,HDRY,HWET,BELADJ,BELCVRT,\
								NWSER,NASER,dx,dy);

							// SHOW.INP write
							std::string efdc_SHOW_filepath = "db\/" + dir_parameters + "\/EFDC\/input\/SHOW.INP";
							write_efdc_SHOW(efdc_SHOW_filepath.c_str(),NSTYPE,NSHOWR,ISHOWC,JSHOWC,NSHFREQ,ZSSMIN,ZSSMAX,SSALMAX);
						}
						else{
							std::string efdc_QSER_filepath = "db\/" + dir_parameters + "\/EFDC\/input\/QSER.INP";
							write_efdc_QSER(efdc_QSER_filepath.c_str(), QSER_type, nlayers, north_bc, south_bc, east_bc, west_bc, north_val, south_val, east_val, west_val, efdc_basetime, tq );														
						}
						//
					}
					
					//if(tq == 3){
				    // RUN EFDC-MPI model
			        //    std::string EFDCrun = "mpirun -wdir db\/" + dir_parameters + "\/EFDC\/input -np " + n_proc.str() + " ./EFDC &";
			        //   int efdc_RUN = system(EFDCrun.c_str());
		            //    n_proc.str("");	
						
					//	int P_2 = system("P2 = $!"); // ID EFDC-MPI process 
						
					//}
					
					
				}
				
				FILE *Res_Output;
				dirRes_Output = "db\/" + dir_parameters + "\/SW2D\/output\/Results_" + tempo + ".vtk";
				Res_Output = fopen(dirRes_Output.c_str(), "w");				

				// output .vtk format

				fprintf(Res_Output, "%s\n", "# vtk DataFile Version 2.0");
				fprintf(Res_Output, "%s\n", "Brazil");
				fprintf(Res_Output, "%s\n", "ASCII");
				fprintf(Res_Output, "%s\n", "DATASET STRUCTURED_POINTS");
				fprintf(Res_Output, "DIMENSIONS %d %d %d\n", cols, rows, 1);
				fprintf(Res_Output, "ASPECT_RATIO %lf %lf %lf\n", dx, dy, 1.0000);
				fprintf(Res_Output, "ORIGIN %lf %lf %lf\n", xcoor, ycoor, 0.000);
				fprintf(Res_Output, "POINT_DATA %d\n", cols*rows);
				int posxy;
				int npout = 0;
				//                 Water depth
				if (out_depth == 1){
					cudaMemcpy(h_h, d_h, N*sizeof(double), cudaMemcpyDeviceToHost);
					fprintf(Res_Output, "%s\n", "SCALARS depth float 1");
					fprintf(Res_Output, "%s\n", "LOOKUP_TABLE default");
					int km = -1;					
					for (int im = 0; im <= rows - 1; im++) {
						posxy = rows*cols - (im + 1)*cols;
						posxy=posxy-1;
						for (int jm = 0; jm <= cols - 1; jm++) {
							km = km + 1;
							posxy = posxy + 1;
							if (out_outlet_on==1){
								//***********************************************								
								for (int i = 0; i < n_out; i++){					
									outx = round(abs((xcoor/resolution) - h_outx[i]));
									outy = rows - round(abs((ycoor/resolution) - h_outy[i]));

									if (km == ((outy)*cols - (cols - (outx + 1)))){
										npout = npout + 1;
										if (out0 == 0){
											if (npout == n_out){												
												fprintf(WL_Output, " %lf\n", h_initial_condition[km]);
											}
											else{												
												fprintf(WL_Output, " %lf", h_initial_condition[km]);
											}
										}
										else{
											if (npout == n_out){												
												fprintf(WL_Output, " %lf\n", h_h[km]);
											}
											else{												
												fprintf(WL_Output, " %lf", h_h[km]);
											}
										}
									}
								}
								//***********************************************
							}

							if (out0 == 0){
								fprintf(Res_Output, "%f\n", h_initial_condition[posxy]);
							}
							else{
								fprintf(Res_Output, "%f\n", h_h[posxy]);
							}
						}
					}
				}
					//                Velocity x direction
				if (out_velocity_x == 1){
				    cudaMemcpy(h_vv, d_vv, N*sizeof(double), cudaMemcpyDeviceToHost);
					fprintf(Res_Output, "%s\n", "SCALARS x_velocity float 1");
					fprintf(Res_Output, "%s\n", "LOOKUP_TABLE default");
					km = -1;
					for (int im = 0; im <= rows - 1; im++) {
						posxy = rows*cols - (im + 1)*cols;
						posxy = posxy-1;
						for (int jm = 0; jm <= cols - 1; jm++) {
							km = km + 1;
							posxy = posxy + 1;
							fprintf(Res_Output, "%f\n", h_vv[posxy]);
						}
					}
				}
					//                Velocity y direction
				if (out_velocity_y == 1){
				    cudaMemcpy(h_uu, d_uu, N*sizeof(double), cudaMemcpyDeviceToHost);
					fprintf(Res_Output, "%s\n", "SCALARS y_velocity float 1");
					fprintf(Res_Output, "%s\n", "LOOKUP_TABLE default");
					km = -1;
					for (int im = 0; im <= rows - 1; im++) {
						posxy = rows*cols - (im + 1)*cols;
						posxy = posxy-1;
						for (int jm = 0; jm <= cols - 1; jm++) {
							km = km + 1;
							posxy = posxy + 1;
							fprintf(Res_Output, "%f\n", -h_uu[posxy]);
						}
					}
				}
					//                 Elevations
				if (out_elevation == 1){
				    cudaMemcpy(h_baseo, d_baseo, N*sizeof(double), cudaMemcpyDeviceToHost);
					fprintf(Res_Output, "%s\n", "SCALARS Elevations float 1");
					fprintf(Res_Output, "%s\n", "LOOKUP_TABLE default");
					km = -1;
					for (int im = 0; im <= rows - 1; im++) {
						posxy = rows*cols - (im + 1)*cols;
						posxy = posxy-1;
						for (int jm = 0; jm <= cols - 1; jm++) {
							km = km + 1;
							posxy = posxy + 1;
							fprintf(Res_Output, "%f\n", h_baseo[posxy]);
						}
					}
				}
				fclose(Res_Output);						
			}
		}
        
		if (efdc==1){
			
		// DYE time series			
			std::string efdc_DSER_filepath = "db\/" + dir_parameters + "\/EFDC\/input\/DSER.INP";
			write_efdc_DSER(efdc_DSER_filepath.c_str(), 0, nlayers, north_bc, south_bc, east_bc, west_bc,north_val,10.0, efdc_basetime,tq);
							
		// Temperature time series	
			std::string efdc_TSER_filepath = "db\/" + dir_parameters + "\/EFDC\/input\/TSER.INP";
			write_efdc_TSER(efdc_TSER_filepath.c_str(), 0, nlayers, north_bc, south_bc, east_bc, west_bc,north_val,h_T, efdc_basetime,tq);
		
		// Salinity time series
			std::string efdc_SSER_filepath = "db\/" + dir_parameters + "\/EFDC\/input\/SSER.INP";
			write_efdc_SSER(efdc_SSER_filepath.c_str(), 0, nlayers, north_bc, south_bc, east_bc, west_bc,north_val,15.0, efdc_basetime,tq);
			
		// RUN EFDC-MPI model
			std::string EFDCrun = "mpirun -wdir db\/" + dir_parameters + "\/EFDC\/input -np " + n_proc.str() + " ./EFDC";
			int efdc_RUN = system(EFDCrun.c_str());
			n_proc.str("");	
			
		}
				
		//if (efdc == 1){
			
		// int wb = system("wait $P2");
			
		//}
		
		//Time
		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		std::cout << "printf: " << duration << '\n';

		// Save Times
		FILE *TimeOutput;
		std::string dirTimeOutput = "db\/" + dir_parameters + "\/SW2D\/output\/TimeSimu_" + tempo + ".txt";
		TimeOutput = fopen(dirTimeOutput.c_str(), "w");
		fprintf(TimeOutput, " %lf\n", duration);
		fclose(TimeOutput);
		fclose(WL_Output);

		// Cleaning Up (GPU memory)
				
		if (evaporation_on == 1){
			cudaFree(d_Ev);
			cudaFree(d_T);
			cudaFree(d_Rg);
			cudaFree(d_Rs);
			cudaFree(d_pw);
			cudaFree(d_lv);
			cudaFree(d_Evapo);
		}
		
		cudaFree(d_inf);
		cudaFree(d_infx);
		cudaFree(d_infy);
		
		cudaFree(d_h);
		cudaFree(d_ho);
		cudaFree(d_hm);
		cudaFree(d_hn);
		cudaFree(d_um);
		cudaFree(d_umo);
		cudaFree(d_uu);
		cudaFree(d_uua);
		cudaFree(d_uu1);
		cudaFree(d_vn);
		cudaFree(d_vno);
		cudaFree(d_vv);
		cudaFree(d_vva);
		cudaFree(d_vv1);
		
		
		cudaFree(d_baseo);
		cudaFree(d_ql);
		
		//cudaFree(d_rr);		
		//cudaFree(d_th);
        
	}

	return 0;

}
