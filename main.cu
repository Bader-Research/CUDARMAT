#include <cuda.h>
#include <stdio.h>
#include <xmalloc.cuh>
#include <sys/time.h>
#include <stdint.h>
#include <getopt.h>
#include <errno.h>
#include <util.h>

#define DEBUG 0

//////////////////////////////
#define INITRANDMULT       0x015A4E35
#define INITRANDINCREMENT  997
#define RANDMULT           214013
#define RANDINCREMENT      2531011
#define TESTRAND           100
#define MAXBLOCKSIZE       512
#define EDGEBATCHSIZE		134217728
#define ACTIONBATCHSIZE		16777216

// Errors
//////////////////////////////
typedef enum  {
   NO_ERROR,
   DELETIONS_OVERRAN_INSERTIONS               
} graphError_t;

const char * graphErrorString[] = {
   "NO ERROR",
   "DELETIONS OVERRAN INSERTIONS - RARE RANDOM ERROR, RUN IT AGAIN."
};


// Device Function Prototypes
//////////////////////////////
__device__ uint32_t cudaRand(uint32_t * randVal);
__global__ void cudaRMATEdges(uint32_t * randVals, uint32_t SCALE, uint32_t edgesPerThread, uint32_t numthreads, 
                               float pA, float pB, float pC, float pD, uint32_t * edgeArray);
__global__ void cudaGenerateActions(uint32_t * randVals, uint32_t edges, uint32_t actions, uint32_t numthreads, 
                                    float pDelete, uint32_t * edgeArray, uint32_t * generatedEdges, int32_t * actionsEdgeArray, graphError_t * error); 

#if(DEBUG)
__global__ void cudaDebugRand(uint32_t * randOut, uint32_t * randVals);
__global__ void cudaDebugEdgeList(uint32_t vertices, uint32_t * edgeList, uint32_t size);
#endif

// Host Functions Prototypes
/////////////////////////////
__host__ void hostParseArgs(int argc, char** argv);
__host__ void hostInitCudaRand(); 
__host__ void hostRMATandFileIO();
__host__ int  hostCompareEdges(const void * a, const void * b);   
__host__ void hostFreeCudaRand();

#if (DEBUG)
__host__ void hostDebugTestRand();
#endif

// Global Device Variables
/////////////////////////////
uint32_t       * d_uip_randvals;
uint32_t       * d_uip_edgelist;
uint32_t			* h_uip_edgelist;
int32_t        * d_ip_actionslist;
int32_t        * h_ip_actionslist;
void           * d_vp_cudastinger;
void           * d_vp_cudavertices;
graphError_t   * d_gep_error;

// Global Host Variables
////////////////////////////
uint32_t h_ui_scale = 14;
uint32_t h_ui_edgefactor = 16;
uint32_t h_ui_actions = 512 * 512 * 10;
uint32_t h_ui_threads = 512;
uint32_t h_ui_blocks = 512;
uint32_t h_ui_vertices = 4096;
uint32_t h_ui_edges = 32768;
const char * h_s_infile = NULL;
const char * h_s_outfile = NULL;
const char * h_s_dimacsoutfile = NULL;
const char * h_s_stinger_outfile = NULL;
const char * h_s_stinger_actionsfile = NULL;
int h_i_streaming = 0;
                                                                                                  
// Host Functions
/////////////////////////////
__host__ int main(int argc, char** argv) {
   printf("CUDA CP2 Implementation\n");
   hostParseArgs(argc, argv);

   d_gep_error = (graphError_t *)cudaXmalloc(sizeof(graphError_t));
   graphError_t hosterror = NO_ERROR;
   cudaMemcpy(d_gep_error, &hosterror, sizeof(graphError_t), cudaMemcpyHostToDevice);

   if(h_ui_threads % 8 != 0 || h_ui_blocks % 8 != 0) {
      fprintf(stderr, "ERROR: Blocks and Threads must be multiples of 8\n");
      exit(-1);
   }

   tic_reset();
   hostInitCudaRand();

#if(DEBUG)
   hostDebugTestRand();
#endif

   hostRMATandFileIO();

   cudaMemcpy(&hosterror, d_gep_error, sizeof(graphError_t), cudaMemcpyDeviceToHost);

   hostFreeCudaRand();
   cudaFree(d_gep_error);

   cudaThreadSynchronize();
   printf("\nfree() %f", tic_sincelast());
   printf("\nTotalTime %f\n", tic_total());

	if(cudaPeekAtLastError() != cudaSuccess) 
		printf("**********************************"
		"\nCUDA ERROR OCCURED :\n\t%s\nRESULTS MAY NOT BE VALID\n"
		"**********************************\n", cudaGetErrorString(cudaGetLastError()));
   else if(hosterror != NO_ERROR)
      printf("************************"
      "\nGRAPH ERROR OCCURED:"
      "\n%d - %s"
       "\n************************\n", hosterror, graphErrorString[hosterror]);
   else
      printf("NO ERRORS\n");

   return 0;
}

__host__ void hostParseArgs(int argc, char** argv) {
   static struct option long_options[] = {
      {"scale", required_argument, 0, 's'},
      {"edgefactor", required_argument, 0, 'e'},
      {"actions", required_argument, 0, 'a'},
      {"help", no_argument, 0, 'h'},
      {"blocks", required_argument, 0, 'b'},
      {"threads", required_argument, 0, 't'},
      {"outfile", required_argument, 0, 'o'},
      {"dimacsoutfile", required_argument, 0, 'd'},
      {"STINGERoutputfile", required_argument, 0, 'S'},
      {"STINGERactionsfile", required_argument, 0, 'A'},
      {"CUDADevice", required_argument, 0, 'c'},
      {0, 0, 0, 0}
   };

   int32_t intout;

   while(1) {
      int option_index = 0;                          
      int c = getopt_long(argc, argv, "s:e:a:h?b:t:o:d:S:A:c:", long_options, &option_index);
      extern char * optarg;
      extern int    optind, opterr, optopt;

      if(-1 == c)
         break;
      
      switch(c) {
         default:
            printf("Unrecognized option: %c\n\n", c);
         case '?':
         case 'h':
            printf("\nUsage"
                   "\n====="
                   "\n\t-s --scale=SCALE"
                   "\n\t-e --edgefact=EDGEFACT"
                   "\n\t-a --actions=NUMBEROFACTIONS"
                   "\n\t-o --outfile=OUTPUTEDGELISTFILE"
                   "\n\t-d --dimacsoutfile=DIMACSFORMATOUTPUTFILE"
                   "\n\t-S --STINGERoutputfile=STINGEROUTPUTFILE"
						 "\n\t-A --STINGERactionsfile=STINGERACTIONSFILE"
                   "\n\n\tTUNING"
                   "\n\t-b --blocks=BLOCKS"
                   "\n\t-t --threads=THREADS"
                   "\n\t-d --CUDADevice=DEVICENUMBER - if not specified, default is used"
                   "\n\nEdge list files are binary files containing uint32 scale, edgefactor, and"
                   " edges as ordered pairs of uint32\n");
            exit(0);
            break;
         case 's':
            errno = 0;
            intout = strtol(optarg, NULL, 10);
            if(errno || intout < 0) {
               printf("Error - Scale = %s\n", optarg);
               exit(-1);
            }
            h_ui_scale = intout;
            break;
         case 'e':
            errno = 0;
            intout = strtol(optarg, NULL, 10);
            if(errno || intout < 0) {
               printf("Error - Edgefactor = %s\n", optarg);
               exit(-1);
            }
            h_ui_edgefactor = intout;
            break;
         case 'a':
            errno = 0;
            intout = strtol(optarg, NULL, 10);
            if(errno || intout < 0) {
               printf("Error - Actions = %s\n", optarg);
               exit(-1);
            }
            h_ui_actions = intout;
            break;
         case 'b':
            errno = 0;
            intout = strtol(optarg, NULL, 10);
            if(errno || intout < 0) {
               printf("Error - BLOCKS = %s\n", optarg);
               exit(-1);
            }
            h_ui_blocks = intout; 
            break;
         case 't':
            errno = 0;
            intout = strtol(optarg, NULL, 10);
            if(errno || intout < 0) {
               printf("Error - THREADS = %s\n", optarg);
               exit(-1);
            }
            h_ui_threads = intout; 
            break;
         case 'c':
            errno =0;
            intout = strtol(optarg, NULL, 10);
            if(errno || intout < 0) {
               printf("Error - CUDA Device = %s\n", optarg);
               exit(-1);
            }
            cudaSetDevice(intout);
            break;
         case 'i':
            if(optarg != NULL)
               h_s_infile = optarg;
            break;
         case 'o':
            if(optarg != NULL)
               h_s_outfile = optarg;
            break; 
         case 'd':
            if(optarg != NULL)
               h_s_dimacsoutfile = optarg;
            break;
         case 'S':
            if(optarg != NULL)
               h_s_stinger_outfile = optarg;
            break;
         case 'A':
            if(optarg != NULL)
               h_s_stinger_actionsfile = optarg;
            break;
         case 'p':
            h_i_streaming = 1;
            break;
      }
   }
   
   h_ui_vertices = (1L << h_ui_scale);
   h_ui_edges = h_ui_vertices * h_ui_edgefactor;
   
   if(h_s_infile == NULL) {
      printf("<BLOCKS, THREADS>  <%u, %u>\n", h_ui_blocks, h_ui_threads);
      printf("\n\tScale      %d\n\tEdgefactor %d\n\tActions    %d\n\t<V,E>      <%d,%d>\n", 
         h_ui_scale, h_ui_edgefactor, h_ui_actions, h_ui_vertices, h_ui_edges);
   }
}

__host__ void hostInitCudaRand() {
   uint32_t totalThreads = h_ui_blocks * h_ui_threads;
   uint32_t * hostRandVals = (uint32_t *)xmalloc(totalThreads * sizeof(uint32_t));
   d_uip_randvals = (uint32_t *)cudaXmalloc(totalThreads * sizeof(uint32_t));

   struct timeval tv;
   gettimeofday(&tv, NULL);
   hostRandVals[0] = tv.tv_sec * INITRANDMULT + INITRANDINCREMENT;
   uint32_t i;
   for(i = 1; i < totalThreads; ++i) {
      hostRandVals[i] = hostRandVals[i-1] * INITRANDMULT + INITRANDINCREMENT;
   }

   cudaMemcpy(d_uip_randvals, hostRandVals, totalThreads * sizeof(uint32_t), cudaMemcpyHostToDevice);

   free(hostRandVals);
   
   cudaThreadSynchronize();
   printf("\nhostInitCudaRand() %f", tic_sincelast());
}

__host__ void hostRMATandFileIO() {
	if(h_ui_edges % (h_ui_blocks * h_ui_threads) != 0) {
		printf("ERROR: Edges must divide evenly by blocks * threads.\n");
		exit(-1);
	};

	d_uip_edgelist    		= (uint32_t *)cudaXmalloc(EDGEBATCHSIZE * 4 * sizeof(uint32_t));
	h_uip_edgelist				= (uint32_t *)xmalloc(h_ui_edges * 4 * sizeof(uint32_t));
	d_ip_actionslist        = (int32_t  *)cudaXmalloc(ACTIONBATCHSIZE * 4 * sizeof(int32_t));
	uint32_t * tempactions  = (uint32_t *)cudaXmalloc(ACTIONBATCHSIZE * 4 * sizeof(uint32_t));
	h_ip_actionslist			= (int32_t  *)xmalloc(h_ui_actions * 4 * sizeof(int32_t));

	cudaThreadSynchronize();
	printf("\nedgeListMalloc %f", tic_sincelast());

	if(h_ui_actions % (h_ui_blocks * h_ui_threads) != 0) {
		printf("ERROR: Actions must divide evenly by blocks * threads.\n");
		exit(-1);
	}

	uint32_t * edgedest = h_uip_edgelist;
	int32_t * actiondest = h_ip_actionslist;
	for(uint64_t j = 0, k = 0; j < h_ui_edges || k < h_ui_actions; j += EDGEBATCHSIZE, k += ACTIONBATCHSIZE) {
		if(j < h_ui_edges) {
			uint32_t generate = (h_ui_edges - j > EDGEBATCHSIZE ? EDGEBATCHSIZE : h_ui_edges - j);
			cudaRMATEdges<<<h_ui_blocks, h_ui_threads>>>(d_uip_randvals, h_ui_scale, generate / (h_ui_blocks * h_ui_threads), h_ui_blocks * h_ui_threads, 0.55, 0.1, 0.1, 0.25, d_uip_edgelist);
			cudaMemcpy(edgedest, d_uip_edgelist, generate * 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			edgedest += EDGEBATCHSIZE;
		}
		if(k < h_ui_actions) {
			uint32_t generate = (h_ui_actions - k > ACTIONBATCHSIZE ? ACTIONBATCHSIZE : h_ui_actions - k);
			cudaRMATEdges<<<h_ui_blocks, h_ui_threads>>>(d_uip_randvals, h_ui_scale, generate / (h_ui_blocks * h_ui_threads), h_ui_blocks * h_ui_threads, 0.55, 0.1, 0.1, 0.25, tempactions);
			cudaGenerateActions<<<h_ui_blocks / 8, h_ui_threads>>>(d_uip_randvals, h_ui_edges, generate, h_ui_blocks * h_ui_threads / 8, 0.0625, d_uip_edgelist, tempactions, d_ip_actionslist, d_gep_error); 
			cudaMemcpy(actiondest, d_ip_actionslist, generate * 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			actiondest += ACTIONBATCHSIZE; 
		}
	}

	cudaFree(d_uip_edgelist);
	cudaFree(d_ip_actionslist);
	cudaFree(tempactions);
	
	cudaThreadSynchronize();
	printf("\ncudaRMATEdges() %f", tic_sincelast());

	if(h_s_outfile != NULL) {
		FILE * fp;
		fp = fopen(h_s_outfile, "w+"); 

		if(fp == NULL) {
			fprintf(stderr, "\nERROR: Could not open output file.\n");
			exit(-1);
		}

		uint32_t written = 0;
		written += fwrite(&h_ui_scale, sizeof(uint32_t), 1, fp);
		written += fwrite(&h_ui_edgefactor, sizeof(uint32_t), 1, fp);

		if(written != 2) {
			fprintf(stderr, "\nERROR: Opened output file, but could not write to it.\n");
			exit(-1);
		}

		written = fwrite(h_uip_edgelist, sizeof(uint32_t), 4 * h_ui_edges, fp); 

		if(written != 4 * h_ui_edges) {
			fprintf(stderr, "\nERROR: Opened output file, but could not write to it.\n");
			exit(-1);
		}
		
		fclose(fp);
		cudaThreadSynchronize();
		printf("\nWriteOutputFile() %f", tic_sincelast());
	} 
	
	if(h_s_dimacsoutfile != NULL) {
		FILE * fp;
		fp = fopen(h_s_dimacsoutfile, "w+"); 

		if(fp == NULL) {
			fprintf(stderr, "\nERROR: Could not open output file.\n");
			exit(-1);
		}

		fprintf(fp, "c graph generated by CUDARMAT\n");
		fprintf(fp, "p sp %d %d\n", h_ui_vertices, 2 * h_ui_edges);

		uint32_t j;
		for(j = 0; j < h_ui_edges * 4; j += 2)
			fprintf(fp, "a %d %d 1", h_uip_edgelist[j], h_uip_edgelist[j+1]);
		
		fclose(fp);
		cudaThreadSynchronize();
		printf("\nWriteDimacsOutputFile() %f", tic_sincelast());
	}

	if(h_s_stinger_outfile != NULL) {
		FILE * fp;
		fp = fopen(h_s_stinger_outfile, "w+"); 

		if(fp == NULL) {
			fprintf(stderr, "\nERROR: Could not open output file.\n");
			exit(-1);
		}

		uint32_t written = 0;
		int64_t v64 = h_ui_vertices;
		int64_t e64 = h_ui_edges * 2;     
		int64_t ec = 0x1234ABCD;
		written += fwrite(&ec, sizeof(int64_t), 1, fp);
		written += fwrite(&v64, sizeof(int64_t), 1, fp);
		written += fwrite(&e64, sizeof(int64_t), 1, fp);

		if(written != 3) {
			fprintf(stderr, "\nERROR: Opened output file, but could not write to it.\n");
			exit(-1);
		}

		qsort(h_uip_edgelist, 2 * h_ui_edges, 2 * sizeof(uint32_t), hostCompareEdges);

		int64_t * off = (int64_t *)xcalloc((h_ui_vertices + 1), sizeof(int64_t));
		int64_t * ind = (int64_t *)xmalloc(h_ui_edges * 2 * sizeof(int64_t));
		int64_t * weight = (int64_t *)xmalloc(h_ui_edges * 2 * sizeof(int64_t));

		off += 1;
		uint32_t j, k = 0;
		for(j = 0; j < h_ui_edges * 4; j += 2) {
			off[h_uip_edgelist[j]]++;
			ind[k] = h_uip_edgelist[j+1];
			weight[k] = 1;
			k++;
		}

		for(j = 1; j < h_ui_vertices; ++j)
			off[j] += off[j - 1];
		
		off -= 1;

		written = fwrite(off, sizeof(int64_t), h_ui_vertices + 1, fp);
		written += fwrite(ind, sizeof(int64_t), h_ui_edges * 2, fp);
		written += fwrite(weight, sizeof(int64_t), h_ui_edges * 2, fp);

		if(written != 4 * h_ui_edges + h_ui_vertices + 1) {
			fprintf(stderr, "\nERROR: Opened output file, but could not write to it.\n");
			exit(-1);
		}
		
		free(off);
		free(ind);
		free(weight);
		fclose(fp);
		cudaThreadSynchronize();
		printf("\nWriteSTINGEROutputFile() %f", tic_sincelast());
	}

	if(h_s_stinger_actionsfile != NULL) {
		FILE * fp;
		fp = fopen(h_s_stinger_actionsfile, "w+"); 

		if(fp == NULL) {
			fprintf(stderr, "\nERROR: Could not open output file.\n");
			exit(-1);
		}

		uint32_t written = 0;
		int64_t actions = h_ui_actions * 2;
		uint64_t ec = 0x1234ABCD;
		written += fwrite(&ec, sizeof(int64_t), 1, fp);
		written += fwrite(&actions, sizeof(int64_t), 1, fp);

		if(written != 2) {
			fprintf(stderr, "\nERROR: Opened output file, but could not write to it.\n");
			exit(-1);
		}

		int64_t * act = (int64_t *)xmalloc(sizeof(int64_t) * h_ui_actions * 4);

		for(uint64_t j = 0; j < h_ui_actions * 4; j++) {
			act[j] = h_ip_actionslist[j];
		}

		written = fwrite(act, sizeof(int64_t), h_ui_actions * 4, fp);

		if(written != 4 * h_ui_actions) {
			fprintf(stderr, "\nERROR: Opened output file, but could not write to it.\n");
			exit(-1);
		}

		free(act);
		fclose(fp);
		cudaThreadSynchronize();
		printf("\nWriteSTINGEROutputFile() %f", tic_sincelast());
	}
	free(h_uip_edgelist);
	free(h_ip_actionslist);
}  

__host__ int hostCompareEdges(const void * a, const void * b) {
   uint32_t * e1 = (uint32_t *)a;
   uint32_t * e2 = (uint32_t *)b;

   if(e1[0] == e2[0])
      return e1[1] - e2[1];
   else
      return e1[0] - e2[0];
}

__host__ void hostFreeCudaRand() { 
   cudaFree(d_uip_randvals);         
}                              

// HOST DEBUGGING FUNCTIONS
/////////////////////////////
#if(DEBUG)
__host__ void hostDebugTestRand() {
   uint32_t * randvalstest;
   uint32_t bins[32] = {0};
   uint32_t boundaries [] = { 134217728U, 268435456U, 402653184U, 536870912U, 671088640U, 805306368U, 939524096U, 1073741824U,
      1207959552U, 1342177280U, 1476395008U, 1610612736U, 1744830464U, 1879048192U, 2013265920U, 2147483648U, 2281701376U,
      2415919104U, 2550136832U, 2684354560U, 2818572288U, 2952790016U, 3087007744U, 3221225472U, 3355443200U, 3489660928U,
      (uint32_t)3623878656U, (uint32_t)3758096384U, (uint32_t)3892314112U, (uint32_t)4026531840U, (uint32_t)4160749568U, (uint32_t)4294967295U };

   uint32_t numrands = h_ui_blocks * h_ui_threads * TESTRAND;

   randvalstest = (uint32_t *)cudaXmalloc(numrands * sizeof(uint32_t));

   cudaDebugRand <<<h_ui_blocks, h_ui_threads>>>(randvalstest, d_uip_randvals);

   uint32_t * results = (uint32_t *) xmalloc(numrands * sizeof(uint32_t));

   cudaMemcpy(results, randvalstest, numrands * sizeof(uint32_t), cudaMemcpyDeviceToHost);

   int i,j;
   //int duplicates = 0;

   for(i = 0; i < numrands; ++i) {
      //printf("%u\n", results[i]);

      for(j = 0; j < 32; ++j) {
         if(results[i] < boundaries[j]) {
            bins[j]++;
            break;
         }
      }

      //for(j = i+1; j < numrands; ++j) {
      // if(results[i] == results[j])
      //    duplicates++;
      //}
   }

   free(results);

   printf("\n\n***BEGIN BINS***\n\n");
   for(i = 0; i < 32; ++i) {
      printf("%u\n", bins[i]);
   }

   //printf("\n***DUPLICATES %d***\n", duplicates);

   cudaFree(randvalstest);
}

#endif

// Device Functions
//////////////////////////////
__device__ uint32_t cudaRand(uint32_t * randVal) {
  (*randVal) = ((*randVal) * RANDMULT + RANDINCREMENT);
  return *randVal;
}

__global__ void cudaRMATEdges(uint32_t * randVals, uint32_t SCALE, uint32_t edgesPerThread, uint32_t numthreads, 
                                 float pA, float pB, float pC, float pD, uint32_t * edgeArray) {
   __shared__ uint32_t aboutToWrite[MAXBLOCKSIZE];
   uint32_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x);
   int swap = thread_id % 2 == 0 ? 1 : -1;
   uint32_t myRand = randVals[thread_id];  

   float A, B, C, D;

   uint32_t iteration; 
   uint32_t step = numthreads * 4;
   uint32_t stop = numthreads * 4 * edgesPerThread;
   for(iteration = 0; iteration < stop; iteration += step) {
      A = pA;
      B = pB;
      C = pC;
      D = pD;

      uint32_t i = 0;
      uint32_t j = 0;
      uint32_t curBit = ((uint32_t) 1) << (SCALE - 1);

      while(1) {
         const float rand = ((float)cudaRand(&myRand)) / (4294967295.0f);

         if(rand > A) {
            if(rand <= A + B)
               j |= curBit;
            else if (rand <= A + B + C)
               i |= curBit;
            else {
               j |= curBit;
               i |= curBit;
            }
         }

         if(1 == curBit)
            break;

         A *= (0.95 + (((float)cudaRand(&myRand)) / (42949672950.0f)));
         B *= (0.95 + (((float)cudaRand(&myRand)) / (42949672950.0f)));
         C *= (0.95 + (((float)cudaRand(&myRand)) / (42949672950.0f)));
         D *= (0.95 + (((float)cudaRand(&myRand)) / (42949672950.0f)));
         
         const float norm = 1.0 / (A + B + C + D);
         A *= norm;
         B *= norm;
         C *= norm;
         D = 1.0 - (A + B + C);

         curBit >>= 1;
      }

      if(swap == 1) {
         aboutToWrite[threadIdx.x] = i;
      } else {
         aboutToWrite[threadIdx.x] = j;
      }
     
      if(swap == 1) { 
         if(aboutToWrite[threadIdx.x + swap] == j) {
            j ^= 1;
         }
      } else {
         if(aboutToWrite[threadIdx.x + swap] == i)  {
            i ^= 1;                            
         }
      }

      __syncthreads();

      uint32_t index = thread_id + iteration;
      edgeArray[index] = i;
      index += numthreads;
      edgeArray[index] = j;
      index += numthreads;
      edgeArray[index + swap] = j;
      index += numthreads;
      edgeArray[index + swap] = i;
   }
}

__global__ void cudaGenerateActions(uint32_t * randVals, uint32_t edges, uint32_t actions, uint32_t numthreads, 
                                    float pDelete, uint32_t * edgeArray, uint32_t * generatedEdges, int32_t * actionsEdgeArray, graphError_t * error) {
   uint32_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x);
   uint32_t threadIdx4 = threadIdx.x * 4;

   __shared__ int32_t sharedActions[2048];

   uint32_t myRand         = randVals[thread_id];  
   uint32_t original_del   = thread_id * 2;
   uint32_t new_del        = original_del;
   uint32_t new_ins        = original_del;
   uint32_t stop           = 4 * actions;
   uint32_t step           = 4 * numthreads;
   uint32_t index = thread_id;
   for(; index < stop; index += step) {
      const float rand = ((float)cudaRand(&myRand)) / (4294967295.0f);
      if(rand >= pDelete) {
         sharedActions[threadIdx4]   = generatedEdges[new_ins];
         sharedActions[threadIdx4 +1] = generatedEdges[new_ins+1];
         new_ins += step;
      } else {
         if(original_del < edges * 4) {
            sharedActions[threadIdx4]   = -edgeArray[original_del];
            sharedActions[threadIdx4 +1] = -edgeArray[original_del+1];
            original_del += step;
         } else if(new_del < new_ins) {
            sharedActions[threadIdx4]   = -generatedEdges[new_del];
            sharedActions[threadIdx4 +1] = -generatedEdges[new_del+1];
            new_del += step;
         } else {
            // Deletes caught up to insertions
            // if you are near a window, check for flying pigs
            *error = DELETIONS_OVERRAN_INSERTIONS;
         }
      }
      
      //reverse edges
      sharedActions[threadIdx4 + 2] = sharedActions[threadIdx4 + 1];
      sharedActions[threadIdx4 + 3] = sharedActions[threadIdx4];
      __syncthreads();

      actionsEdgeArray[index]                   = sharedActions[threadIdx.x];
      actionsEdgeArray[index + blockDim.x]      = sharedActions[threadIdx.x + blockDim.x];
      actionsEdgeArray[index + blockDim.x * 2]  = sharedActions[threadIdx.x + blockDim.x * 2];
      actionsEdgeArray[index + blockDim.x * 3]  = sharedActions[threadIdx.x + blockDim.x * 3]; 
   }
}


// DEVICE DEBUGGING FUNCTIONS
//////////////////////////////
__global__ void cudaDebugRand(uint32_t * randOut, uint32_t * randVals) {
   int i;
   int thread_id = (blockIdx.x * blockDim.x + threadIdx.x); 
   int thread_offset = thread_id * TESTRAND; 

   for(i = 0; i < TESTRAND; ++i) {
      randOut[i + thread_offset] = cudaRand(randVals + thread_id);
   }
}

__global__ void cudaDebugEdgeList(uint32_t vertices, uint32_t * edgeList, uint32_t size) {
   uint32_t i = 0;
   for(i = 0; i < size; ++i) {
      if(edgeList[i] > vertices)
         edgeList[i] = 0xFFFFFFFF - i;
   }
}
