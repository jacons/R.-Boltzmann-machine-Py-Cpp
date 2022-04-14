#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib> 
#include <cmath>
#include <iomanip>
#include <chrono>

#define RAND static_cast<double>(rand()) / static_cast<double>(RAND_MAX)
#define TIME std::chrono::steady_clock::time_point
#define GETTIME std::chrono::steady_clock::now()
#define ELAPSED_ML(a,b) std::chrono::duration_cast<std::chrono::milliseconds>((a) - (b)).count()
#define ELAPSED_S(a,b) std::chrono::duration_cast<std::chrono::seconds>((a) - (b)).count()

#define S(x)  1 / (1 + exp(-(x))) // Sigmoid activation function
#define EPOCHS 50                 // Number of epochs
#define V 784                     // Number of pixels in handigit'image
#define H 81                      // Hidden units
#define SIZEDS 60000              // Number of examples

class BoltzmannMachine {
    private: 

        int v;            // Number of visibile nodes
        int h;            // Number of hidden nodes
        float lRate;      // Learning rate
        int bSize;        // Batch size
        double **weights; // Matrix of weights
        double **deltaW;  // Matrix of deltas of weights (minibatch approach)

        double *v_bias;   // Vector of visible bias
        double *h_bias;   // Vector of hidden bias
        double *deltaV;   // Vector of deltas of visible bias (mb approach)
        double *deltaH;   // Vector of deltas of hidden bias (mb approach)

        // Temporary vectors allocated to avoid several malloc

        double *h0_p;  // Hidden state a time 0 probability 
        double *h0_s;  // Hidden state a time 0 sampled 
        double *v1_p;  // Visible state a time 0 probability 
        double *v1_s;  // Visible state a time 0 probability 
        double *h1_p;  // Hidden state a time 1 probability 

        std::vector<double*> ds; // Dataset
        std::vector<short*> recon; // Restricted dataset reconstructed

        /**
         * @brief Allocate a matrix of double values
         * 
         * @param r numbers of rows
         * @param c number of columns
         * @return double** pointer to matrix
         */
        double** matrix(int r,int c) {
            double** m = (double**) malloc (r * sizeof(double*));
            for (int i = 0; i < r; i++) m[i] = (double*) malloc(c * sizeof(double));
            return m;
        }

        /**
         * @brief Allocate an array of double values
         * 
         * @param dim number of elements
         * @return double* pointer to array
         */
        double* array(int dim) {
            double* arr = (double*)malloc(dim*sizeof(double));
            return arr;
        }

        /**
         * @brief Custumized dot product, it multiplies m x b and put
            the result in "result"
         * 
         * @param result result array pointer 
         * @param a matrix pointer
         * @param b array pointer
         * @param t type of dot product
         * @return double* result pointer
         */
        double* dot(double* result,double**m,double* b,bool t) {
            int i,j;

            if(t) {
                for(i = 0; i < V; i++) result[i] = 0;
                for(i = 0; i < V; i++)
                    for(j = 0; j < H; j++) result[i] += m[j][i] * b[j];
            } else {
                for(i = 0; i < H; i++) result[i] = 0;
                for(i = 0; i < H; i++)
                    for(j = 0; j < V; j++) result[i] += m[i][j] * b[j];                
               }
            return result;
        }

        /**
         * @brief Sigmoid activaction function (mapping), apply sigmoid
            for each element in the input
         * 
         * @param a input vector
         * @param dim 
         */
        void sigmoid(double*a,int dim) {
            for(int i = 0; i < dim; i++) a[i] = S(a[i]);
            return;
        }

        /**
         * @brief Summing array "a" and "b" and put the result in "a"
         * 
         * @param a source 1 (end result)
         * @param b source 2
         * @param dim dimentions of array
         * @return double* 
         */
        double* add(double*a,double*b,int dim) {
            for(int i = 0; i < dim; i++) a[i] += b[i];
            return a;          
        }
    
        /**
         * @brief Update the weights (minibach approach)
         * 
         * @param dim number of example seen
         */
        void updateWs(int s) {
            int i,j;
            for (i = 0; i < H; i++) {
                for (j = 0; j < V; j++) {
                    this->weights[i][j] += lRate * (this->deltaW[i][j] / s);
                    this->deltaW[i][j] = 0;
                }
                this->h_bias[i] += lRate * (this->deltaH[i] / s);
                this->deltaH[i] = 0;
            }
            for (j = 0; j < V; j++) {
                this->v_bias[j] += lRate * (this->deltaV[j] / s) ;
                this->deltaV[j] = 0;
            }
            return;
        }

        /**
         * @brief  Contrastive Divergence - 1 algorithm
         * 
         * @param v0 input array (visible units)
         */
        void cd0(double* v0) {
            int i,j; 

            // Construct phase (hidden state a time 0 probability)
            sigmoid(add(dot( h0_p , weights , v0 , false ), h_bias , H ), H );
            
            // Sampling h0, transform the probability into a bool vector
            for( i = 0; i < H; i++) h0_s[i] = h0_p[i] > RAND;
            
            // Reconstruct phase (visible state a time 1 probability)
            sigmoid(add(dot( v1_p , weights , h0_s , true ), v_bias , V ), V );

            // Sampling h0, transform the probability into a bool vector
            for( i = 0;i < V; i++) v1_s[i] = v1_p[i] > RAND;

            // Construct phase (hidden state a time 1 probability)
            sigmoid(add(dot( h1_p , weights , v1_s , false ), h_bias , H ), H );
            
            // Update deltas
            for (j = 0; j < V; j++) {
                for(i = 0; i < H; i++) this->deltaW[i][j] += h0_p[i] * v0[j] - h1_p[i] * v1_s[j];
                this->deltaV[j] += v0[j] - v1_p[j];   
            }
            for(i = 0;i < H; i++) this->deltaH[i] += h0_p[i] - h1_p[i];
            return;
        }

        /**
         * @brief The core of RBM is the ability to apply the representation of learning,
                indeed it has learnt by unlabelled examples how to transform the huge visible
                state in a smaller representation. (Feature reduction
         * 
         * @param v0 Visible stare (huge vector)
         * @param r Hidden state (smaller vector)
         */
        void inference(double* v0,short* r) {

            // We use h0_p as temporary variable
            sigmoid(add(dot( h0_p , weights , v0 , false ), h_bias , H ), H );

            // We need apply the sampling, in order to return a boolean vector
            for(int i = 0; i < H ; i++) r[i] = h0_p[i] > RAND;     
            return;
        }

    public:

        /**
         * @brief Constructor, it takes a number of visibile and hidden unit,
            this Restricted Boltzmann Machine has just one layer
         * 
         * @param vNode Number of visibile units
         * @param hNode Number of hidden units
         * @param lRate Learning rate
         * @param bSize Batch size
         * @param ds    dataset
         */
        BoltzmannMachine(int vNode,int hNode,float lRate,int bSize,std::vector<double*> ds) {

            this->v = vNode;                     // Set the number of visible node
            this->h = hNode;                     // Set the number of hidden node
            this->lRate = lRate;                 // Learning rate   
            this->bSize = bSize;                 // Batch Size  
            this->ds = ds;                       // Dataset  
            this->weights = this->matrix(H,V);   // Allocate a matrix of weights (no initializated)
            this->v_bias  = this->array(V);      // Allocate visible bias 
            this->h_bias  = this->array(H);      // Allocate hidden bias
            this->deltaW  = this->matrix(H,V);   // Allocate a matrix of weights (no initializated)
            this->deltaV  = this->array(V);      // Allocate visible bias 
            this->deltaH  = this->array(H);      // Allocate hidden bias
            this->h0_p    = this->array(H);      // Hidden0 construction (probability)
            this->h0_s    = this->array(H);      // Hidden0 construction (sampled)
            this->v1_p    = this->array(V);      // Visible1 reconstruction (probability)
            this->v1_s    = this->array(V);      // Visible1 reconstruction (sampled)
            this->h1_p    = this->array(V);      // Hidden1 constuction (probability)

            /* Inizialize the weights and bias with random numbers between 0 and 1
                initialize also the deltas to 0 */

            int i,j;
            srand((unsigned)time(0)); 
            for (i = 0; i < H; i++) {
                for (j = 0; j < V; j++) {
                    this->weights[i][j] = RAND;
                    this->deltaW[i][j] = 0; 
                }
                this->h_bias[i] = RAND;
                this->deltaH[i] = 0;
            }
            for (j = 0; j < V; j++) {
                this->v_bias[j] = RAND;
                this->deltaV[j] = RAND;
            }

        }

        /**
         * @brief Apply learning algorithm, we scan for a number of epoch the dataset
            by applying the correction rule. The algorthm uses a minibatch approach, thus
            the weights (and bias) are update only a "bSize" steps.
         * 
         */
        void fit() {
            int c = 0; // to keep the number of iteration done 

            // for each epoch we scan the entire dataset
            for(int e=0;e<EPOCHS;e++) {
                std::cout << "Epoch #" << e << std::endl;

                // foreach example in the dataset, x represent a verctor of V value 0 or 1
                for(double* x : this->ds) { 

                    if(c==this->bSize) {
                        // updating the weights
                        this->updateWs(c);
                        c = 0; // erase the counter 
                    }
                    // apply learning algorithm 
                    this->cd0(x);
                    c++;
                }
                // flushing the deltas, before the next dataset iteration
                this->updateWs(c);
            }
            return;
        }

        /**
         * @brief Build a Restricted dataset with onyl H values, 
            each example is the "hidden state (10 values)" of the visible state (768 values) 
         */
        void reconstruct() {
            // Inizialize the memory for the restricted dataset
            recon.reserve(SIZEDS);

            // foreach element in the original dataset
            for(double* x : this->ds) {
                // define a boolean vector with only H values
                short* t = (short*) malloc(H*sizeof(short));

                /* apply the inference, so we "representing" the inital input (768) visible node
                 to only 10 node, a sort of DIMENSIONALLY REDUCTION
                */
                this->inference(x,t);
                // insert the "compressed example" in the restricted dataset
                recon.push_back(t);
            }
            return;
        }

        void exportDs() {
            std::ofstream f("restrictedDataset.csv");   
            int i;
            for(short* row : recon) {
                for(i = 0; i < H-1; i++) {
                    f << row[i] << ",";
                }
                f << row[H-1] << std::endl;
            }
            f << std::flush;
            f.close();  
            return; 
        }

        void exportFilters() {
            std::ofstream f("weights.csv");
            f << std::setiosflags(std::ios::fixed) << std::setprecision(8);
            int i,j;
            for(i = 0; i < H; i++) {
                for(j = 0; j < V-1; j++) {
                    f << weights[i][j] << ",";
                }
                f << weights[i][V-1] << std::endl;
            }
            f << std::flush;
            f.close();
            return;
        }

        void printW() {
            int i,j;
            for (i = 0; i < H; i++) {
                for (j = 0; j < V; j++) std::cout << this->weights[i][j] << " " ;
                std::cout << std:: endl;
            }
            return;
        }

        void clear() {

                for (int i = 0; i < H; i++) {
                    free(weights[i]);
                    free(deltaW[i]);
                }
                free(deltaV);free(deltaH);
                free(v_bias);free(h_bias);
                free(h0_p);  free(h0_s);
                free(h1_p);  free(v1_s);   
                free(v1_p);  free(weights);
                free(deltaW);

                for(short* i : recon) free(i);
            return;
        }
};

int main() {
    

    std::vector<double*> ds; // origial dataset
    ds.reserve(SIZEDS); // initialize space to load the entire dataset



    // ---------------------------  LOADING PHASE ---------------------------
    std::cout << "Loading dataset" << std::endl;

    TIME begin = GETTIME;
    TIME start = begin;

    std::ifstream file("dataset.csv");    
    if (file.is_open()) {

        std::string line;
        int count = 0;

        std::getline(file, line); // ignore first row (labels)
        while (std::getline(file, line)) {

                // initialize the vector that contatins all features for each example
                double* row = (double*) malloc(V*sizeof(double));

                std::stringstream sstr(line.c_str());

                sstr.good(); // ignore first element (output)
                while(sstr.good()) {
                    std::string substr;
                    getline(sstr, substr, ',');
                    
                    /* foreach pixel values we apply a threshold,
                    we rescaling the value from (0,255) to 0 or 1 */
                    row[count++] = stof(substr)>128;
                }
                count=0;
                // put inside the dataset the example
                ds.push_back(row);
            }
    }

    std::cout << "Loading complete" << "(it takes :" << ELAPSED_ML(GETTIME,begin) << " ms)" << std::endl;
    // ---------------------------  LOADING PHASE ---------------------------


    // ---------------------------  LEARNING PHASE ---------------------------
    std::cout << "Learning phase" << std::endl;
    begin = GETTIME;

    BoltzmannMachine* rbm = new BoltzmannMachine(V,H,1,30,ds);
    rbm->fit();

    std::cout << "Learning complete" << "(it takes :" << ELAPSED_S(GETTIME,begin) << " ms)" << std::endl;
    // ---------------------------  LEARNING PHASE ---------------------------


    // ---------------------------  RECONSTRUCTING PHASE ---------------------------
    std::cout << "Reconstructing phase" << std::endl << std::endl;
    begin = GETTIME;

    rbm->reconstruct();

    std::cout << "Reconstructing complete" << "(it takes :" << ELAPSED_ML(GETTIME,begin) << " ms)" << std::endl;
    // ---------------------------  RECONSTRUCTING PHASE ---------------------------


    // ---------------------------  EXPORTING PHASE ---------------------------
    std::cout << std::endl << "Exporting/Clean up phase" << std::endl;
    begin = GETTIME;

    rbm->exportDs();
    rbm->exportFilters();
    
    rbm->clear();
    for(double* i : ds) free(i);

    std::cout << "Exporting/Clean up complete" << "(it takes :" << ELAPSED_ML(GETTIME,begin) << " ms)" << std::endl;
    // ---------------------------  EXPORTING PHASE ---------------------------


    std::cout << "Total execution in : " << ELAPSED_S(GETTIME,start) << " ms" << std::endl;

    return 0;
}
