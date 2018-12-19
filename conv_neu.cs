using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet5
{
    public class conv_neu : neural_
    {
        public double[] Kernel;//random generated kernel
        //private double[] Kernel_old;
        public int Kernel_Width;
        public double[] BP_Error_to_Previous_Layer;
        public double[] Second_Deri_Error_to_Previous_Layer;
        public double[] Learning_Rate_Kernels;
        public double Learning_Rate_Bias;
        public double[] hkk_kernel;
        public double hkk_bias;

        public conv_neu(int Width_Image,int amountPreviousLayer, int AmountInThisLayer, int outputWidth)
        {
            Input_Width = Width_Image;
            Output_Width = outputWidth;
            Kernel_Width = Width_Image - outputWidth + 1;
            NeuralAmountThisLayer = AmountInThisLayer;
            NeuralAmountPreviousLayer = amountPreviousLayer;
            Weights_Initialize();
        }

        protected override void Weights_Initialize()
        {
            Kernel = Weight.RandomWeights(25, NeuralAmountPreviousLayer);
            Biased_Weight = Weight.Random(NeuralAmountPreviousLayer);
            hkk_kernel = new double[25];
            hkk_bias = 0.0;
            Learning_Rate_Kernels = new double[25];
        }


        public double[] Evaluation(double[] Input_image)
        {
            Input_Array = new double[Input_image.Length];
            for (int j = 0; j < Input_image.Length; j++)
            {
                Input_Array[j] += Input_image[j];
            }
            Output_Array = Convolution.Cov_array(Input_Array, Kernel, Biased_Weight);
            return Output_Array;
        }



        public void Back_Propagation(double[][] BP_Error_from_Next_Layer)
        {


            var Error_array = new double[Output_Width * Output_Width];


            for (int i = 0; i < BP_Error_from_Next_Layer.Length; i++)
            {
                for (int j = 0; j < Error_array.Length; j++)
                {
                    Error_array[j] += BP_Error_from_Next_Layer[i][j];
                }
            }
            Update_Bias(Error_array);
            UpdateError_to_Previous(Error_array);
            UpdateKernel(Error_array);
        }
        public double[] Back_Propagation(double[] BP_Error_from_Next_Neural)
        {
            var Error_array = new double[Output_Width * Output_Width];
            for (int j = 0; j < Error_array.Length; j++)
            {
                Error_array[j] += BP_Error_from_Next_Neural[j];
            }

            Update_Bias(Error_array);
            var err = UpdateError_to_Previous(Error_array);
            UpdateKernel(Error_array);
            return err;
        }
        public double[] Back_Propagation(double BP_Error_from_Next_Neural)//F6->C5
        {
            var Error_array = new double[1];//1            
                Error_array[0] = BP_Error_from_Next_Neural;
           
            Update_Bias(Error_array);
            var err = UpdateError_to_Previous(Error_array);
            UpdateKernel(Error_array);
            return err;
        }

        public double[] PreTrain(double Second_Deri_Error_from_Next_Neural)//F6->C5
        {
            var Second_Deri_Error_array = new double[] { Second_Deri_Error_from_Next_Neural };
          
            PreTrain_Kernel(Second_Deri_Error_array);
            PreTrain_Bias(Second_Deri_Error_array);
           return Second_Deri_Error_to_Previous(Second_Deri_Error_array);

        }
        public double[] PreTrain(double[] Second_Deri_Error_from_Next_Neural)
        {
            var Second_Deri_Error_array = Second_Deri_Error_from_Next_Neural;

            PreTrain_Kernel(Second_Deri_Error_array);
            PreTrain_Bias(Second_Deri_Error_array);
            return Second_Deri_Error_to_Previous(Second_Deri_Error_array);

        }
        public void PreTrain(double[][] Second_Deri_Error_from_Next_Layer)
        {
            var Second_Deri_Error_array = new double[Output_Width * Output_Width];

            for (int i = 0; i < Second_Deri_Error_from_Next_Layer.Length; i++)
            {
                for (int j = 0; j < Second_Deri_Error_array.Length; j++)
                {
                    Second_Deri_Error_array[j] += Second_Deri_Error_from_Next_Layer[i][j];
                }
            }
            PreTrain_Kernel(Second_Deri_Error_array);
            PreTrain_Bias(Second_Deri_Error_array);
            Second_Deri_Error_to_Previous(Second_Deri_Error_array);
        }

        protected virtual void UpdateKernel(double[] Error_Array)
        {
            //dOutput[a,b]/dw[x,y]
           // Kernel_old = Kernel;
            int kernel_Width = Input_Width - Output_Width + 1;
            for (int x = 0; x < kernel_Width; x++)
            {
                for (int y = 0; y < kernel_Width; y++)
                {
                    double do_dw = 0.0;
                    for (int i = 0; i < Output_Width; i++)
                    {
                        for (int j = 0; j < Output_Width; j++)
                        {
                            int k = i * Output_Width + j;
                            //do_dw += Input_Array[(i + x) * Input_Width + (j + y)] * 2 / 3 * (1-Output_Array[i * Output_Width + j]* Output_Array[i * Output_Width + j]) * (Error_Array[i * Output_Width + j]);
                            do_dw += Input_Array[(i + x) * Input_Width + (j + y)] * Deri_ATanhSa(Output_Array[k]) * Error_Array[k];
                           //d(Output)/dW_xy=sumation(d(Output_ij)/dW_xy)
                        }
                    }
                    Kernel[x * kernel_Width + y] = Kernel[x * kernel_Width + y] - Learning_Rate_Kernels[x * kernel_Width + y] * do_dw;
                }
            }
        }
       


        protected virtual void Update_Bias(double[] Error_Array)
        {
           // var Sum = Error_Array.Sum();
            double res = 0;
            for (int i = 0; i < Output_Width; i++)
            {
                for (int j = 0; j < Output_Width; j++)
                {
                    int k = i * Output_Width + j;
                    res += Error_Array[k] * Deri_ATanhSa(Output_Array[k]);
                }

            }


           // Biased_Weight = Biased_Weight - Learning_Rate * Sum;
            Biased_Weight = Biased_Weight - Learning_Rate_Bias * res;
            //Console.WriteLine($"Biased_Weight{Biased_Weight}");
        }


        protected virtual double[] UpdateError_to_Previous(double[] Error_Array)
        {
            int kernel_Width = Input_Width - Output_Width+1 ;
            BP_Error_to_Previous_Layer = new double[Input_Array.Length];
            for (int x = 0; x < Output_Width; x++)
            {
                for (int y = 0; y < Output_Width; y++)
                {
                    for (int a = 0; a < kernel_Width; a++)
                    {
                        for (int b = 0; b < kernel_Width; b++)
                        {
                            if (a + x < Input_Width && b + y < Input_Width)
                            {
                                int k = x * Output_Width + y;
                                //BP_Error_to_Previous_Layer[(a + x) * Input_Width + b + y] += Error_Array[x * Output_Width + y] * 2 / 3 * (1 - Output_Array[x * Output_Width + y]* Output_Array[x * Output_Width + y]) * Kernel[a * kernel_Width + b];
                                BP_Error_to_Previous_Layer[(a + x) * Input_Width + b + y] += Error_Array[k] * Deri_ATanhSa(Output_Array[k]) * Kernel[a * kernel_Width + b];
                                //d(Error)/dX_a+x,y+b
                            }
                        }
                    }
                }
            }
            return BP_Error_to_Previous_Layer;
        }

        private void PreTrain_Kernel(double[] Second_Deri_Error_Array)
        {
            int kernel_Width = Input_Width - Output_Width + 1;
            for (int x = 0; x < kernel_Width; x++)
            {
                for (int y = 0; y < kernel_Width; y++)
                {
                    double Second_Deri = 0.0;
                    for (int i = 0; i < Output_Width; i++)
                    {
                        for (int j = 0; j < Output_Width; j++)
                        {
                            int k = i * Output_Width + j;
                            int index = (i + x) * Input_Width + (j + y);
                            double deri = Deri_ATanhSa(Output_Array[k]);
                            Second_Deri += Input_Array[index] * Input_Array[index] * deri*deri * Second_Deri_Error_Array[k];
                        }
                    }

                    hkk_kernel[x * kernel_Width + y] += Second_Deri;
                }
            }


        }
        protected virtual void PreTrain_Bias(double[] Second_Deri_Error_Array)
        {
           
            double res = 0;
            for (int i = 0; i < Output_Width; i++)
            {
                for (int j = 0; j < Output_Width; j++)
                {
                    int k = i * Output_Width + j;
                    double deri = Deri_ATanhSa(Output_Array[k]);
                    res += Second_Deri_Error_Array[k]*deri*deri *1.0*1.0;
                }

            }

            hkk_bias += res;
         
        }
        private double[] Second_Deri_Error_to_Previous(double[] Second_Deri_Error_Array)
        {
            int kernel_Width = Input_Width - Output_Width + 1;
            double[] Second_Error_to_Previous_Layer = new double[Input_Array.Length];
            for (int x = 0; x < Output_Width; x++)
            {
                for (int y = 0; y < Output_Width; y++)
                {
                    for (int a = 0; a < kernel_Width; a++)
                    {
                        for (int b = 0; b < kernel_Width; b++)
                        {
                            if (a + x < Input_Width && b + y < Input_Width)
                            {
                                int k = x * Output_Width + y;
                                int index = (a + x) * Input_Width + b + y;
                                double deri = Deri_ATanhSa(Output_Array[k]);
                                Second_Error_to_Previous_Layer[index] += Second_Deri_Error_Array[k] * deri*deri * Kernel[a * kernel_Width + b] * Kernel[a * kernel_Width + b];
                                //d(Error)/dX_a+x,y+b
                            }
                        }
                    }
                }
            }
            return Second_Error_to_Previous_Layer;
        }

        public void Evaluate_Learning_Rate(double Sample_Count)
        {
            for(int i=0;i<Kernel.Length;i++)
            {
                Learning_Rate_Kernels[i] = TAU / (hkk_kernel[i] / Sample_Count + MU);



            }
            Learning_Rate_Bias = TAU / (hkk_bias / Sample_Count + MU);

        }
        public void Zeroing_HKKs()
        {
            for (int i = 0; i < Kernel.Length; i++)
            {
                hkk_kernel[i] = 0.0;
            }
            hkk_bias = 0.0;

        }


    }
}
