using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet5
{
    public class sub_neu : neural_
    {
        public double Coeff;
        public double[] BP_Error_to_Previous_Layer;
        public double[] Second_Deri_Error_to_Previous_Layer;
        public double hkk_coeff, hkk_bias;
        public double Learning_Rate_coeff, Learning_Rate_bias;

        public sub_neu(int Width_Image, int AmountInThisLayer, int outputWidth)
        {
            Input_Width = Width_Image;
            Output_Width = outputWidth;
            NeuralAmountThisLayer = AmountInThisLayer;
            Weights_Initialize();
        }

        protected override void Weights_Initialize()
        {
            Coeff = Weight.Random(NeuralAmountThisLayer);
            Biased_Weight = Weight.Random(NeuralAmountThisLayer);
            hkk_coeff = 0.0;
            hkk_bias = 0.0;
            Learning_Rate_bias = 0.0;
            Learning_Rate_coeff = 0.0;
        }


        public double[] Evaluation(double[] Input_Image)
        {
            Input_Array = Input_Image;
            var outp = Sub_Sampling.Pooling_img(Input_Image, Coeff, Biased_Weight);

            
           Output_Array = outp;
            return Output_Array;
        }

        public double[] PreTrain(double[][] Second_Deri_Error_from_Next_Layer)//C5->S4
        {
            var Second_Deri_Error_array = new double[Output_Width * Output_Width];


            for (int i = 0; i < Second_Deri_Error_from_Next_Layer.Length; i++)
            {
                for (int j = 0; j < Second_Deri_Error_array.Length; j++)
                {
                    Second_Deri_Error_array[j] += Second_Deri_Error_from_Next_Layer[i][j];
                }
            }           
            return PreTrain(Second_Deri_Error_array);

        }
        public double[] PreTrain(double[] Second_Deri_Error_from_Next_Layer)
        {
            Pretrain_Bias_Coeff_And_Error_to_Previous(Second_Deri_Error_from_Next_Layer);
            return Second_Deri_Error_to_Previous_Layer;
        }

        public double[] Back_Propagation(double[][] BP_Error_from_Next_Layer)//C5->S4
            //each neural has one-Dimensional array []BP_errors
        {

            var Error_array = new double[Output_Width * Output_Width];


            for (int i = 0; i < BP_Error_from_Next_Layer.Length; i++)
            {
                for (int j = 0; j < Error_array.Length; j++)
                {
                    Error_array[j] += BP_Error_from_Next_Layer[i][j];
                }
            }
            Update_Bias_Coeff_And_Error_to_Previous(Error_array);
            return BP_Error_to_Previous_Layer;

        }

        public double[] Back_Propagation(double[] BP_ERROR)
        {
            Update_Bias_Coeff_And_Error_to_Previous(BP_ERROR);
            return BP_Error_to_Previous_Layer;
        }


        private void Update_Bias_Coeff_And_Error_to_Previous(double[] error_array)
        {
            double res = 0, coef = 0;
            BP_Error_to_Previous_Layer = new double[Input_Array.Length];

            for (int i = 0; i < Output_Width; i++)
            {
                for (int j = 0; j < Output_Width; j++)
                {
                    int k = i * Output_Width + j;

                   // res += error_array[k] * 2 / 3 * (1 - Output_Array[k] * Output_Array[k]);
                    res += error_array[k] * Deri_ATanhSa(Output_Array[k]);
                    double temp = 0;
                    for (int q = 0; q < 2; q++)
                    {
                        for (int r = 0; r < 2; r++)
                        {
                            int idx = (i * 2 + q) * Input_Width + (j * 2 + r);

                            // BP_Error_to_Previous_Layer[idx] = error_array[k]  *2/3* (1 - Output_Array[k] * Output_Array[k]) * Coeff;
                            BP_Error_to_Previous_Layer[idx] = error_array[k] * Deri_ATanhSa(Output_Array[k]) * Coeff;
                            //BP_Error_to_Previous_Layer[idx] = error_array[k]  * Coeff;
                            // BP_Error_to_Previous_Layer[idx] = error_array[k] * Deri_Sigmoid(Output_Array[k]) * Coeff;
                            temp += Input_Array[idx];
                        }
                    }
                    coef += error_array[k] * Deri_ATanhSa(Output_Array[k]) * temp;
                }
            }
            Coeff = Coeff - Learning_Rate_coeff * coef;
            Biased_Weight = Biased_Weight - Learning_Rate_bias * res;
        }

        private void Pretrain_Bias_Coeff_And_Error_to_Previous(double[] Second_deri_error_array)
        {
            double res = 0, coef = 0;
            Second_Deri_Error_to_Previous_Layer = new double[Input_Array.Length];

            for (int i = 0; i < Output_Width; i++)
            {
                for (int j = 0; j < Output_Width; j++)
                {
                    int k = i * Output_Width + j;

                    // res += error_array[k] * 2 / 3 * (1 - Output_Array[k] * Output_Array[k]);
                    double deri = Deri_ATanhSa(Output_Array[k]);
                    res += Second_deri_error_array[k] *deri*deri ;
                   
                    for (int q = 0; q < 2; q++)
                    {
                        for (int r = 0; r < 2; r++)
                        {
                            int idx = (i * 2 + q) * Input_Width + (j * 2 + r);

                           
                            Second_Deri_Error_to_Previous_Layer[idx] = Second_deri_error_array[k] * deri*deri * Coeff*Coeff;

                            coef += Second_deri_error_array[k] * deri * deri * Input_Array[idx] * Input_Array[idx];
                        }
                    }
                   
                }
            }
            hkk_coeff += coef;
            hkk_bias += res;
        }

        public void Evaluate_Learning_Rate(double Sample_Count)
        {
           
            Learning_Rate_coeff= TAU / (hkk_coeff / Sample_Count + MU);

            Learning_Rate_bias = TAU / (hkk_bias / Sample_Count + MU);

        }
        public void Zeroing_HKKs()
        {
           
                hkk_coeff = 0.0;
            
            hkk_bias = 0.0;

        }

    }
}
