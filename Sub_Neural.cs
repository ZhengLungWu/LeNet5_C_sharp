using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet5
{
    public class Sub_Neural : Neural
    {
        public double Coeff;

        public Sub_Neural(int Width_Image, int AmountInThisLayer, int outputWidth)
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
        }

        public override void Evaluation(List<double[]> Input_Images)
        {

            double[] tota = new double[Input_Width * Input_Width];
            foreach (var q in Input_Images)
            {
                for (int i = 0; i < tota.Length; i++)
                {
                    tota[i] += q[i];
                }
            }

            var outp = Sub_Sampling.Pooling_img(tota, Coeff, Biased_Weight);

            Output_Array = new double[outp.Length];
            for (int j = 0; j < outp.Length; j++)
            {
                Output_Array[j] = Sigmoid(outp[j]);
            }
            //throw new NotImplementedException();
        }

        public override void Evaluation(double[] Input_Image)
        {           
            var outp = Sub_Sampling.Pooling_img(Input_Image, Coeff, Biased_Weight);

            Output_Array = new double[outp.Length];
            for (int j = 0; j < outp.Length; j++)
            {
                Output_Array[j] = Sigmoid(outp[j]);
            }
            //throw new NotImplementedException();
        }


        public override void Get_Next_Layer_Neurals(Neural[] neurals)
        {
            Next_Neurals = neurals;

            //throw new NotImplementedException();
        }
        public override void Get_Previous_Layer_Neurals(Neural[] neurals)
        {
            Previous_Neurals = neurals;
            //throw new NotImplementedException();
        }

        public override void Back_Propagate()
        {
            BP_Error_from_Next_Layer = (from x in Next_Neurals select x.BP_Error_to_Previous_Layer).ToArray();
            var Error_array = new double[Output_Width * Output_Width];


            for (int i = 0; i < BP_Error_from_Next_Layer.Length; i++)
            {
                for (int j = 0; j < Error_array.Length; j++)
                {
                    Error_array[j] += BP_Error_from_Next_Layer[i][j];
                }
            }
            Update_Bias_Coeff_And_Error_to_Previous(Error_array);

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

                    res += error_array[k] * Output_Array[k] * (1 - Output_Array[k]);
                    double temp = 0;
                    for (int q = 0; q < 2; q++)
                    {
                        for (int r = 0; r < 2; r++)
                        {
                            int idx = (i * 2 + q) * Output_Width + (j * 2 + r);

                            BP_Error_to_Previous_Layer[idx] = error_array[k] * Output_Array[k] * (1 - Output_Array[k]) * Coeff;
                            temp += Input_Array[idx];
                        }
                    }
                    coef += error_array[k] * Output_Array[k] * (1 - Output_Array[k]) * temp;
                }
            }
            Coeff = Coeff - Leraning_Rate * coef;
            Biased_Weight = Biased_Weight - Leraning_Rate * res;
        }




    }

    

}
