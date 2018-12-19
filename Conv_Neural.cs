using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet5
{
    public class Conv_Neural : Neural
    {
        public double[] Kernel;//random generated kernel
        private double[] Kernel_old;
        public int Kernel_Width;


        public Conv_Neural(int Width_Image, int AmountInThisLayer, int outputWidth) 
        {
            Input_Width = Width_Image;
            Output_Width = outputWidth;
            NeuralAmountThisLayer = AmountInThisLayer;
            Weights_Initialize();
        }

        protected override void Weights_Initialize()
        {
            Kernel = Weight.RandomWeights(25, NeuralAmountThisLayer);
            Biased_Weight = Weight.Random(NeuralAmountThisLayer);
        }

        public override void Evaluation(List<double[]> Input_Images)
        {
            //Input_Array_List = Input_Images;
            if (Input_Images.Count > 0)
            {
                Input_Array = new double[Input_Images[0].Length];
                for (int i = 0; i < Input_Images.Count; i++)
                {
                    for (int j = 0; j < Input_Images[i].Length; j++)
                    {
                        Input_Array[j] += Input_Images[i][j];
                    }
                }
                Output_Array = Convolution.Cov_array(Input_Array, Kernel, Biased_Weight);
            }
        }

        public override void Evaluation(double[] Input_image)
        {          
                Input_Array = new double[Input_image.Length];
                
                    for (int j = 0; j < Input_image.Length; j++)
                    {
                        Input_Array[j] += Input_image[j];
                    }               
                Output_Array = Convolution.Cov_array(Input_Array, Kernel, Biased_Weight);            
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
            //LINQ style
            BP_Error_from_Next_Layer = (from x in Next_Neurals select x.BP_Error_to_Previous_Layer).ToArray();
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

        protected virtual void UpdateKernel(double[] Error_Array)
        {
            //dOutput[a,b]/dw[x,y]
            Kernel_old = Kernel;
            int kernel_Width = Input_Width - Output_Width + 1;
            for (int x = 0; x < kernel_Width; x++)
            {
                for (int y = 0; y < kernel_Width; y++)
                {
                    double do_dw = 0;
                    for (int i = 0; i < Output_Width; i++)
                    {
                        for (int j = 0; j < Output_Width; j++)
                        {
                            do_dw += Input_Array[(i + x) * Input_Width + (j + y)] * (Error_Array[i * Output_Width + j]);
                        }
                    }
                    Kernel[x * kernel_Width + y] = Kernel[x * kernel_Width + y] - Leraning_Rate * do_dw;
                }
            }
        }
        protected virtual void Update_Bias(double[] Error_Array)
        {
            var Sum = Error_Array.Sum();
            Biased_Weight = Biased_Weight - Leraning_Rate * Sum;
        }


        protected virtual void UpdateError_to_Previous(double[] Error_Array)
        {
            int kernel_Width = Input_Width - Output_Width + 1;
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
                                BP_Error_to_Previous_Layer[(a + x) * Input_Width + b + y] += Error_Array[x * Output_Width + y] * Kernel[a * kernel_Width + b];
                            }
                        }
                    }
                }
            }
        }
    }

  
}
