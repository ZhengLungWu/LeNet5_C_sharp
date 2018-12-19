using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet5
{
    public class scalar_neu : neural_
    {
       // public double Coeff;
        public double[] BP_Error_to_Previous_Layer;
        public double[] Weights_inputs;
        private int amount_input;
        private double sum;
        public double[] hkk_weights;
        public double hkk_bias=0.0;
        public double[] Learning_Rate_Weights;
        public double Learning_Rate_Bias;
        //private double Epsilon;

        public scalar_neu(int AmountInThisLayer,int amount_input)
        {
            this.amount_input = amount_input;
            NeuralAmountThisLayer = AmountInThisLayer;
            NeuralAmountPreviousLayer = amount_input;
            Weights_Initialize();
        }

        protected override void Weights_Initialize()
        {
            Weights_inputs = Weight.RandomWeights(amount_input, NeuralAmountPreviousLayer);            
            //Coeff = Weight.Random(NeuralAmountThisLayer);
            Biased_Weight = Weight.Random(NeuralAmountPreviousLayer);
            hkk_weights = new double[amount_input];
            Learning_Rate_Weights = new double[amount_input];
            Learning_Rate_Bias = 0.0;
           
        }

        public double Evaluation(double[] Input_Array)
        {
            this.Input_Array = Input_Array;
            var a_b= Input_Array.Zip(Weights_inputs, (a, b) => a * b).ToArray();
            sum = a_b.Sum();
            sum = sum  + Biased_Weight;
            var tanh =ATanhSa(sum);
            Output_Array = new double[] { tanh };
            return tanh;
        }

        public double[] PreTrain()//Second_Deri_Error_from_Result=d^2Error/(d_this_output)^2
        {
            double Second_Deri_Error_to_unit = 0.0;
            double deri = Deri_ATanhSa(Output_Array[0]);
            //double second_deri = Second_Deri_ATanhSa(Output_Array[0]);
            double Second_Deri_Loss_Function = 2.0;

            Second_Deri_Error_to_unit= Second_Deri_Loss_Function*deri*deri; //+ second_deri * Error_from_Result;
            for (int i = 0; i < Input_Array.Length; i++)
            {
                hkk_weights[i] += Second_Deri_Error_to_unit * Input_Array[i] * Input_Array[i];
            }
            hkk_bias += Second_Deri_Error_to_unit * 1.0 * 1.0;

            return Second_Deri_Error_Back(Second_Deri_Error_to_unit);
        }

        public void Back_Propagation(double[][] BP_Error_from_Next_Layer)
        {
            throw new NotImplementedException();
        }

        public double[] Back_Propagation(double Error_from_Result)
        {
            double s = Output_Array[0];

            //double de_dw = Error_from_Result * s * (1 - s) * sum;//sigmoid
            double[] de_dwi = new double[Weights_inputs.Length];
            for (int i = 0; i < Weights_inputs.Length; i++)
            {
                de_dwi[i] = Error_from_Result * Deri_ATanhSa(s) * Input_Array[i];


            }
          
            BP_Error_to_Previous_Layer = Error_Back(Error_from_Result, Deri_ATanhSa(s));

            Biased_Weight = Biased_Weight - Learning_Rate_Bias*Error_from_Result* Deri_ATanhSa(s);//tanh
            Weights_Update(de_dwi);
            return BP_Error_to_Previous_Layer;
        }
        private double[] Error_Back(double Error_from,double deri_)
        {
            var vals = new double[Input_Array.Length];
            for (int i = 0; i < vals.Length; i++)
            {
                vals[i] = Error_from * deri_ * Weights_inputs[i];
            }
            return vals;
        }
        private double[] Second_Deri_Error_Back(double Second_Deri_Error_to_unit)//pre-train
        {
            var vals = new double[Input_Array.Length];
            for (int i = 0; i < vals.Length; i++)
            {
                vals[i] = Second_Deri_Error_to_unit*Weights_inputs[i] * Weights_inputs[i];
            }
            return vals;
        }




        private void Weights_Update(double[] de_dw)
        {
            for (int i = 0; i < Weights_inputs.Length; i++)
            {
                Weights_inputs[i] = Weights_inputs[i] - Learning_Rate_Weights[i] * de_dw[i];

            }
        }


        public void Evaluate_Learning_Rate(double Sample_Count)
        {
            for (int i = 0; i < Weights_inputs.Length; i++)
            {
                Learning_Rate_Weights[i] = TAU / (hkk_weights[i] / Sample_Count + MU);



            }
            Learning_Rate_Bias = TAU / (hkk_bias / Sample_Count + MU);

        }
        public void Zeroing_HKKs()
        {
            for (int i = 0; i < Weights_inputs.Length; i++)
            {
                hkk_weights[i] = 0.0;
            }
            hkk_bias = 0.0;

        }



    }
}
