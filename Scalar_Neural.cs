using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet5
{
    public class Scalar_Neural : Neural
    {
        public double Coeff;
        public double Error;

        public Scalar_Neural(int AmountInThisLayer)
        {

        }
        protected override void Weights_Initialize()
        {
            Coeff = Weight.Random(NeuralAmountThisLayer);
            Biased_Weight = Weight.Random(NeuralAmountThisLayer);
        }

        public override void Evaluation(List<double[]> Input_Images)
        {
            var s = (from x in Input_Images select x[0]).ToArray();
            double sum = s.Sum();
            sum = sum * Coeff + Biased_Weight;
            sum = Sigmoid(sum);
            Output_Array = new double[] { sum };
        }
        public override void Evaluation(double[] Input_Image)
        {          
            double sum = Input_Image.Sum();
            sum = sum * Coeff + Biased_Weight;
            sum = Sigmoid(sum);
            Output_Array = new double[] { sum };
        }

        public override void Get_Next_Layer_Neurals(Neural[] neurals)
        {
            throw new NotImplementedException();
        }
        public override void Get_Previous_Layer_Neurals(Neural[] neurals)
        {
            Previous_Neurals = neurals;
        }
        public void Get_Error(double Error)
        {
            this.Error = Error;
        }
        public override void Back_Propagate()
        {
            double s = Output_Array[0];
            var q = (from x in Previous_Neurals select x.Output_Array[0]).ToArray();
            double sum = q.Sum();
            double de_dw = Error * s * (1 - s) * sum;
            BP_Error_to_Previous_Layer = new double[Previous_Neurals.Length];
            for (int i = 0; i < BP_Error_to_Previous_Layer.Length; i++)
            {
                BP_Error_to_Previous_Layer[i] = Error * s * (1 - s) * Coeff;
            }
            Biased_Weight = Biased_Weight - Error * s * (1 - s);
            Coeff = Coeff - Leraning_Rate * de_dw;
        }
    }

   
}
