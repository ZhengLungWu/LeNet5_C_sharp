using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet5
{
   
    public abstract class Neural
    {
       // public double[] Input_Neural_Weights;
        public double Biased_Weight;
       // public double[] Output;
        protected int Input_Width,Output_Width;
        //protected List<double[]> Input_Array_List;
        public double[] Input_Array,Output_Array;
        public double[] BP_Error_to_Previous_Layer;// d(E)/d(input)
        public double[][] BP_Error_from_Next_Layer;//delta
        public  double Leraning_Rate=0.002;
        //public double[]BackProgError_
       // public readonly Layer BelongedLayer;
       
        public Neural[] Previous_Neurals,Next_Neurals;
        protected int Input_Amount, NeuralAmountThisLayer;

       
       
        protected abstract void Weights_Initialize();
        public abstract void Get_Previous_Layer_Neurals(Neural[]neurals);
        public abstract void Get_Next_Layer_Neurals(Neural[] neurals);
 
        public abstract void Evaluation(List<double[]> Input_Images);
        public abstract void Evaluation(double[] Input_Images);

        public abstract void Back_Propagate();
 
        protected double Sigmoid(double sum)
        {
            return 1 / (1 + Math.Exp(-sum));
        }
    }

  

   
   
}





