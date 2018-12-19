using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet5
{
    public enum Squashing {Tanh,Sig };

    public abstract class neural_
    {
        protected int NeuralAmountThisLayer,NeuralAmountPreviousLayer;
        public double Biased_Weight;
        public double[] Input_Array;
        public double[] Output_Array;   
        public int Input_Width, Output_Width;
        //public double Learning_Rate =0.00001;
        public static double TAU = 0.00001;
        public static double MU = 0.005;
        
        private readonly static double A = 1.7159,S=2.0/3.0;
        protected abstract void Weights_Initialize();

        

        protected static double ATanhSa(double a)
        {
            return A * (Math.Tanh(a*S ));
        }

        protected static double Deri_ATanhSa(double val)
        {          
            return A*S-val*val*S/A;
        }
        protected static double Second_Deri_ATanhSa(double val)
        {
            return -2*S*S*val+2*S*S*val*val*val/A/A;
        }
       

    }

    
}



