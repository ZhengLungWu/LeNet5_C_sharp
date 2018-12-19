using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet5
{
    class Back_propagation
    {
        public static double CalculateEdDistance(double output, double nominal)
        {
            return 0.5 * (output - nominal) * (output - nominal);
        }

        public static double Loss_calculation(double Z, double result)
        {
            return -Z * Math.Log10(result) - (1 - Z) * Math.Log10(1 - result);            
        }
        public static double Loss_MSE(double[] actual,double[] nominal)
        {
            double res = 0;
            for (int i = 0; i < nominal.Length; i++)
            {
                res += (actual[i] - nominal[i]) * (actual[i] - nominal[i]);
            }
            res /= nominal.Length;

            return res;
        }



        public static double Evaluate(double GetIn_Error)
        {
            return 0;
        }

    }
}
