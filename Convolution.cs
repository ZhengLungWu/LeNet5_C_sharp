using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet5
{
    class Convolution
    {      
        public static double[] Cov_array(double[] Src, double[] Kernel, double bias)
        {
            int root = (int)Math.Sqrt(Src.Length);

            var res = new List<double>();
            var K_len = (int)Math.Sqrt(Kernel.Length);

            for (int i = 0; i <= root - K_len; i++)
            {
                for (int j = 0; j <= root - K_len; j++)
                {
                    double temp = 0;
                    for (int p = 0; p < K_len; p++)
                    {
                        for (int q = 0; q < K_len; q++)
                        {
                            temp += Src[(i + p) * root + j + q] * Kernel[p * K_len + q];
                        }
                    }
                    res.Add(ATanhSa(temp  + bias));
                    //res.Add((temp + bias));
                }
            }
            return res.ToArray();
        }

        private static double ATanhSa(double val)
        {

            return 1.7159 * Math.Tanh( 2.0 / 3.0 * val);
        }
       
        
    }


}
