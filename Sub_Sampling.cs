using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet5
{
    class Sub_Sampling
    {
        public static double[] Pooling_img(double[] src,double coeff,double bias)
        {
           

            var root =(int) Math.Sqrt(src.Length);
            int new_root =(int)(root*0.5);
            List<double> res = new List<double>();
            double temp;
            int heigh = new_root, width = new_root;
            for (int i = 0; i < root; i+=2)
            {
                for (int j = 0; j < root; j+=2)
                {                
                    temp = src[j + i * root] + src[j + 1 + i * root] + src[j + (i + 1) * root] + src[j + 1 + (i + 1) * root];
                    //temp *= 0.25;
                    temp = temp * coeff + bias;
                    res.Add(Tanh(temp));
                    //res.Add(Sigmoid(temp));
                    //res.Add(temp);
                }
            }
           // Console.WriteLine($"length of res:{res.Count}");
            return res.ToArray();
        }

      
        public static byte[] Pooling_img(byte[] src)
        {
            double[] dsrc = new double[src.Length];
            for (int i = 0; i < dsrc.Length; i++)
            {
                dsrc[i] = src[i];
            } 
            dsrc= Pooling_img(dsrc, 1.0, 0);
            src = new byte[dsrc.Length];
            for (int i = 0; i < dsrc.Length; i++)
            {
                src[i]=(byte)dsrc[i];
               // Console.WriteLine($"dsrc{dsrc[i]};src{src[i]}");
            }
            return src;
        }

        private static double Sigmoid(double input)
        {

            return 1.0 / (1.0 + Math.Exp(-input));
        }

        private static double Tanh(double val)
        {

            return 1.7159 * Math.Tanh( 2.0 / 3.0 * val);
        }
    }
}
