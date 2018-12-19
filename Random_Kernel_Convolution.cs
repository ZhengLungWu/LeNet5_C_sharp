using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet5
{
    class Random_Kernel
    {

        public static List<double[]> GenerateKernel(int Width,int Height,int Amount)
        {
            var list = new List<double[]>();         
            var rdm = new Random();
            int i = 0;
            while (i < Amount)
            {
                double[] arr = new double[Width * Height];
                for (int j = 0; j < arr.Length; j++)
                {
                   // arr[j] =rdm.Next(-1,1);
                    arr[j]= (rdm.NextDouble() - 0.5) * (4.8 / (Width*Height));
                }              
                    list.Add(arr);
               // Console.WriteLine($"kernel:{arr[0]},{arr[1]},{arr[2]},{arr[5]}");
                    i++;                
            }
            return list;
        }
        public static double[] GenerateKernel(int Width, int Height)
        {
           
            var rdm = new Random();
         
            
                double[] arr = new double[Width * Height];
                for (int j = 0; j < arr.Length; j++)
                {
                    // arr[j] =rdm.Next(-1,1);
                    arr[j] = (rdm.NextDouble() - 0.5) * (4.8 / (Width * Height));
                }

            // Console.WriteLine($"kernel:{arr[0]},{arr[1]},{arr[2]},{arr[5]}");

            return arr;
        }
    }
}
