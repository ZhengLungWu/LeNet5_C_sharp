//using Microsoft.Analytics.Interfaces;
//using Microsoft.Analytics.Types.Sql;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;

// this section is refered from https://github.com/patrickmeiring/LeNet
namespace LeNet5
{
    class Weight
    {

        private static ThreadLocal<Random> randomWrapper = new ThreadLocal<Random>(()=>new Random());

        
        public static double[] RandomWeights(int Amount,int FanIn)
        {
          
            var res = new double[Amount];
            for (int i = 0; i < res.Length; i++)
            {
                
                res[i] = (randomWrapper.Value.NextDouble() - 0.5) * (4.8 / FanIn);
            }
            return res;
        }
       
        public static double Random(int FanIn)
        {
            
            return (randomWrapper.Value.NextDouble() - 0.5) * (4.8 / FanIn);
        }


    }
}
