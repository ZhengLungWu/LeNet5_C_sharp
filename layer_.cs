using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet5
{
    public delegate void Layer_Done_Handler(double[][] Output_results);
    public delegate void Image_Hander(double[] IMAGE);


    public abstract class layer_
    {
        
        // neural_[] neurals_;
        protected readonly bool[,] exec_array23 = new bool[16, 6]
              {
                    {true ,true ,true ,false,false,false},
                    {false,true ,true ,true ,false,false},
                    {false,false,true ,true ,true ,false},
                    {false,false,false,true ,true ,true },

                    {true ,false,false,false,true ,true },
                    {true ,true ,false,false,false,true },
                    {true ,true ,true ,true ,false,false},
                    {false,true ,true ,true ,true ,false},

                    {false,false,true ,true ,true ,true },
                    {true ,false,false,true ,true ,true },
                    {true ,true ,false,false,true ,true },
                    {true ,true ,true ,false,false,true },

                    {true ,true ,false,true ,true ,false},
                    {false,true ,true ,false,true ,true },
                    {true ,false,true ,true ,false,true },
                    {true ,true ,true ,true ,true ,true }
              };
        //public abstract void Set_Learning_Rate(double rate);
    }

    public enum Mapping_type {one2one,one2All,All2one, All2All, Specified};
  

  
}
