
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;


namespace LeNet5
{
    public abstract class Layer
    {
        //public bool IsLastLayer;
    }

        public abstract class N_Layer:Layer
    {
        public N_Layer Previous_Layer;
        public N_Layer Next_Layer;
        public Neural[] Neurals;

       

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


        protected abstract void GenerateNeurals(int amount);
        //public abstract void Proc(double[]input_img);
       // public abstract void Back_Proc();
        public N_Layer(int Neural_Amount)
        {
            GenerateNeurals(Neural_Amount);
        }
       // public abstract void SetConnection(Layer Previous_Layer, Layer Next_Layer);
    }

    public class C1 : N_Layer
    {
       
        public C1(int Neural_Amount) :base(Neural_Amount)
        {
        }

        protected override void GenerateNeurals(int amount)
        {
            Neurals = new Conv_Neural[amount];
            for (int i = 0; i < Neurals.Length; i++)
            {
                Neurals[i] = new Conv_Neural(32, amount, 28);
            }           
        }

        public void SetConnection(N_Layer Next_Layer)
        {
            for (int i = 0; i < Neurals.Length; i++)
            {
                Neurals[i].Next_Neurals =new Sub_Neural[] { Next_Layer.Neurals[i] as Sub_Neural};
            }           
        }

        public  void Proc(double[]SRC)
        {            
            for (int i = 0; i < Neurals.Length; i++)
            {
               Neurals[i].Evaluation(SRC);
            }
        }

        public  void Back_Proc()
        {
            for (int i = 0; i < Neurals.Length; i++)
            {
                Neurals[i].Back_Propagate();
            }

        }

    }
    public class S2 : N_Layer
    {
        public S2(int Neural_Amount) : base(Neural_Amount)
        {
        }
        protected override void GenerateNeurals(int amount)
        {
            Neurals = new Neural[amount];
            for (int i = 0; i < Neurals.Length; i++)
            {
               
                Neurals[i] = new Sub_Neural(28, amount, 14);
            }
            //throw new NotImplementedException();
        }
        public  void SetConnection(N_Layer Previous_Layer, N_Layer Next_Layer)
        {

            for (int i = 0; i < Neurals.Length; i++)
            {
                Neurals[i].Previous_Neurals = new Neural[] {Previous_Layer.Neurals[i]};
                Neurals[i].Next_Neurals = PartialConnectionS2(i);

            }


        }

        private Neural[] PartialConnectionS2(int index)
        {
            var list = new List<Neural>();
            for (int i = 0; i < 6; i++)
            {
                var ar = exec_array23[index, i];
                if (ar)
                {
                    list.Add(Next_Layer.Neurals[i]);
                }
            }
            return list.ToArray();
        }

        public  void Proc()
        {
            for (int i = 0; i < Neurals.Length; i++)
            {
                var img = Neurals[i].Previous_Neurals[0].Output_Array;
                Neurals[i].Evaluation(img);                  
            }
        }

        public void Back_Proc()
        {
            for (int i = 0; i < Neurals.Length; i++)
            {
                Neurals[i].Back_Propagate();
            }

        }

    }
    public class C3: N_Layer
    {
        

        public C3(int Neural_Amount) : base(Neural_Amount)
        {
        }
        protected override void GenerateNeurals(int amount)
        {
            Neurals = new Neural[amount];
            for (int i = 0; i < Neurals.Length; i++)
            {
                Neurals[i] = new Sub_Neural(14, amount, 10);
            }
            //throw new NotImplementedException();
        }
        public  void SetConnection(N_Layer Previous_Layer, N_Layer Next_Layer)
        {

            for (int i = 0; i < Neurals.Length; i++)
            {
                Neurals[i].Previous_Neurals=PartialConnectionC3(i);
                Neurals[i].Next_Neurals =new Neural[] { Next_Layer.Neurals[i] };

            }


        }

        private Neural[] PartialConnectionC3(int index)
        {
            var list = new List<Neural>();
            for (int i = 0; i < 6; i++)
            {
                var ar = exec_array23[index,i];
                if (ar)
                {
                    list.Add(Previous_Layer.Neurals[i]);
                }
            }
            return list.ToArray();
        }

        public void Proc()
        {
            for (int i = 0; i < Neurals.Length; i++)
            {
                var imgs = Neurals[i].Previous_Neurals.Output_Array;
                Neurals[i].Evaluation(img);
            }
        }

        public void Back_Proc()
        {
            for (int i = 0; i < Neurals.Length; i++)
            {
                Neurals[i].Back_Propagate();
            }

        }

    }

    public class S4 : N_Layer
    {
        public S4(int Neural_Amount) : base(Neural_Amount)
        {
        }
        protected override void GenerateNeurals(int amount)
        {
            Neurals = new Neural[amount];
            for (int i = 0; i < Neurals.Length; i++)
            {
                Neurals[i] = new Sub_Neural(10, amount, 5 );
            }
           
        }
        public  void SetConnection(N_Layer Previous_Layer, N_Layer Next_Layer)
        {
            for (int i = 0; i < Neurals.Length; i++)
            {
                Neurals[i].Previous_Neurals = new Neural[] { Previous_Layer.Neurals[i] };
                Neurals[i].Next_Neurals =  Next_Layer.Neurals;

            }
        }

    }
    public class C5 : N_Layer
    {
        public C5(int Neural_Amount) : base(Neural_Amount)
        {
        }
        protected override void GenerateNeurals(int amount)
        {
            Neurals = new Neural[amount];
            for (int i = 0; i < Neurals.Length; i++)
            {
                Neurals[i] = new Sub_Neural(5,120,1);
            }
            
        }

        public  void SetConnection(N_Layer Previous_Layer, N_Layer Next_Layer)
        {
            for (int i = 0; i < Neurals.Length; i++)
            {
                Neurals[i].Previous_Neurals = Previous_Layer.Neurals;
                Neurals[i].Next_Neurals = Next_Layer.Neurals;

            }
        }
    }

    public class F6 : N_Layer
    {
        public F6(int Neural_Amount) : base(Neural_Amount)
        {
        }

       
        protected override void GenerateNeurals(int amount)
        {
            Neurals = new Neural[amount];
            for (int i = 0; i < Neurals.Length; i++)
            {
                Neurals[i] = new Scalar_Neural( 84);
            }          
        }
        private  void SetConnection(N_Layer Previous_Layer, Output_Layer Next_Layer)
        {
            for (int i = 0; i < Neurals.Length; i++)
            {
                Neurals[i].Previous_Neurals = Previous_Layer.Neurals;
                Neurals[i].Next_Neurals = null;

            }
        }
    }
   
    
}