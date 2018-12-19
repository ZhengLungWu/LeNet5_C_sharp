using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;

namespace LeNet5
{
   public class Output_Layer
    {
      
        public double[][] NumArray=ASCII_Numbers();
        public double[] Result_img;
        public int Result_int;
        public double Error;
        private readonly double Hand_picked_j = -0.9;
        private double[] Input_F6;
        public double[] Error_to_F6;// Amount:84 ,Value:dE/dF6i

        public  void Proc(double[]Input_F6_Layer)
        {
            Input_F6 = Input_F6_Layer;
            double res =double.MaxValue;
            int index = -1;
            for (int i = 0; i < NumArray.Length; i++)
            {
                var distance= RBF(NumArray[i], Input_F6_Layer);
               // Console.WriteLine($"Distance of {i}is:{distance}");
              
                if (res > distance)
                {
                    res = distance;
                    index = i;
                }
            }
            Result_int = index;
            Result_img = NumArray[index];
        }

        public double Error_(int Correct_Number)
        {
            double res = 0.0;
                          
            res+= RBF(NumArray[Correct_Number], Input_F6);//correct
            if (Correct_Number != Result_int)
            {
               // res += Penalities();
            }
            //res /= NumArray[Correct_Number].Length;
            Error = res;
            return res; 
            
        }
        public void Back_Proc(int Correct_Number)
        {
            Error_to_F6 = new double[Input_F6.Length];
            double len = Input_F6.Length;
            for (int i = 0; i < Input_F6.Length; i++)
            {
                Error_to_F6[i] =2.0*(Input_F6[i] - NumArray[Correct_Number][i]);
                
            }



        }

        private double Penalities()
        {
            double temp = 0.0;

            
            temp += Math.Exp(-RBF(NumArray[Result_int], Input_F6));

            temp += Math.Log10(Math.Exp(-Hand_picked_j) + temp);

            return temp;

        }



        private static double RBF(double[] NumberArray, double[] Inputted_img)
        {
            double temp = 0.0;
            for (int i = 0; i < NumberArray.Length; i++)
            {
                temp += (Inputted_img[i] - NumberArray[i]) * (Inputted_img[i] - NumberArray[i]);
            }

            return temp;
        }
        private static readonly Font Numberfont = new Font("Lucida Console", 14.0f, FontStyle.Bold, GraphicsUnit.Pixel);

            public static double[][] ASCII_Numbers()
            {
                var result = new double[10][];
                // string[] Numbers = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };

                for (int i = 0; i < 10; i++)
                {
                    Bitmap m = new Bitmap(7, 12);
                    Graphics graph = Graphics.FromImage(m);
                    graph.TextRenderingHint = System.Drawing.Text.TextRenderingHint.SingleBitPerPixelGridFit;
                    graph.FillRectangle(new SolidBrush(Color.White), new Rectangle(0, 0, m.Width, m.Height));
                    if (i == 2)
                    {
                        graph.DrawString(i.ToString(), Numberfont, new SolidBrush(Color.Black), -2, 0);
                    }
                    else
                    {
                        graph.DrawString(i.ToString(), Numberfont, new SolidBrush(Color.Black), -3, 0);
                    }
                    graph.Flush();
                  //  m.Save($"D:\\{i}num.jpg");
                    result[i] = Image_Processing.ToDoubleArray(m);
                    graph.Dispose();
                }
                return result;

            }

        //----------------------------------------------
        //pre-train section
        //----------------------------------------------

        private static double Second_Deri_Loss_Function
        {
            get
            {
                return 2.0;
            }
        }


        private void PreTrain()
        {



        }
        

    }
}
