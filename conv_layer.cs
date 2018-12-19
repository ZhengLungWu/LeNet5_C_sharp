using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet5
{
    public class Conv_layer : layer_
    {
        public conv_neu[] neurals;
        public double[][] Output_Proc_Result;
        public double[][] error_back_to_previous,Second_Deri_error_back_to_previous;
        int Input_img_width;
       

        public Conv_layer(int Input_img_width, int neural_amount_previous_layer,int neural_amount, int Output_img_width)
        {
            this.Input_img_width = Input_img_width;
            neurals = new conv_neu[neural_amount];
            for (int i = 0; i < neurals.Length; i++)
            {
                neurals[i] = new conv_neu(Input_img_width,neural_amount_previous_layer, neural_amount, Output_img_width);
            }
        }

        public void Proc(double[] Input_Image)
        {
            List<double[]> temp_saver = new List<double[]>();
            for (int i = 0; i < neurals.Length; i++)
            {
                var array = neurals[i].Evaluation(Input_Image);
                temp_saver.Add(array);
            }
            Output_Proc_Result = temp_saver.ToArray();
           
        }
       




        public void ProcIntermed(double[][] result_Prev_layer, bool FullConnection)
        {
            List<double[]> temp_saver = new List<double[]>();
            if (FullConnection)
            {
                double[] Matrix_sum = new double[Input_img_width * Input_img_width];
                for (int j = 0; j < result_Prev_layer.Length; j++)
                {
                    Matrix_sum = Matrix_sum.Zip(result_Prev_layer[j], (a, b) => a + b).ToArray();
                    //var array = neurals[i].Evaluation(result_Prev_layer[j]);
                }
                for (int i = 0; i < neurals.Length; i++)
                {
                    var array = neurals[i].Evaluation(Matrix_sum);
                    temp_saver.Add(array);
                }
                Output_Proc_Result = temp_saver.ToArray();
               
            }
            else
            {
               for (int i = 0; i < exec_array23.GetUpperBound(0) + 1; i++)
                {
                    double[] Matrix_sum = new double[Input_img_width * Input_img_width];

                    for (int j = 0; j < exec_array23.GetUpperBound(1) + 1; j++)
                    {
                        if (exec_array23[i, j])
                        {
                            Matrix_sum = Matrix_sum.Zip(result_Prev_layer[j], (a, b) => a + b).ToArray();
                        }
                    }
                    var array = neurals[i].Evaluation(Matrix_sum);
                    temp_saver.Add(array);
                }
                Output_Proc_Result = temp_saver.ToArray();
            }
        }
        public void BP_Proc(double[][] Error_back)// 1-1 mapping
        {
            BP_internal(Error_back);
        }
        public void BP_Proc(double[] Error_back)
        {           
                BP_internal(Error_back);          
        }
        private void BP_internal(double[][] Error_back)
        {
            error_back_to_previous = new double[neurals.Length][];


            for (int i = 0; i < neurals.Length; i++)
            {
                error_back_to_previous[i] = neurals[i].Back_Propagation(Error_back[i]);                
            }
        }

        private void BP_internal(double[] Error_back)//F6->C5
        {
            error_back_to_previous = new double[neurals.Length][];
            for (int i = 0; i < neurals.Length; i++)
            {
                error_back_to_previous[i] = neurals[i].Back_Propagation(Error_back);
            }
        }

        public void PreTraining(double[][] Second_Deri_error_back)//1-1 mapping
        {
            Second_Deri_error_back_to_previous = new double[neurals.Length][];

            for (int i = 0; i < neurals.Length; i++)
            {
                Second_Deri_error_back_to_previous[i] = neurals[i].PreTrain(Second_Deri_error_back[i]);
            }
        }

        public void PreTraining(double[] Second_Deri_error_back)//F6->C5
        {
            Second_Deri_error_back_to_previous = new double[neurals.Length][];
            for (int i = 0; i < neurals.Length; i++)
            {
                Second_Deri_error_back_to_previous[i] = neurals[i].PreTrain(Second_Deri_error_back);
            }
        }

        public void Evaluate_Learning_rate(double sample_count)
        {
            for (int i = 0; i < neurals.Length; i++)
            {
                neurals[i].Evaluate_Learning_Rate(sample_count);      
            }
        }
        public void Zero_Hkks()
        {
            for (int i = 0; i < neurals.Length; i++)
            {
                neurals[i].Zeroing_HKKs();
            }

        }
       

    }
}
