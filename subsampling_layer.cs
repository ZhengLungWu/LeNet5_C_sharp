using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet5
{

    public class SubPooling_layer : layer_
    {
        public sub_neu[] Neurals;
        public double[][] Output_Proc_Result;
        public double[][] error_back_to_previous,Second_Deri_error_to_previous;

        public SubPooling_layer(int Input_img_Width, int AmountNeural, int Output_img_Width)
        {
            Neurals = new sub_neu[AmountNeural];
            for (int i = 0; i < Neurals.Length; i++)
            {
                Neurals[i] = new sub_neu(Input_img_Width, AmountNeural, Output_img_Width);
            }
        }

        public void Proc(double[][] result_Prev_layer)
        {
            Output_Proc_Result = new double[Neurals.Length][];
            for (int i = 0; i < Neurals.Length; i++)
            {
                Output_Proc_Result[i] = Neurals[i].Evaluation(result_Prev_layer[i]);
            }
        }
        public void BP_Proc(double[][] Error_back, Mapping_type mapping_Type)
        {
            if (mapping_Type == Mapping_type.All2All)//C5->S4
            {
                BP_internal_All2All(Error_back);//one2All       
            }
            else//specified  C3->S2
            {
                BP_internal_specified(Error_back);
            }
        }

        private void BP_internal_All2All(double[][] Error_back)
        {
            error_back_to_previous = new double[Neurals.Length][];

            for (int i = 0; i < Neurals.Length; i++)
            {
                error_back_to_previous[i] = Neurals[i].Back_Propagation(Error_back);
            }
        }

        private void BP_internal_specified(double[][] Error_back)
        {
            error_back_to_previous = new double[Neurals.Length][];
            for (int i = 0; i < exec_array23.GetUpperBound(1) + 1; i++)//6
            {
                var Error_temp_list = new List<double[]>();
                for (int j = 0; j < exec_array23.GetUpperBound(0) + 1; j++)//16
                {
                    if (exec_array23[j, i])
                    {
                        Error_temp_list.Add(Error_back[j]);
                    }
                }
                error_back_to_previous[i] = Neurals[i].Back_Propagation(Error_temp_list.ToArray());
            }
        }
        public void PreTraining(double[][] Second_deri_Error_back, Mapping_type mapping_Type)
        {
            if (mapping_Type == Mapping_type.All2All)//C5->S4
            {
                PT_internal_All2All(Second_deri_Error_back);//one2All       
            }
            else//specified  C3->S2
            {
                PT_internal_specified(Second_deri_Error_back);
            }



        }
        private void PT_internal_All2All(double[][] Error_back)
        {
            Second_Deri_error_to_previous = new double[Neurals.Length][];

            for (int i = 0; i < Neurals.Length; i++)
            {
                Second_Deri_error_to_previous[i] = Neurals[i].PreTrain(Error_back);
            }
        }
        private void PT_internal_specified(double[][] Error_back)
        {
            Second_Deri_error_to_previous = new double[Neurals.Length][];
            for (int i = 0; i < exec_array23.GetUpperBound(1) + 1; i++)//6
            {
                var Error_temp_list = new List<double[]>();
                for (int j = 0; j < exec_array23.GetUpperBound(0) + 1; j++)//16
                {
                    if (exec_array23[j, i])
                    {
                        Error_temp_list.Add(Error_back[j]);
                    }
                }
                Second_Deri_error_to_previous[i] = Neurals[i].PreTrain(Error_temp_list.ToArray());
            }
        }


        public void Evaluate_Learning_rate(double sample_count)
        {
            for (int i = 0; i < Neurals.Length; i++)
            {
                Neurals[i].Evaluate_Learning_Rate(sample_count);
            }
        }
        public void Zero_Hkks()
        {
            for (int i = 0; i < Neurals.Length; i++)
            {
                Neurals[i].Zeroing_HKKs();
            }

        }


    }

}
