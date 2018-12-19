using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet5
{
    public class Scalar_Layer : layer_
    {
        scalar_neu[] Neurals;
        public double[] error_back_to_previous;//::dE/d(L_input)=dE/d(L+1_input)*d(L+1_input)/d(L_output)*d(L_output)/d(L_input)
        public double[] Second_Deri_error_back_to_previos;
        public double[] Output_Result;
        private int amount_input;
        public Scalar_Layer(int AmountNeurals,int amount_input)
        {
            this.amount_input = amount_input;
            Neurals = new scalar_neu[AmountNeurals];
            for (int i = 0; i < Neurals.Length; i++)
            {
                Neurals[i] = new scalar_neu(AmountNeurals, amount_input);
            }
        }
        public void Proc(double[][] Input_results)
        {
            if (Input_results[0].Length == 1)
            {
                double[] results = new double[Input_results.Length];
                results = results.Zip(Input_results, (a, b) => b[0]).ToArray();
                Output_Result = new double[Neurals.Length];
                for (int i = 0; i < Neurals.Length; i++)
                {
                    Output_Result[i] = Neurals[i].Evaluation(results);
                }
                // return Output_Result;
            }
            else
            {
                throw new ArgumentOutOfRangeException();
            }
        }
       
        public void BP_Proc(double[] Error_back)
        {
           // error_back_to_previous = new double[Neurals.Length][];
            var error_neu = new double[Neurals.Length][];
            error_back_to_previous = new double[amount_input];
            for (int i = 0; i < Neurals.Length; i++)
            {
                error_neu[i] = Neurals[i].Back_Propagation(Error_back[i]);
            }
            for (int j = 0; j < amount_input; j++)
            {
                for (int k = 0; k < error_neu.Length; k++)
                {
                    error_back_to_previous[j] += error_neu[k][j];
                }
            }
        }

        public void PreTraining()
        {
            var error_neu = new double[Neurals.Length][];
            Second_Deri_error_back_to_previos = new double[amount_input];
            for (int i = 0; i < Neurals.Length; i++)
            {
                error_neu[i] = Neurals[i].PreTrain();
            }
            for (int j = 0; j < amount_input; j++)
            {
                for (int k = 0; k < error_neu.Length; k++)
                {
                    Second_Deri_error_back_to_previos[j] += error_neu[k][j];
                }
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
