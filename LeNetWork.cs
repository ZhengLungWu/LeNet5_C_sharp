using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeNet5
{
    public delegate void Correct_Handler();

    class LeNetWork
    {
        
        Conv_layer C1, C3, C5;
        SubPooling_layer S2, S4;
        Scalar_Layer F6;
        Output_Layer OL;


       
        public event Image_Hander input_0_event, F6_event,output_7_event;
        public event Layer_Done_Handler C1_event, S2_event, C3_event, S4_event, C5_evnet;
        public event Correct_Handler Correct_event;
       // private int Correct_Counter = 0;

        public LeNetWork()
        {
            neural_.TAU = 0.0005;
            neural_.MU = 0.5;
            InitializeComps();
        }


        public void Processing_Single(double[]input_img,int label)
        {
            FeedData(input_img);
            BackProp(label);
        }
        public void Processing_PreTrain(double[][]input_imgs)
        {

            C1.Zero_Hkks();
            S2.Zero_Hkks();
            C3.Zero_Hkks();
            S4.Zero_Hkks();
            C5.Zero_Hkks();
            F6.Zero_Hkks();
            double Sample_count = 500.0;
            if (input_imgs.Length >= Sample_count)
            {
                for (int i = 0; i < Sample_count; i++)
                {
                    FeedData(input_imgs[i]);
                    PreTrain();
                }

                C1.Evaluate_Learning_rate(Sample_count);
                S2.Evaluate_Learning_rate(Sample_count);
                C3.Evaluate_Learning_rate(Sample_count);
                S4.Evaluate_Learning_rate(Sample_count);
                C5.Evaluate_Learning_rate(Sample_count);
                F6.Evaluate_Learning_rate(Sample_count);

            }



        }



        private void InitializeComps()
        {
            
            C1 = new Conv_layer(32,1, 6, 28);
            S2 = new SubPooling_layer(28, 6, 14);
            C3 = new Conv_layer(14,6, 16, 10);
            S4 = new SubPooling_layer(10, 16, 5);
            C5 = new Conv_layer(5,16, 120, 1);
            F6 = new Scalar_Layer(84,120);
            OL = new Output_Layer();
        }
       
        public void FeedData(double[] Input_img)
        {
            input_0_event?.Invoke(Input_img);
            //Console.WriteLine("C1 proceed");
            C1.Proc(Input_img);
            C1_event?.Invoke(C1.Output_Proc_Result);

           // Console.WriteLine("S2 proceed");
            S2.Proc(C1.Output_Proc_Result);
            S2_event?.Invoke(S2.Output_Proc_Result);

           // Console.WriteLine("C3 proceed");
            C3.ProcIntermed(S2.Output_Proc_Result, false);
            C3_event?.Invoke(C3.Output_Proc_Result);

           // Console.WriteLine("S4 proceed");
            S4.Proc(C3.Output_Proc_Result);
            S4_event?.Invoke(S4.Output_Proc_Result);

           // Console.WriteLine("C5 proceed");
            C5.ProcIntermed(S4.Output_Proc_Result, true);
            C5_evnet?.Invoke(C5.Output_Proc_Result);

           // Console.WriteLine("F6 proceed");
            F6.Proc(C5.Output_Proc_Result);
            F6_event?.Invoke(F6.Output_Result);

          //  Console.WriteLine("Output proceed");
            OL.Proc(F6.Output_Result);
            

        }

        public void BackProp(int Correct_Number)
        {
            Console.WriteLine($"Correct Number{Correct_Number},Evaluated:{OL.Result_int}");
            if (Correct_Number == OL.Result_int)
            { Correct_event?.Invoke(); }
          //  Console.WriteLine("Output BP proceed");
            OL.Back_Proc(Correct_Number);
            output_7_event?.Invoke(OL.Result_img);

          //  Console.WriteLine("F6 BP proceed");
            //-----Error from OL to F6 is 1-1
            F6.BP_Proc(OL.Error_to_F6);
          //  Console.WriteLine("C5 BP proceed");
            //-------- Error from F6 to C5 is 1-all, one F6 neural connection to every C5 neural.
            // but the dF6_input/dC5_output=1, so the error every F6 neural comes back to C5 only need 1 error.            
            C5.BP_Proc(F6.error_back_to_previous);
         //   Console.WriteLine("S4 BP proceed");
            //C5 has 120*25 errors needs to be come back to S4: dE/dC5_input,
            //dC5_i_input[x,y]/dS4_j_output[a,b]=1 if x=a,y=b, otherwise,0.  
            //C5->S4 all c5 
            S4.BP_Proc(C5.error_back_to_previous,Mapping_type.All2All);
           // Console.WriteLine("C3 BP proceed");
            //S4-C3 is 1-1 mapping.
            C3.BP_Proc(S4.error_back_to_previous);
          //  Console.WriteLine("S2 BP proceed");
            //C3-S2 has specified connection
            S2.BP_Proc(C3.error_back_to_previous,Mapping_type.Specified);
          //  Console.WriteLine("C1 BP proceed");
            //S2-C1 is 1-1 mapping
            C1.BP_Proc(S2.error_back_to_previous);

        }
        private void PreTrain()
        {
            //OL.Back_Proc(Correct_Number);
           // output_7_event?.Invoke(OL.Result_img);

           
            //-----Second_deriative_Error from OL to F6 is 1-1
            F6.PreTraining();
                  
            C5.PreTraining(F6.Second_Deri_error_back_to_previos);
           
            S4.PreTraining(C5.Second_Deri_error_back_to_previous, Mapping_type.All2All);
            
            C3.PreTraining(S4.Second_Deri_error_to_previous);
           
            S2.PreTraining(C3.Second_Deri_error_back_to_previous, Mapping_type.Specified);
           
            C1.PreTraining(S2.Second_Deri_error_to_previous);




        }
        





    }
}
