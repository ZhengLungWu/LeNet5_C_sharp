using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Diagnostics;
using System.IO;

namespace LeNet5
{

    public delegate void ListboxUpdateHandler(object item);

    public partial class Form1 : Form
    {
        LeNetWork myNetWork;
        private PictureBox input_0, PIC_F6, output_7;
        private PictureBox[] PIC_C1, PIC_S2, PIC_C3, PIC_S4, PIC_C5;
        private TextBox Correct_txb;
        private ListBox State_Lbx;
        private float Correct_Counter = 0;

        private static event ListboxUpdateHandler StateLbxUpdateEvent;


        public Form1()
        {
            InitializeComponent();
            myNetWork = new LeNetWork();
            EventSetting();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            IniPictureBoxes();
            InitListBox();
        }

        private void EventSetting()
        {
            myNetWork.Correct_event += new Correct_Handler(CorrectNess);

            myNetWork.input_0_event += new Image_Hander(UpdateInput0);
            myNetWork.C1_event += new Layer_Done_Handler(UpdateC1);
            myNetWork.S2_event += new Layer_Done_Handler(UpdateS2);
            myNetWork.C3_event += new Layer_Done_Handler(UpdateC3);
            myNetWork.S4_event += new Layer_Done_Handler(UpdateS4);
            myNetWork.C5_evnet += new Layer_Done_Handler(UpdateC5);
            myNetWork.F6_event += new Image_Hander(UpdateF6);
            myNetWork.output_7_event += new Image_Hander(UpdateOutput7);
            StateLbxUpdateEvent += new ListboxUpdateHandler(StateLbx_Update);

        }
        private void UpdateInput0(double[] image)
        {
            input_0.Image = Image_Processing.ToBitmap(image, 32);
        }
        private void UpdateC1(double[][] IMAGES)
        {
            for (int i = 0; i < PIC_C1.Length; i++)
            {
                PIC_C1[i].Image = Image_Processing.ToBitmap(IMAGES[i], 28);
            }


        }
        private void UpdateS2(double[][] IMAGES)
        {
            for (int i = 0; i < PIC_S2.Length; i++)
            {
                PIC_S2[i].Image = Image_Processing.ToBitmap(IMAGES[i], 14);
            }
        }
        private void UpdateC3(double[][] IMAGES)
        {
            for (int i = 0; i < PIC_C3.Length; i++)
            {
                PIC_C3[i].Image = Image_Processing.ToBitmap(IMAGES[i], 10);
            }
        }
        private void UpdateS4(double[][] IMAGES)
        {
            for (int i = 0; i < PIC_S4.Length; i++)
            {
                PIC_S4[i].Image = Image_Processing.ToBitmap(IMAGES[i], 5);
            }
        }

       

        private void UpdateC5(double[][] IMAGES)
        {
            for (int i = 0; i < PIC_C5.Length; i++)
            {
                PIC_C5[i].Image = Image_Processing.ToBitmap(IMAGES[i], 1);
            }
        }
        private void UpdateF6(double[] image)
        {
            PIC_F6.Image = Image_Processing.ToBitmap(image, 12, 7);
        }
        private void UpdateOutput7(double[] image)
        {
            output_7.Image = Image_Processing.ToBitmap(image, 12,7);
        }
        private void UpdatePercentage(double percent)
        {
            Correct_txb.Text = $"{percent}%";

        }


        private void button1_Click(object sender, EventArgs e)
        {

            if (!backgroundWorker1.IsBusy)
            {
                backgroundWorker1.RunWorkerAsync();
            }

        }

        private void CorrectNess()
        {
            Correct_Counter++;
            

        }

        private void IniPictureBoxes()
        {
            input_0 = new PictureBox
            {
                Location = new Point(20, 20),
                Size = new Size(100, 100),
                Visible = true,
                Parent=this,
                SizeMode=PictureBoxSizeMode.StretchImage
            };
            PIC_C1 = new PictureBox[6];
            PIC_S2 = new PictureBox[6];
            for (int i = 0; i < 6; i++)
            {
                PIC_C1[i] = new PictureBox
                {
                    Location = new Point(150, 20 + i * 60),
                    Size = new Size(50, 50),
                    Visible = true,
                    Parent = this,
                    SizeMode = PictureBoxSizeMode.StretchImage,
                };
                PIC_S2[i]=new PictureBox
                {
                    Location = new Point(220, 20 + i * 60),
                    Size = new Size(50, 50),
                    Visible = true,
                    Parent = this,
                    SizeMode = PictureBoxSizeMode.StretchImage,
                };
            }
            PIC_C3 = new PictureBox[16];
            PIC_S4 = new PictureBox[16];
            for (int j = 0; j < 16; j++)
            {
                PIC_C3[j] = new PictureBox
                {
                    Location = new Point(280, 20 + j * 60),
                    Size = new Size(50, 50),
                    Visible = true,
                    Parent = this,
                    SizeMode = PictureBoxSizeMode.StretchImage,
                };
                PIC_S4[j]=new PictureBox
                {
                    Location = new Point(340, 20 + j * 60),
                    Size = new Size(50, 50),
                    Visible = true,
                    Parent = this,
                    SizeMode = PictureBoxSizeMode.StretchImage,
                };
            }
            PIC_C5 = new PictureBox[120];
            for (int j = 0; j < PIC_C5.Length; j++)
            {
                PIC_C5[j] = new PictureBox
                {
                    Location = new Point(400, 20 + j * 20),
                    Size = new Size(20,20),
                    Visible = true,
                    Parent = this,
                    SizeMode = PictureBoxSizeMode.StretchImage,

                };
            }
            PIC_F6 = new PictureBox
            {
                Location = new Point(450, 20),
                Size = new Size(100,100),
                Visible = true,
                Parent = this,
                SizeMode = PictureBoxSizeMode.StretchImage,
            };
            output_7 = new PictureBox
            {
                Location = new Point(600, 20),
                Size = new Size(7*10,12*10),
                Visible = true,
                Parent = this,
                SizeMode = PictureBoxSizeMode.StretchImage,
            };
            Correct_txb = new TextBox
            {
                Location = new Point(800, 20),
                Size = new Size(200, 200),
                Visible = true,
                Parent = this,
                Font = new Font(FontFamily.GenericMonospace, 18.0f)


            };

        }

        private void InitListBox()
        {
            State_Lbx = new ListBox
            {
                Location=new Point(800,100),
                Size=new Size(200,400),
                Visible=true,
                Parent=this,



            };

        }


        private void backgroundWorker1_DoWork(object sender, DoWorkEventArgs e)
        {

            string[][] saver= new string[25][];
            
           
            for (int k = 0; k < 25; k++)
            {
                Correct_Counter = 0;
                Debug.Print($"RUN EPOCH{k + 1}");
               StateLbxUpdateEvent?.Invoke($"PreTrain:EPOCH{k + 1}");
                myNetWork.Processing_PreTrain(MNIST.Get_PreTrain_Images_double);
                StateLbxUpdateEvent?.Invoke($"Start training:EPOCH{k+1}");
                File.AppendAllText("D:\\correctness.csv", $"Epoch{k+1}"+Environment.NewLine);
                float cor=0.0F;
                for (int i = 0; i < 60000; i++)
                {
                    myNetWork.Processing_Single(MNIST.Get_Single_Training_Image_double(i), MNIST.Get_All_Training_Labels[i]);
                     cor = Correct_Counter / (float)(i + 1);
                    Debug.Print($"correctness:{cor * 100 }....{i + 1}");
                    if ((i + 1) % 1000 == 0)
                    {
                        File.AppendAllText("D:\\correctness.csv", $"{cor * 100},{i+1}"+Environment.NewLine);

                    }


                    if (Correct_txb.InvokeRequired)
                    {
                        Correct_txb.Invoke((MethodInvoker)delegate
                        {
                            UpdatePercentage(cor * 100);
                        });
                    }
                }
                StateLbxUpdateEvent?.Invoke($"correctness Epoch{k + 1}:{cor * 100 }");
            }
        }
        private void button2_Click(object sender, EventArgs e)
        {
            
        }

        private void StateLbx_Update(object sender)
        {
            if (State_Lbx.InvokeRequired)
            {
                State_Lbx.Invoke((MethodInvoker)delegate
                  {
                      State_Lbx.Items.Add(sender);
                   }
                  );
            }
        }

    }
}
