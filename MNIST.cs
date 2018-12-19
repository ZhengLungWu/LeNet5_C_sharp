using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace LeNet5
{
    class MNIST
    {
        /*
         * some of the MNIST processing algorithm was refered from:
         * https://github.com/patrickmeiring/LeNet
         * 
         */

        static byte[] tr_imgs = Properties.Resources.train_images;
       static byte[] tr_labels = Properties.Resources.train_labels;
       

       
        public static List<byte[]> Get_All_Training_Images
        {
            get
            {
               // Console.WriteLine(tr_imgs.Length / 28 / 28);
                var result = new List<byte[]>();
                byte[] myby = new byte[28 * 28];
                for (int i = 0; i <tr_imgs.Length/(28*28); i++)
                {
                    myby = new byte[28 * 28];                  
                    Array.Copy(tr_imgs, 16 + 28 * 28 * i, myby, 0, 28*28);
                    myby=Image_Processing.Padding_2_EachSide(myby);
                    result.Add(myby);
                }
                return result;
            }


        }
        public static List<double[]> Get_All_Training_Images_double
        {
            get
            {
               // Console.WriteLine(tr_imgs.Length / 28 / 28);
                var result = new List<double[]>();
                for (int i = 0; i < tr_imgs.Length / (28 * 28); i++)
                {
                   var mydb = new double[28 * 28];

                    for (int j = 0; j < mydb.Length; j++)
                    {
                        mydb[j] = ((double)tr_imgs[16 + i * 28 * 28 + j] / 255.0) * 1.275 - 0.1;
                    }
                   mydb=Image_Processing.Padding_2_EachSide(mydb);

                    result.Add(mydb);   
                }
                return result;
            }
        }

        public static double[][] Get_PreTrain_Images_double
        {
            get
            {
                var result = new List<double[]>();
                
                for (int i = 0; i < 500; i++)
                {
                    var mydb = new double[28 * 28];

                    for (int j = 0; j < mydb.Length; j++)
                    {
                        mydb[j] = ((double)tr_imgs[16 + i * 28 * 28 + j] / 255.0) * 1.275 - 0.1;
                    }
                    mydb = Image_Processing.Padding_2_EachSide(mydb);

                    result.Add(mydb);
                }
                return result.ToArray();






            }

        }



        public static double[] Get_Single_Training_Image_double(int index)
        {
            
                var mydb = new double[28 * 28];

                for (int j = 0; j < mydb.Length; j++)
                {
                    mydb[j] = ((double)tr_imgs[16 + index * 28 * 28 + j] / 255.0) * 1.275 - 0.1;
                }
                mydb = Image_Processing.Padding_2_EachSide(mydb);

            return mydb;
        }

        public static List<int> Get_All_Training_Labels
        {
            get
            {
                var res = new List<int>();

                for (int i = 8; i < tr_labels.Length; i++)
                {
                    res.Add((int)tr_labels[i]);
                }
                return res;
            }
        }
       
        public static byte[] TakeFirstImg
        {
            get
            {
                var img_bytes = tr_imgs.Skip(16).Take(28 * 28);
                return img_bytes.ToArray();
            }
        }
      




    }
}
