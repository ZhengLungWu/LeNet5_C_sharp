using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Drawing;
using System.Drawing.Imaging;

namespace LeNet5
{
    class Image_Processing
    {
        /*
        * some of the pixel operating algorithm was refered from:
        * https://github.com/patrickmeiring/LeNet
        * 
        */
        public static double[] ToDoubleArray(byte[] src)
        {
            double[] a = new double[src.Length];
            for (int i = 0; i < src.Length; i++)
            {
                a[i] = src[i];
            }

            return a;
        }

        public static Bitmap ToBitmap( double[] doubles, int width)
        {
            if (width <= 0) throw new ArgumentException();
            int length = doubles.Length;
            int height = (length + width - 1) / width;

            Bitmap result = new Bitmap(width, height, PixelFormat.Format32bppPArgb);
            for (int i = 0; i < length; i++)
            {
                result.SetPixel(i % width, i / width, ToPixel(doubles[i]));
            }
            return result;
        }
        public static Bitmap ToBitmap(double[] doubles,int height,int width)
        {
            if (width <= 0) throw new ArgumentException();
            int length = doubles.Length;
            

            Bitmap result = new Bitmap(width, height, PixelFormat.Format32bppPArgb);
            for (int i = 0; i < length; i++)
            {
                result.SetPixel(i % width, i / width, ToPixel(doubles[i]));
            }
            return result;
        }


        private static Color ToPixel(double value)
        {
            double boundedValue = Math.Min(Math.Max(value + 2, 0), 4);
            byte pixelState = (byte)(boundedValue * 255.0 / 4.0);
            return Color.FromArgb(255, pixelState, pixelState, pixelState);
        }


        public static double[] ToDoubleArray( Bitmap bitmap)
        {
            int width = bitmap.Width;
            int height = bitmap.Height;
            int length = width * height;
            double[] result = new double[width * height];

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = 1.0 - (bitmap.GetPixel(i % width, i / width).GetBrightness() * 2.0);
            }
            return result;
        }
        public static byte[] ToByteArray(double[] src )
        {
            byte[] x = new byte[src.Length];
            for (int i = 0; i < x.Length; i++)
            {
                x[i] =(byte) src[i];
            }
            return x;

        }
        
        public static double[] Padding_2_EachSide(double[]Image)
        {           
            int root =(int) Math.Sqrt(Image.Length) ;
            int root_new = root + 2 * 2;
            var Grays = new double[root_new];
            //var avg = Calculate_AVG(Image);
            for (int i = 0; i < Grays.Length; i++)
            {
                Grays[i] = -0.1;
            }
            
            var res = new List<double>();

            res.AddRange(Grays);
            res.AddRange(Grays);

            for (int i = 0; i < root; i++)
            {
                res.Add(-0.1);
                res.Add(-0.1);
                res.AddRange(Image.Skip(i * root).Take(root));
                res.Add(-0.1);
                res.Add(-0.1);
            }
            res.AddRange(Grays);
            res.AddRange(Grays);
            return res.ToArray();
        }



        public static byte[] Padding_2_EachSide(byte[] src)
        {
            int root = (int)Math.Sqrt(src.Length);
            int root_new = root + 2 * 2;
            var Grays = new byte[root_new];
            var avg = Calculate_AVG(src);
            for (int i = 0; i < Grays.Length; i++)
            {
                Grays[i] = avg;
            }

            var res = new List<byte>();
            res.AddRange(Grays);
            res.AddRange(Grays);

            for (int i = 0; i < root; i++)
            {
                res.Add(avg);
                res.Add(avg);
                res.AddRange(src.Skip(i * root).Take(root));
                res.Add(avg);
                res.Add(avg);
            }
            res.AddRange(Grays);
            res.AddRange(Grays);
            return res.ToArray();

        }

        private static double Calculate_AVG(double[] IMG)
        {
            double v=0;
            for (int i = 0; i < IMG.Length; i++)
            {
                v += IMG[i];
            }
            v /= IMG.Length;
            return v;
            
        }

        private static byte Calculate_AVG(byte[] IMG)
        {
            double v = 0;
            for (int i = 0; i < IMG.Length; i++)
            {
                v += IMG[i];
            }
            v /= IMG.Length;
            return(byte) v;

        }

        public static byte[] Histogram(double[] input)
        {
            // int root =(int) Math.Sqrt(input.Length);
            var ITEMS = new List<double>();

            for (int i = 0; i < input.Length; i++)
            {
                if (!ITEMS.Contains(input[i]))
                {
                    ITEMS.Add(input[i]);
                }
            }
            ITEMS.Sort((a, b) => a.CompareTo(b));
            var amount = new double[ITEMS.Count];

            for (int i = 0; i < input.Length; i++)
            {
                for (int j = 0; j < ITEMS.Count; j++)
                {
                    if (ITEMS[j] == input[i])
                    {
                        for (; j < ITEMS.Count; j++)
                        {
                            amount[j]++;
                        }
                        break;
                    }
                }
            }

            // float Min= amount.Min();
            double output = 0;
            byte[] res = new byte[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                for (int j = 0; j < ITEMS.Count; j++)
                {
                    if (ITEMS[j] == input[i])
                    {
                        output = ((amount[j] - amount[0]) / (input.Length - amount[0]) * 255);
                        output = Math.Round(output);
                        res[i] = (byte)output;
                    }
                }
            }
            return res;


        }
        public static Bitmap Bytes2bitmap(byte[] img)
        {
            int root = (int)Math.Sqrt(img.Length);
            Bitmap im = new Bitmap(root, root, PixelFormat.Format24bppRgb);
            var BitData = im.LockBits(new Rectangle(0, 0, root, root), ImageLockMode.WriteOnly, im.PixelFormat);
            var Scan0 = BitData.Scan0;
            var Stride = BitData.Stride;
            // Console.WriteLine($"stride:{Stride}");
            var width = BitData.Width;
            // Console.WriteLine($"Width:{width}");
            int delta = Stride - width * 3;

            var list_img = new List<byte>();
            for (int i = 0; i < img.Length; i++)
            {
                int s = (255 - img[i]);

                for (int j = 0; j < 3; j++)
                {
                    list_img.Add((byte)s);
                }
                if (i % width == 0 && i > 0)
                {
                    for (int j = 0; j < delta; j++)
                    {
                        list_img.Add(255);
                    }
                }
            }

            Marshal.Copy(list_img.ToArray(), 0, Scan0, list_img.Count);
            im.UnlockBits(BitData);

            return im;
        }
    }
}
