using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace SoundRecognition
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            startUdpServer();
        }
        byte[] data = new byte[100];
        
        private void button1_Click(object sender, EventArgs e)
        {
            foreach (var process in Process.GetProcessesByName("python.exe"))
            {
                process.Kill();
            }
            run_cmd("run.py","");
            
        }
        private void run_cmd(string cmd, string args)
        {
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = "python.exe";
            start.Arguments = string.Format("{0} {1}", cmd, args);
            //start.UseShellExecute = false;
            //start.RedirectStandardOutput = true;
            Process process = Process.Start(start);
            /*using (Process process = Process.Start(start))
            {
                using (StreamReader reader = process.StandardOutput)
                {
                    string result = reader.ReadToEnd();
                    Console.Write(result);
                }
            }*/
        }
        private void startUdpServer()
        {
            Thread listener = new Thread(() =>
            {
                IPEndPoint ServerEndPoint = new IPEndPoint(IPAddress.Any, 5005);
                Socket WinSocket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
                WinSocket.Bind(ServerEndPoint);
                IPEndPoint packetsender = new IPEndPoint(IPAddress.Any, 0);
                EndPoint Remote = (EndPoint)(packetsender);
                while (true)
                {
                    try
                    {

                        this.richTextBox1.Invoke((MethodInvoker)delegate {
                            // Running on the UI thread
                            richTextBox1.Text += " ";
                        });
                        int recv = WinSocket.ReceiveFrom(data, ref Remote);

                        this.richTextBox1.Invoke((MethodInvoker)delegate {
                            // Running on the UI thread
                            richTextBox1.Text += data[0].ToString();
                        });
                    }

                    catch (SocketException)
                    {
                        continue; // exit the while loop
                    }
                }
            });
            listener.IsBackground = true;
            listener.Start();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            run_cmd("train.py", "");
        }

        private void button3_Click(object sender, EventArgs e)
        {
            run_cmd("loadData.py", "");
        }
    }
}
