using System;
using System.IO.Ports;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;
using ExtendedSerialPort_NS;
using ScottPlot;
using Color = ScottPlot.Color;
using static System.Net.Mime.MediaTypeNames;

namespace WpfApp1
{
    public partial class MainWindow : Window
    {
        ExtendedSerialPort serialPort1;
        DispatcherTimer timerAffichage;
        Robot robot = new Robot();
        private List<double> txI = new();
        private List<double> txQ = new();

        public MainWindow()
        {
            InitializeComponent();

            timerAffichage = new DispatcherTimer();
            timerAffichage.Interval = new TimeSpan(0, 0, 0, 0, 100);
            timerAffichage.Tick += TimerAffichage_Tick;
            timerAffichage.Start();

            serialPort1 = new ExtendedSerialPort("COM5", 115200, Parity.None, 8, StopBits.One);
            serialPort1.DataReceived += SerialPort1_DataReceived;
            serialPort1.Open();
        }

        private void TimerAffichage_Tick(object? sender, EventArgs e)
        {
            var byteQueue = robot.byteListReceived;
            while (byteQueue.Count > 0)
            {
                byte b = byteQueue.Dequeue();
                //textBoxReception.Text += $"0x{b.ToString("X2")} ";  // Format 0xhh
                DecodeMessage(b);
            }

            robot.receivedText = string.Empty;
        }

        private void SerialPort1_DataReceived(object? sender, DataReceivedArgs e)
        {
            for (int i = 0; i < e.Data.Length; i++)
            {
                robot.byteListReceived.Enqueue(e.Data[i]);
            }
        }

        private void SendMessage()
        {
            byte[] msgPayload = Encoding.ASCII.GetBytes(textBoxEmission.Text);
            int msgPayloadLength = msgPayload.Length;
            //int msgFunction = (int)CommandId.Text;
            int msgFunction = (int)CommandId.QpskModDemod;
            UartEncodeAndSendMessage(msgFunction, msgPayloadLength, msgPayload);

            textBoxEmission.Clear();
        }

        private void buttonEnvoyer_Click(object sender, RoutedEventArgs e)
        {
            SendMessage();
        }

        private void textBoxEmission_KeyUp(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter)
            {
                SendMessage();
            }
        }

        private void buttonClear_Click(object sender, RoutedEventArgs e)
        {
            textBoxReception.Clear();
            WpfPlotTx.Plot.Clear();
            WpfPlotRx.Plot.Clear();
        }

        private void buttonTest_Click(object sender, RoutedEventArgs e)
        {
            string s = "Bonjour";
            byte[] msgPayload = Encoding.ASCII.GetBytes(s);
            int msgPayloadLength = msgPayload.Length;
            //int msgFunction = (int)CommandId.Text;
            int msgFunction = (int)CommandId.QpskModDemod;
            UartEncodeAndSendMessage(msgFunction, msgPayloadLength, msgPayload);
            //SendTextMessage("Bonjour");
        }


        private void UpdateConstellationPlotTX()
        {
            WpfPlotTx.Plot.Clear();

            var scatter = WpfPlotTx.Plot.Add.Scatter(txI.ToArray(), txQ.ToArray());
            scatter.MarkerSize = 5;
            scatter.Color = ScottPlot.Color.FromHex("#2196F3");
            scatter.LineWidth = 0;

            WpfPlotTx.Plot.Title("Constellation TX");
            WpfPlotTx.Plot.Axes.SetLimits(-1.5, 1.5, -1.5, 1.5);
            WpfPlotTx.Refresh();
        }

        private void UpdateConstellationPlotRX()
        {
            WpfPlotRx.Plot.Clear();

            var scatter = WpfPlotRx.Plot.Add.Scatter(txI.ToArray(), txQ.ToArray());
            scatter.MarkerSize = 5;
            scatter.Color = ScottPlot.Color.FromHex("#FF0000"); 
            scatter.LineWidth = 0;

            WpfPlotRx.Plot.Title("Constellation TX");
            WpfPlotRx.Plot.Axes.SetLimits(-1.5, 1.5, -1.5, 1.5);
            WpfPlotRx.Refresh();
        }





        private byte CalculateChecksum(int msgFunction, int msgPayloadLength, byte[] msgPayload)
        {
            byte checksum = 0;
            checksum ^= 0xFE;
            checksum ^= (byte)(msgFunction >> 8);
            checksum ^= (byte)(msgFunction >> 0);
            checksum ^= (byte)(msgPayloadLength >> 8);
            checksum ^= (byte)(msgPayloadLength >> 0);

            foreach (byte bt in msgPayload)
            {
                checksum ^= bt;
            }

            return checksum;
        }

        void UartEncodeAndSendMessage(int msgFunction, int msgPayloadLength, byte[] msgPayload)
        {
            byte[] message = new byte[6 + msgPayloadLength];
            int pos = 0;
            message[pos++] = 0xFE;
            message[pos++] = (byte)(msgFunction >> 8);
            message[pos++] = (byte)(msgFunction >> 0);
            message[pos++] = (byte)(msgPayloadLength >> 8);
            message[pos++] = (byte)(msgPayloadLength >> 0);
            for (int i = 0; i < msgPayloadLength; i++)
            {
                message[pos++] = msgPayload[i];
            }
            byte checksum = CalculateChecksum(msgFunction, msgPayloadLength, msgPayload);
            message[pos++] = checksum;
            serialPort1.Write(message, 0, pos);
        }

        public enum CommandId
        {
            Text = 0x0080,
            QpskModDemod = 0x1010,
            QpskResult = 0x9010,
            IQ_DATA = 0x55AA

        }

        void ProcessDecodedMessage(int msgFunction, int msgPayloadLength, byte[] msgPayload)
        {
            //System.Diagnostics.Debug.WriteLine($"msgFunction=0x{msgFunction:X4}, payloadLen={msgPayloadLength}");
            switch (msgFunction)
            {
                case (int)CommandId.Text:
                    string receivedText = Encoding.ASCII.GetString(msgPayload);
                    textBoxReception.Text = receivedText;
                    break;
                case (int)CommandId.QpskResult:
                    string qpskText = Encoding.ASCII.GetString(msgPayload);
                    textBoxReception.Text += "[QPSK Demodulated] : " + qpskText + "\n";
                    break;
                case (int)CommandId.IQ_DATA:
                    byte type = msgPayload[0];
                    sbyte i = (sbyte)msgPayload[1];
                    sbyte q = (sbyte)msgPayload[2];
                    if (type == (byte)'T')
                    {
                        double iNorm = i / 127.0;
                        double qNorm = q / 127.0;
                        txI.Add(iNorm);
                        txQ.Add(qNorm);
                        UpdateConstellationPlotTX();
                    }
                    if (type == (byte)'R')
                    {
                        double iNorm = i / 127.0;
                        double qNorm = q / 127.0;
                        txI.Add(iNorm);
                        txQ.Add(qNorm);
                        UpdateConstellationPlotRX();
                    }
                    break;
                default:
                    break;
            }
        }


        public enum StateReception
        {
            Waiting,
            FunctionMSB,
            FunctionLSB,
            PayloadLengthMSB,
            PayloadLengthLSB,
            Payload,
            CheckSum
        }

        StateReception rcvState = StateReception.Waiting;
        int msgDecodedFunction = 0;
        int msgDecodedPayloadLength = 0;
        byte[] msgDecodedPayload;
        int msgDecodedPayloadIndex = 0;

        private void DecodeMessage(byte c)
        {
            switch (rcvState)
            {
                case StateReception.Waiting:
                    if (c == 0xFE)
                    {
                        rcvState = StateReception.FunctionMSB;
                    }
                    break;
                case StateReception.FunctionMSB:
                    msgDecodedFunction = (c << 8);
                    rcvState = StateReception.FunctionLSB;
                    break;
                case StateReception.FunctionLSB:
                    msgDecodedFunction |= (c << 0) ;
                    rcvState = StateReception.PayloadLengthMSB;
                    break;
                case StateReception.PayloadLengthMSB:
                    msgDecodedPayloadLength = (c << 8) ;
                    rcvState = StateReception.PayloadLengthLSB;
                    break;
                case StateReception.PayloadLengthLSB:
                    msgDecodedPayloadLength |= (c << 0) ;
                    if (msgDecodedPayloadLength > 1024)
                    {
                        rcvState = StateReception.Waiting;
                    }
                    else if (msgDecodedPayloadLength > 0)
                    {
                        msgDecodedPayload = new byte[msgDecodedPayloadLength];
                        msgDecodedPayloadIndex = 0;
                        rcvState = StateReception.Payload;
                    }
                    else
                    {
                        rcvState = StateReception.CheckSum;
                    }
                    break;
                case StateReception.Payload:
                    msgDecodedPayload[msgDecodedPayloadIndex++] = c;
                    if (msgDecodedPayloadIndex >= msgDecodedPayloadLength)
                    {
                        rcvState = StateReception.CheckSum;
                    }
                    break;
                case StateReception.CheckSum:
                    byte receivedChecksum = c;
                    byte calculatedChecksum = CalculateChecksum(msgDecodedFunction, msgDecodedPayloadLength, msgDecodedPayload);

                    if (receivedChecksum == calculatedChecksum)
                    {
                        /*textBoxReception.Text += "\n";
                        textBoxReception.Text += "Valid Message \n";
                        textBoxReception.Text += "Function : " + msgDecodedFunction + "\n";
                        textBoxReception.Text += "Payload Length: " + msgDecodedPayloadLength + "\n";
                        textBoxReception.Text += "Payload : " + Encoding.ASCII.GetString(msgDecodedPayload) + "\n";
                        */
                        // Affichage hexadécimal pour debug :
                        /*textBoxReception.Text += "Payload HEX : ";
                        for (int i = 0; i < msgDecodedPayloadLength; i++)
                            textBoxReception.Text += $"{msgDecodedPayload[i]:X2} ";
                        textBoxReception.Text += "\n";
                        */

                        ProcessDecodedMessage(msgDecodedFunction, msgDecodedPayloadLength, msgDecodedPayload); 
                    }
                    else
                    {
                        textBoxReception.Text += "Checksum Error \n";
                    }
                    rcvState = StateReception.Waiting;
                    break;
                default:
                    rcvState = StateReception.Waiting;
                    break;
            }
        }
    }
}