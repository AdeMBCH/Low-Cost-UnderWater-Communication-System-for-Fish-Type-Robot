using System;
using System.IO.Ports;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;
using ExtendedSerialPort_NS;
using static System.Net.Mime.MediaTypeNames;

namespace WpfApp1
{
    public partial class MainWindow : Window
    {
        ExtendedSerialPort serialPort1;
        DispatcherTimer timerAffichage;
        Robot robot = new Robot();

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
            int msgFunction = (int)CommandId.Text;
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
        }

        private void buttonTest_Click(object sender, RoutedEventArgs e)
        {
            string s = "Bonjour";
            byte[] msgPayload = Encoding.ASCII.GetBytes(s);
            int msgPayloadLength = msgPayload.Length;
            int msgFunction = (int)CommandId.Text;
            UartEncodeAndSendMessage(msgFunction, msgPayloadLength, msgPayload);
            //SendTextMessage("Bonjour");
        }

        private byte CalculateChecksum(int msgFunction, int msgPayloadLength, byte[] msgPayload)
        {
            byte checksum = 0;
            checksum ^= 0xFE;
            checksum ^= (byte)(msgFunction >> 8);
            checksum ^= (byte)(msgFunction >> 0); // On le fait par symétrie de construction
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
        }

        void ProcessDecodedMessage(int msgFunction, int msgPayloadLength, byte[] msgPayload)
        {
            switch (msgFunction)
            {
                case (int)CommandId.Text:
                    string receivedText = Encoding.ASCII.GetString(msgPayload);
                    textBoxReception.Text = receivedText;
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
                        textBoxReception.Text += "\n";
                        textBoxReception.Text += "Message valide \n";
                        textBoxReception.Text += "Fonction : " + msgDecodedFunction + "\n";
                        textBoxReception.Text += "Taille Payload: " + msgDecodedPayloadLength + "\n";
                        textBoxReception.Text += "Payload : " + Encoding.ASCII.GetString(msgDecodedPayload) + "\n";

                        ProcessDecodedMessage(msgDecodedFunction, msgDecodedPayloadLength, msgDecodedPayload); 
                    }
                    else
                    {
                        textBoxReception.Text += "Erreur Checksum \n";
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