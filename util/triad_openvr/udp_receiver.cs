//The following code can be used to receive pose data from udp_emitter.py and use it to track an object in unity

using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;

public class udp_tracked_object : MonoBehaviour {
    Thread receiveThread;
    UdpClient client;
    private Double[] float_array;
    private int port = 8051;

    // Use this for initialization
    void Start () {
        float_array = new Double[7];
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }
	
	// Update is called once per frame
	void Update () {
        transform.position = new Vector3((float) float_array[0], (float)float_array[1], (float)float_array[2]);
        transform.rotation = new Quaternion((float)float_array[3], (float)float_array[4], (float)float_array[5], (float)float_array[6]);
	}

    void OnApplicationQuit()
    {
        if (receiveThread != null)
            receiveThread.Abort();
        client.Close();
    }

    // receive thread
    private void ReceiveData()
    {
        port = 8051;
        client = new UdpClient(port);
        print("Starting Server");
        while (true)
        {
            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                byte[] data = client.Receive(ref anyIP);
                for (int i = 0; i < data.Length; i++)
                    float_array[i] = BitConverter.ToDouble(data, i * 8);
            }
            catch (Exception err)
            {
                print(err.ToString());
            }
        }
    }

}
