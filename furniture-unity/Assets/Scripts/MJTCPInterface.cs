/*
Original work Copyright 2019 Roboti LLC
Modified work Copyright 2019 Panasonic Beta, a division of Panasonic Corporation of North America

Redistribution and use of this file (hereafter "Software") in source and 
binary forms, with or without modification, are permitted provided that 
the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, 
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation 
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
This class is based on code originally located in MJRemote.cs
*/

using UnityEngine;

using System;
using System.Net.Sockets;
using System.Collections;
using System.Collections.Generic;

public class MJTCPInterface : MonoBehaviour
{
    // socket commands from client
    enum Command : int
    {
        None = 0,    // null command
        GetInput = 1,    // send: key, select, active, refpos[3], refquat[4] (40 bytes)
        GetImage = 2,    // send: rgb image (3*width*height bytes)
        SaveSnapshot = 3,    // (no data exchange)
        SaveVideoframe = 4,    // (no data exchange)
        SetCamera = 5,    // receive: camera index (4 bytes)
        SetQpos = 6,    // receive: qpos (4*nqpos bytes)
        SetMocap = 7,     // receive: mocap_pos, mocap_quat (28*nmocap bytes)
        GetSegmentationImage = 8,    // send: rgb image (3*width*height bytes)
        ChangeWorld = 9,   // send: rgb image (3*width*height bytes)
        GetWorldInfo = 10,
        RandomizeAppearance = 11,
        GetDepthImage = 12,

        // For Furniture Assembly Environment
        SetResolution = 13,    // receive: width, height (4 * 2 bytes)
        SetGeomPos = 14,    // receive: geom_name, pos (string, 3 * 4 bytes)
        GetKeyInput = 15,
        SetBackground = 16,
        SetGraphicsQuality = 17,
    }

    public string tcpAddress = "0.0.0.0";
    public int tcpPort;

    // GUI
    static GUIStyle style = null;

    // remote data
    TcpListener listener = null;
    TcpClient client = null;
    NetworkStream stream = null;
    byte[] buffer;
    int buffersize = 0;

    float lastcheck = 0;

    public GameObject root = null;

    // For Furniture Assembly Environment
    Queue<string> keyInputs = new Queue<string>();
    readonly KeyCode[] keyCodes =
    {
        KeyCode.Space, KeyCode.Return,
        KeyCode.Alpha1, KeyCode.Alpha2,
        KeyCode.Q, KeyCode.W, KeyCode.E,
        KeyCode.A, KeyCode.S, KeyCode.D,
        KeyCode.U, KeyCode.I, KeyCode.O,
        KeyCode.J, KeyCode.K, KeyCode.L,
        KeyCode.C, KeyCode.Y,
        KeyCode.R, KeyCode.T, KeyCode.Escape,
    };

    // Use this for initialization
    void Start()
    {
        tcpPort = 1050;

        // For Furniture Assembly Environment: update tcpPort if specified in the command line
        var args = System.Environment.GetCommandLineArgs();
        for (var i = 0; i < args.Length; i++) {
            print(args[i]);
            if (args[i] == "--port") {
                tcpPort = int.Parse(args[i + 1]);
                print(string.Format("Update port with {0} from argument", tcpPort.ToString()));
            }
        }
        print(string.Format("Open port {0}", tcpPort.ToString()));

        // For Furniture Assembly Environment: no full screen
        Screen.SetResolution(500, 500, false);
        // print all available graphics quality
        print("All available quality settings:");
        string[] names = QualitySettings.names;
        string qualities = "";
        for (int i = 0; i < names.Length; i++) {
            qualities += i.ToString() + ". " + names[i] + " / ";
        }
        print("Current quality setting = " + QualitySettings.GetQualityLevel().ToString());

        // preallocate buffer with maximum possible message size
        buffersize = 2048; // Math.Max(4, Math.Max(4*nqpos, 28*nmocap));
        buffer = new byte[buffersize];

        // start listening for connections
        listener = new TcpListener(System.Net.IPAddress.Parse(tcpAddress), tcpPort);
        listener.Start();
    }

    // Update is called once per frame
    void Update()
    {
        MJRemote ext = root.GetComponent<MJRemote>();
        if (ext == null) return;

        // check conection each 0.1 sec
        if (lastcheck + 0.1f < Time.time)
        {
            // broken connection: clear
            if (!CheckConnection())
            {
                client = null;
                stream = null;
            }

            lastcheck = Time.time;
        }

        // not connected: accept connection if pending
        if (client == null || !client.Connected)
        {
            if (listener != null && listener.Pending())
            {
                // make connection
                client = listener.AcceptTcpClient();
                client.NoDelay = true;
                stream = client.GetStream();

                ext.writeSettings(stream);
            }
        }

        if (client != null && client.Connected) {
            foreach (var key in keyCodes) {
                if (Input.GetKeyDown(key)) {
                    keyInputs.Enqueue(key.ToString());
                }
            }
        }

        // data available: handle communication
        while (client != null && client.Connected && stream != null && stream.DataAvailable)
        {
            // get command
            ReadAll(stream, 4);
            int cmd = BitConverter.ToInt32(buffer, 0);

            // process command
            switch ((Command)cmd)
            {
                // GetInput: send lastkey, select, active, refpos[3], refquat[4]
                case Command.GetInput:
                    ext.writeInput(stream);
                    break;

                // GetImage: send 3*width*height bytes
                case Command.GetImage:
                    ext.writeColorImage(stream);
                    break;

                case Command.GetSegmentationImage:
                    ext.writeSegmentationImage(stream);
                    break;

                // SaveSnapshot: no data exchange
                case Command.SaveSnapshot:
                    ext.writeSnapshot();
                    break;

                // SaveVideoframe: no data exchange
                case Command.SaveVideoframe:
                    ext.writeVideo();
                    
                    break;

                // SetCamera: receive camera index
                case Command.SetCamera:
                    ext.setCamera(stream);
                    break;

                // SetQpos: receive qpos vector
                case Command.SetQpos:
                    ext.setQpos(stream);
                    break;

                // SetMocap: receive mocap_pos and mocap_quat vectors
                case Command.SetMocap:
                    ext.setMocap(stream);

                    break;
                case Command.ChangeWorld:
                    ReadAll(stream, 4);
                    int strlen = BitConverter.ToInt32(buffer, 0);
                    ReadAll(stream, strlen);
                    string path = System.Text.Encoding.UTF8.GetString(buffer, 0, strlen);
                    Debug.Log("Change World");
                    GameObject.Find("Importer").GetComponent<MJImport>().Import(path); //Kills MJRemote

                    // For Furniture Assembly Environment: clear key input queue
                    keyInputs.Clear();

                    return; //don't processess more commands until next frame

                case Command.GetWorldInfo:
                    Debug.Log("Write Info");
                    ext.writeSettings(stream);
                    break;

                case Command.RandomizeAppearance:
                    ext.randomizeAppearance();
                    break;

                case Command.GetDepthImage:
                    ext.writeDepthImage(stream);
                    break;

                // For Furniture Assembly Environment
                case Command.SetResolution:
                    ext.setResolution(stream);
                    break;

                case Command.SetGeomPos:
                    ext.setGeomPos(stream);
                    break;

                case Command.GetKeyInput:
                    string key = "None";
                    if (keyInputs.Count > 0)
                        key = keyInputs.Dequeue();
                    ext.getKeyInput(stream, key);
                    break;

                case Command.SetBackground:
                    ext.setBackground(stream);
                    break;

                case Command.SetGraphicsQuality:
                    ext.setGraphicsQuality(stream);
                    break;
            }
        }

    }

    // GUI
    private void OnGUI()
    {
        // set style once
        if (style == null)
        {
            style = GUI.skin.textField;
            style.normal.textColor = Color.white;

            // scale font size with DPI
            if (Screen.dpi < 100)
                style.fontSize = 14;
            else if (Screen.dpi > 300)
                style.fontSize = 34;
            else
                style.fontSize = Mathf.RoundToInt(14 + (Screen.dpi - 100.0f) * 0.1f);
        }

        // show connected status
        if (client != null && client.Connected)
            GUILayout.Label("Connected", style);
        else
            GUILayout.Label("Waiting on port "+ tcpPort, style);
    }

    // read requested number of bytes from socket
    void ReadAll(NetworkStream stream, int n)
    {
        int i = 0;
        while (i < n)
            i += stream.Read(buffer, i, n - i);
    }


    // check if connection is still alive
    private bool CheckConnection()
    {
        try
        {
            if (client != null && client.Client != null && client.Client.Connected)
            {
                if (client.Client.Poll(0, SelectMode.SelectRead))
                {
                    if (client.Client.Receive(buffer, SocketFlags.Peek) == 0)
                        return false;
                    else
                        return true;
                }
                else
                    return true;
            }
            else
                return false;
        }
        catch
        {
            return false;
        }
    }
}
