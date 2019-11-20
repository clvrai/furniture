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

using System;
using System.IO;
using System.Net.Sockets;

using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;
using UnityEngine.SceneManagement;


public class MJRemote : MonoBehaviour
{
    // script options
    public string modelFile = "";

    // offscreen rendering
    OffscreenRenderer off_render = null;

    static int snapshots = 0;
    FileStream videofile = null;

    // data from plugin
    int nqpos = 0;
    int nmocap = 0;
    int ncamera = 0;
    int nobject = 0;
    GameObject[] objects;
    Color selcolor;
    public GameObject root = null;
    Camera thecamera = null;
    Camera dummycamera = null;
    float[] camfov;

    int camindex = -1;

    byte[] buffer;
    int buffersize = 0;

    // input state
    float lastx = 0;        // updated each frame
    float lasty = 0;        // updated each frame
    float lasttime = 0;     // updated on click
    int lastbutton = 0;     // updated on click
    int lastkey = 0;        // cleared on send

    // For Furniture Assembly Environment
    Dictionary<string, Vector3> modifiedObjects = null;
    List<GameObject> backgrounds = null;


    // convert transform from plugin to GameObject
    static unsafe void SetTransform(GameObject obj, MJP.TTransform transform)
    {
        Quaternion q = new Quaternion(0, 0, 0, 1);
        q.SetLookRotation(
            new Vector3(transform.yaxis[0], -transform.yaxis[2], transform.yaxis[1]),
            new Vector3(-transform.zaxis[0], transform.zaxis[2], -transform.zaxis[1])
        );

        obj.transform.localPosition = new Vector3(-transform.position[0], transform.position[2], -transform.position[1]);
        obj.transform.localRotation = q;
        obj.transform.localScale = new Vector3(transform.scale[0], transform.scale[2], transform.scale[1]);
    }


    // convert transform from plugin to Camera
    static unsafe void SetCamera(Camera cam, MJP.TTransform transform)
    {
        Quaternion q = new Quaternion(0, 0, 0, 1);
        q.SetLookRotation(
            new Vector3(transform.zaxis[0], -transform.zaxis[2], transform.zaxis[1]),
            new Vector3(-transform.yaxis[0], transform.yaxis[2], -transform.yaxis[1])
        );

        cam.transform.localPosition = new Vector3(-transform.position[0], transform.position[2], -transform.position[1]);
        cam.transform.localRotation = q;
    }

    private void OnDestroy()
    {

    }

    // initialize
    unsafe void Start()
    {
        // For Furniture Assembly Environment: record geom positions
        modifiedObjects = new Dictionary<string, Vector3>();
        backgrounds = new List<GameObject>();
        foreach (GameObject child in SceneManager.GetActiveScene().GetRootGameObjects()) {
            if (child.name.StartsWith("Background_")) {
                backgrounds.Add(child);
                child.SetActive(false);
            }
        }

        //material_random_parameters["Plastic (Instance)"] = new RandomColor(0.0f, 1.0f, 0.0f, 1.0f, 0.2f, 1.0f);

        // set selection color
        selcolor = new Color(0.5f, 0.5f, 0.5f, 1);

        // preallocate buffer with maximum possible message size
        buffersize = 2048; // Math.Max(4, Math.Max(4*nqpos, 28*nmocap));
        buffer = new byte[buffersize];

        // initialize plugin
        //   MJP.Initialize();
        //  MJP.LoadModel(Application.streamingAssetsPath + "/" + modelFile);

        // get number of renderable objects, allocate map
        MJP.TSize size;
        MJP.GetSize(&size);
        nqpos = size.nqpos;
        nmocap = size.nmocap;
        ncamera = size.ncamera;
        nobject = size.nobject;
        objects = new GameObject[nobject];

        // get root
        //root = GameObject.Find("MuJoCo");
        if (root == null)
            throw new System.Exception("MuJoCo root object not found");

        root.transform.localPosition = transform.localPosition;
        root.transform.localRotation = transform.localRotation;
        root.transform.localScale = transform.localScale;

        // get camera under root
        int nchild = root.transform.childCount;
        for (int i = 0; i < nchild; i++)
        {
            thecamera = root.transform.GetChild(i).gameObject.GetComponent<Camera>();
            if (thecamera != null)
            {
                // thecamera.enabled = false;
                break;
            }
        }
        if (thecamera == null)
            throw new System.Exception("No camera found under MuJoCo root object");

        // make map of renderable objects
        for (int i = 0; i < nobject; i++)
        {
            // get object name
            StringBuilder name = new StringBuilder(100);
            MJP.GetObjectName(i, name, 100);

            // find corresponding GameObject
            for (int j = 0; j < nchild; j++)
                if (root.transform.GetChild(j).name == name.ToString())
                {
                    objects[i] = root.transform.GetChild(j).gameObject;
                    break;
                }

            // set initial state
            if (objects[i])
            {
                MJP.TTransform transform;
                int visible;
                int selected;
                MJP.GetObjectState(i, &transform, &visible, &selected);
                SetTransform(objects[i], transform);
                objects[i].SetActive(visible > 0);
            }
        }

        // get camera fov and offscreen resolution
        camfov = new float[ncamera + 1];
        int offwidth = 1280;
        int offheight = 720;

        for (int i = -1; i < ncamera; i++)
        {
            MJP.TCamera cam;
            MJP.GetCamera(i, &cam);
            camfov[i + 1] = cam.fov;

            // plugin returns offscreen width and height for all cameras
            offwidth = cam.width;
            offheight = cam.height;
        }

        //TODO: The dummy camera and camera added by mjonline import should be merged together
        GameObject camobj = GameObject.Find("DummyCamera");
        if (camobj != null)
        {
            dummycamera = camobj.GetComponent<Camera>();
            dummycamera.enabled = true;
        }

        // prepare offscreen rendering
        off_render = new OffscreenRenderer(offwidth, offheight);

        // synchronize time
        MJP.SetTime(Time.time);

        //randomizeAppearance();

        Debug.Log("New simulation init'd " + offwidth + "x" + offheight);
    }


    internal string getPrefix(string name)
    {
        var i = name.Length-1;
        for (; i >= 0; i--)
            if (!char.IsNumber(name[i]))         
                return name.Substring(0, i+1);
         
        return name;
    }

    internal void randomizeAppearance()
    {
        var clustering_map = new SortedDictionary<string, Tuple<string, List<GameObject>>>();

        foreach (Transform t in root.transform)
        {
            string prefix = getPrefix(t.gameObject.name);

            Renderer render = t.gameObject.GetComponent<Renderer>();
            if (render != null)
            {
                String matname = render.material.name;

                if (clustering_map.ContainsKey(prefix))
                {
                    if (clustering_map[prefix].Item1 != matname) Debug.LogWarning(t.gameObject.name + "has inconsistent materials with other objects in its group");
                }
                else
                {
                    clustering_map.Add(prefix, new Tuple<string, List<GameObject>>(matname, new List<GameObject>()));
                }

                clustering_map[prefix].Item2.Add(t.gameObject);
            }
        }

        foreach (var tup in clustering_map.Values)
        {
            RandomColor randomizer;
            if (DoorGym.Configuration.MaterialParameters.TryGetValue(tup.Item1, out randomizer))
            {
                var color = randomizer.Next();
                foreach (var gobj in tup.Item2)
                {
                    InstancedColor instancer = gobj.GetComponent<InstancedColor>();
                    if (instancer != null)
                    {
                        instancer.Diffuse = color;
                    }
                }
            }
        }
    }

    // per-frame mouse input; called from Update
    unsafe void ProcessMouse()
    {
        // get modifiers
        bool alt = Input.GetKey(KeyCode.LeftAlt) || Input.GetKey(KeyCode.RightAlt);
        bool shift = Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift);
        bool control = Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.RightControl);

        // get button pressed, swap left-right on alt
        int buttonpressed = 0;
        if (Input.GetMouseButton(0))           // left
            buttonpressed = (alt ? 2 : 1);
        if (Input.GetMouseButton(1))           // right
            buttonpressed = (alt ? 1 : 2);
        if (Input.GetMouseButton(2))           // middle
            buttonpressed = 3;

        // get button click, swap left-right on alt
        int buttonclick = 0;
        if (Input.GetMouseButtonDown(0))       // left
            buttonclick = (alt ? 2 : 1);
        if (Input.GetMouseButtonDown(1))       // right
            buttonclick = (alt ? 1 : 2);
        if (Input.GetMouseButtonDown(2))       // middle
            buttonclick = 3;

        // click
        if (buttonclick > 0)
        {
            // set perturbation state
            int newstate = 0;
            if (control)
            {
                // determine new perturbation state
                if (buttonclick == 1)
                    newstate = 2;              // rotate
                else if (buttonclick == 2)
                    newstate = 1;              // move

                // get old perturbation state
                MJP.TPerturb current;
                MJP.GetPerturb(&current);

                // syncronize if starting perturbation now
                if (newstate > 0 && current.active == 0)
                    MJP.PerturbSynchronize();
            }
            MJP.PerturbActive(newstate);

            // process double-click
            if (buttonclick == lastbutton && Time.fixedUnscaledTime - lasttime < 0.25)
            {
                // relative screen position and aspect ratio
                float relx = Input.mousePosition.x / Screen.width;
                float rely = Input.mousePosition.y / Screen.height;
                float aspect = (float)Screen.width / (float)Screen.height;

                // left: select body
                if (buttonclick == 1)
                    MJP.PerturbSelect(relx, rely, aspect);

                // right: set lookat
                else if (buttonclick == 2)
                    MJP.CameraLookAt(relx, rely, aspect);
            }

            // save mouse state
            lastx = Input.mousePosition.x;
            lasty = Input.mousePosition.y;
            lasttime = Time.fixedUnscaledTime;
            lastbutton = buttonclick;
        }

        // left or right drag: manipulate camera or perturb
        if (buttonpressed == 1 || buttonpressed == 2)
        {
            // compute relative displacement and modifier
            float reldx = (Input.mousePosition.x - lastx) / Screen.height;
            float reldy = (Input.mousePosition.y - lasty) / Screen.height;
            int modifier = (shift ? 1 : 0);

            // perturb
            if (control)
            {
                if (buttonpressed == 1)
                    MJP.PerturbRotate(reldx, -reldy, modifier);
                else
                    MJP.PerturbMove(reldx, -reldy, modifier);
            }

            // camera
            else
            {
                if (buttonpressed == 1)
                    MJP.CameraRotate(reldx, -reldy);
                else
                    MJP.CameraMove(reldx, -reldy, modifier);
            }
        }

        // middle drag: zoom camera
        if (buttonpressed == 3)
        {
            float reldy = (Input.mousePosition.y - lasty) / Screen.height;
            MJP.CameraZoom(-reldy);
        }

        // scroll: zoom camera
        if (Input.mouseScrollDelta.y != 0)
            MJP.CameraZoom(-0.05f * Input.mouseScrollDelta.y);

        // save position
        lastx = Input.mousePosition.x;
        lasty = Input.mousePosition.y;

        // release left or right: stop perturb
        if (Input.GetMouseButtonUp(0) || Input.GetMouseButtonUp(1))
            MJP.PerturbActive(0);
    }


    // update Unity representation of MuJoCo model
    unsafe private void UpdateModel()
    {
        MJP.TTransform transform;

        // update object states
        for (int i = 0; i < nobject; i++)
            if (objects[i])
            {
                // set transform and visibility
                int visible;
                int selected;
                MJP.GetObjectState(i, &transform, &visible, &selected);

                // For Furniture Assembly Environment: apply new geom position
                if (modifiedObjects.ContainsKey(objects[i].name)) {
                    objects[i].transform.position = modifiedObjects[objects[i].name];
                } else { 
                    SetTransform(objects[i], transform);
                }
                objects[i].SetActive(visible > 0);

                // set emission color
                if (selected > 0)
                    objects[i].GetComponent<Renderer>().material.SetColor("_EmissionColor", selcolor);
                else
                    objects[i].GetComponent<Renderer>().material.SetColor("_EmissionColor", Color.black);
            }

        // update camera
        MJP.GetCameraState(camindex, &transform);
        SetCamera(thecamera, transform);
        thecamera.fieldOfView = camfov[camindex + 1];
    }


    // per-frame update
    unsafe void Update()
    {
        // mouse interaction
        ProcessMouse();
        UpdateModel();


    }


    // GUI
    private void OnGUI()
    {
        // save lastkey
        if (Event.current.isKey)
            lastkey = (int)Event.current.keyCode;
    }


    // cleanup
    void OnApplicationQuit()
    {
        if (videofile != null)
            videofile.Close();
    }


    public void writeSettings(NetworkStream stream)
    {
        // send 20 bytes: nqpos, nmocap, ncamera, width, height
        stream.Write(BitConverter.GetBytes(nqpos), 0, 4);
        stream.Write(BitConverter.GetBytes(nmocap), 0, 4);
        stream.Write(BitConverter.GetBytes(ncamera), 0, 4);
        stream.Write(BitConverter.GetBytes(off_render.Width), 0, 4);
        stream.Write(BitConverter.GetBytes(off_render.Height), 0, 4);
    }

    public unsafe void writeInput(NetworkStream stream)
    {
        MJP.TPerturb perturb;
        MJP.GetPerturb(&perturb);
        stream.Write(BitConverter.GetBytes(lastkey), 0, 4);
        stream.Write(BitConverter.GetBytes(perturb.select), 0, 4);
        stream.Write(BitConverter.GetBytes(perturb.active), 0, 4);
        stream.Write(BitConverter.GetBytes(perturb.refpos[0]), 0, 4);
        stream.Write(BitConverter.GetBytes(perturb.refpos[1]), 0, 4);
        stream.Write(BitConverter.GetBytes(perturb.refpos[2]), 0, 4);
        stream.Write(BitConverter.GetBytes(perturb.refquat[0]), 0, 4);
        stream.Write(BitConverter.GetBytes(perturb.refquat[1]), 0, 4);
        stream.Write(BitConverter.GetBytes(perturb.refquat[2]), 0, 4);
        stream.Write(BitConverter.GetBytes(perturb.refquat[3]), 0, 4);
        lastkey = 0;
    }

    public void writeColorImage(NetworkStream stream)
    {
        Texture2D tex = off_render.RenderColor(thecamera);
        byte[] data = tex.GetRawTextureData();
        int size = off_render.GetColorBufferSize();
        stream.Write(data, 0, size);
    }

    internal void writeDepthImage(NetworkStream stream)
    {
        Texture2D tex = off_render.ReadDepth(thecamera);
        byte[] data = tex.GetRawTextureData();
        int size = off_render.GetColorBufferSize();
        stream.Write(data, 0, size);
    }

    public void writeSegmentationImage(NetworkStream stream)
    {
        stream.Write(off_render.RenderSegmentation(thecamera).GetRawTextureData(), 0, off_render.GetSegmentationBufferSize());
    }

    public void writeSnapshot()
    {
        byte[] bytes = off_render.RenderColor(thecamera).EncodeToPNG();
        File.WriteAllBytes(Application.streamingAssetsPath + "/../../" + "img_" +
                           snapshots + ".png", bytes);
        snapshots++;
    }

    public void writeVideo()
    {
        if (videofile == null)
            videofile = new FileStream(Application.streamingAssetsPath + "/../../" + "video.raw",
                                       FileMode.Create, FileAccess.Write);

        videofile.Write(off_render.RenderColor(thecamera).GetRawTextureData(), 0, off_render.GetColorBufferSize());
    }

    public void setCamera(NetworkStream stream)
    {
        ReadAll(stream, 4);
        camindex = BitConverter.ToInt32(buffer, 0);
        camindex = Math.Max(-1, Math.Min(ncamera - 1, camindex));
    }

    public unsafe void setQpos(NetworkStream stream)
    {
        if (nqpos > 0)
        {
            ReadAll(stream, 4 * nqpos);
            fixed (byte* qpos = buffer)
            {
                MJP.SetQpos((float*)qpos);
            }
            MJP.Kinematics();
            UpdateModel();
        }
    }

    public unsafe void setMocap(NetworkStream stream)
    {
        if (nmocap > 0) {
            ReadAll(stream, 28 * nmocap);
            fixed (byte* pos = buffer, quat = &buffer[12 * nmocap]) {
                MJP.SetMocap((float*)pos, (float*)quat);
            }
            MJP.Kinematics();
            UpdateModel();
        }
    }

    // read requested number of bytes from socket
    void ReadAll(NetworkStream stream, int n)
    {
        int i = 0;
        while (i < n)
            i += stream.Read(buffer, i, n - i);
    }

    // For Furniture Assembly Environment
    public unsafe void setResolution(NetworkStream stream)
    {
        ReadAll(stream, 4);
        int width = BitConverter.ToInt32(buffer, 0);
        ReadAll(stream, 4);
        int height = BitConverter.ToInt32(buffer, 0);
        off_render.SetResolution(width, height);
    }

    public unsafe void setGeomPos(NetworkStream stream)
    {
        ReadAll(stream, 4);
        float x = BitConverter.ToSingle(buffer, 0);
        ReadAll(stream, 4);
        float y = BitConverter.ToSingle(buffer, 0);
        ReadAll(stream, 4);
        float z = BitConverter.ToSingle(buffer, 0);
        Vector3 pos = new Vector3(-x, z, -y);

        ReadAll(stream, 4);
        int strlen = BitConverter.ToInt32(buffer, 0);
        ReadAll(stream, strlen);
        string name = System.Text.Encoding.UTF8.GetString(buffer, 0, strlen);

        GameObject cursor0 = GameObject.Find(name);
        cursor0.transform.position = pos;
        modifiedObjects[name] = pos;
    }

    public unsafe void setBackground(NetworkStream stream)
    {
        ReadAll(stream, 4);
        int strlen = BitConverter.ToInt32(buffer, 0);
        ReadAll(stream, strlen);
        string name = System.Text.Encoding.UTF8.GetString(buffer, 0, strlen);
        print(string.Format("Change background {0}", name));
        foreach (GameObject child in backgrounds) {
            if (child.name == "Background_" + name) {
                child.SetActive(true);
            } else {
                child.SetActive(false);
            }
        }
    }

    public unsafe void setGraphicsQuality(NetworkStream stream)
    {
        ReadAll(stream, 4);
        int quality = BitConverter.ToInt32(buffer, 0);
        if (quality == -1) {
            string[] names = QualitySettings.names;
            quality = names.Length - 1;
        }
        int currentQuality = QualitySettings.GetQualityLevel();

        print(String.Format("Change the quality level from {0} to {1}", currentQuality, quality));
        QualitySettings.SetQualityLevel(quality);
    }

    public unsafe void getKeyInput(NetworkStream stream, string keyCode)
    {
        string key = string.Format("{0,-6}", keyCode);
        byte[] data = new byte[6];
        System.Text.Encoding.UTF8.GetBytes(key, 0, 6, data, 0);
        stream.Write(data, 0, 6);
    }

}