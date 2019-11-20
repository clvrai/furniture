/*
Original work Copyright 2019 Roboti LLC

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

using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;
using UnityEngine.SceneManagement;


public class MJInternalSimulation : MonoBehaviour 
{

    // prevent repeated instances
    private static MJInternalSimulation instance;
    private MJInternalSimulation() {}
    public static MJInternalSimulation Instance
    {
        get 
        {
            if( instance==null )
            {
                instance = new MJInternalSimulation();
                return instance;
            }
            else
                throw new System.Exception("MJInternalSimulation can only be instantiated once");
        }
    }


    // script options
    public string modelFile = "";

    // GUI settings
    static GUIStyle style = null;
    bool pause = false;
    bool record = false;
    int showhelp = 1;
    int camindex = -1;


    static int snapshots = 0;
    static int videos = 0;
    FileStream videofile = null;
    float videotime = 0;

    // data from plugin
    int ncamera = 0;
    int nobject = 0;
    GameObject[] objects;
    float[] camfov;
    Color selcolor;
    public GameObject root = null;
    Camera thecamera = null;

    OffscreenRenderer off_render = null;

    // previous mouse state
    float lastx = 0;        // updated each frame
    float lasty = 0;        // updated each frame
    float lasttime = 0;     // updated on click
    int lastbutton = 0;     // updated on click

    
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
    

    // GUI: help
    private void OnGUI()
    {
        // set style once
        if( style==null )
        {
            style = GUI.skin.textField;
            style.normal.textColor = Color.white;

            // scale font size with DPI
            if( Screen.dpi<100 )
                style.fontSize = 14;
            else if( Screen.dpi>300 )
                style.fontSize = 34;
            else
                style.fontSize = Mathf.RoundToInt(14 + (Screen.dpi-100.0f)*0.1f);
        }

        // long help
        const string helpLeft = 
            "F1\n\n" +
            "Space\n" +
            "Back Space\n" +
            "G\n" +
            "E\n" +
            "L\n\n" +
            "Shift 0-9\n" + 
            "0-9\n" +
            "Esc\n" +
            "S\n" +
            "R\n\n" +
            "Mouse\n" +
            "Ctrl Mouse\n" +
            "Shift Mouse\n" +
            "Left Drag\n" +
            "Right Drag\n" +
            "Middle Drag/Scroll   \n" +
            "Left Double Click   \n" +
            "Right Double Click   ";

        const string helpRight = 
            "Help\n\n" +
            "Pause\n" +
            "Reset\n" +
            "Gravity\n" +
            "Equality\n" +
            "Limits\n\n" +
            "Load Scene\n" + 
            "Model Camera\n" +
            "Free Camera\n" +
            "Snapshot\n" +
            "Record Video\n\n" +
            "Adjust Camera\n" +
            "Pertrub\n" +
            "Other Plane\n" +
            "Rotate\n" +
            "Translate\n" +
            "Zoom\n" +
            "Select\n" +
            "Look At";

        // brief help
        if( showhelp==1 )
            GUILayout.Label("F1: Help", style);

        // full help
        else if( showhelp==2 )
        {
            GUILayout.BeginHorizontal();
            GUILayout.BeginVertical();
            GUILayout.Label(helpLeft, style);
            GUILayout.EndVertical();
            GUILayout.BeginVertical();
            GUILayout.Label(helpRight, style);
            GUILayout.EndVertical();
            GUILayout.EndHorizontal();
        }

        // recording
        if( record )
        {
            int oldsize = style.fontSize;
            style.fontSize = Mathf.RoundToInt(1.3f*oldsize);
            style.normal.textColor = Color.red;
            GUILayout.Label("RECORD", style);
            style.fontSize = oldsize;
            style.normal.textColor = Color.white;
        }
    }


    // initialize
    unsafe void Start ()
    {
        // set selection color
        selcolor = new Color(0.5f, 0.5f, 0.5f, 1);

        // initialize plugin
      ////  MJP.Initialize();
      //  MJP.LoadModel(Application.streamingAssetsPath + "/" + modelFile);

        // get number of renderable objects, allocate map
        MJP.TSize size;
        MJP.GetSize(&size);
        ncamera = size.ncamera;
        nobject = size.nobject;
        objects = new GameObject[nobject];

        // get root
       // root = GameObject.Find("MuJoCo");
        if( root==null )
            throw new System.Exception("MuJoCo root object not found");

        // get camera under root
        int nchild = root.transform.childCount;
        for( int i=0; i<nchild; i++ )
        {
            thecamera = root.transform.GetChild(i).gameObject.GetComponent<Camera>();
            if( thecamera!=null )
                break;
        }
        if( thecamera==null )
            throw new System.Exception("No camera found under MuJoCo root object");

        // make map of renderable objects
        for( int i=0; i<nobject; i++ )
        {
            // get object name
            StringBuilder name = new StringBuilder(100);
            MJP.GetObjectName(i, name, 100);

            // find corresponding GameObject
            for( int j=0; j<nchild; j++ )
                if( root.transform.GetChild(j).name == name.ToString() )
                {
                    objects[i] = root.transform.GetChild(j).gameObject;
                    break;
                }

            // set initial state
            if( objects[i] )
            {
                MJP.TTransform transform;
                int visible;
                int selected;
                MJP.GetObjectState(i, &transform, &visible, &selected);
                SetTransform(objects[i], transform);
                objects[i].SetActive( visible > 0 );
            }
        }

        int offwidth  = 1280;
        int offheight = 720;

        // get camera fov and offscreen resolution
        camfov = new float[ncamera+1];
        for( int i=-1; i<ncamera; i++ )
        {
            MJP.TCamera cam;
            MJP.GetCamera(i, &cam);
            camfov[i+1] = cam.fov;

            // plugin returns offscreen width and height for all cameras
            offwidth = cam.width;
            offheight = cam.height;
        }

        off_render = new OffscreenRenderer(offwidth, offheight);

        // synchronize time
        MJP.SetTime(Time.time);
        videotime = Time.time;
    }
    

    // physics update
    unsafe void FixedUpdate()
    {
        // sync if too late
        float time;
        MJP.GetTime(&time);
        if( time+0.2<Time.time )
        {
            MJP.SetTime(Time.time);
            videotime = Time.time;
        }

        // simulate
        int reset;
        MJP.Simulate(Time.time, (pause ? 1 : 0), &reset);

        // sync if internal reset
        if( reset>0 )
        {
            MJP.SetTime(Time.time);
            videotime = Time.time;
        }

        // save video frames at 60Hz
        if( record && Time.time-videotime>1.0f/60.0f )
        {
           
            videofile.Write(off_render.RenderColor(thecamera).GetRawTextureData(), 0, off_render.GetColorBufferSize());
            videotime = Time.time;
        }
    }

    // per-frame keyboard input; called from Update
    unsafe void ProcessKeyboard ()
    {
        // F1: toggle help
        if( Input.GetKeyDown(KeyCode.F1) )
        {
            showhelp++;
            if( showhelp>2 )
                showhelp = 0;
        }

        // G: toggle gravity
        else if( Input.GetKeyDown("g") )
        {
            MJP.TOption opt;
            MJP.GetOption(&opt);
            opt.gravity = 1-opt.gravity;
            MJP.SetOption(&opt);
        }

        // E: toggle equality
        else if( Input.GetKeyDown("e") )
        {
            MJP.TOption opt;
            MJP.GetOption(&opt);
            opt.equality = 1-opt.equality;
            MJP.SetOption(&opt);
        }

        // L: toggle limit
        else if( Input.GetKeyDown("l") )
        {
            MJP.TOption opt;
            MJP.GetOption(&opt);
            opt.limit = 1-opt.limit;
            MJP.SetOption(&opt);
        }

        // R: toggle video recording
        else if( Input.GetKeyDown("r") )
        {
            // recording: close
            if( record )
            {
                record = false;
                videofile.Close();
                videofile = null;
            }

            // not recording: open
            else
            {
                record = true;
                videofile = new FileStream(Application.streamingAssetsPath + "/../../" + "video_" + videos + ".raw",
                                           FileMode.Create, FileAccess.Write);
                videos++;
            }
        }

        // S: save snapshot
        else if( Input.GetKeyDown("s") )
        {
            byte[] bytes;

            if (Input.GetKey(KeyCode.LeftShift) == true)
            {
                bytes = off_render.RenderSegmentation(thecamera).EncodeToPNG();
            }
            else
            {

                bytes = off_render.RenderColor(thecamera).EncodeToPNG();
            }

            File.WriteAllBytes(Application.streamingAssetsPath + "/../../" + "img_" + snapshots + ".png", bytes);
            snapshots++;
        }


        // backspace: reset and sync
        else if( Input.GetKeyDown(KeyCode.Backspace) )
        {
            MJP.Reset();
            MJP.SetTime(Time.time);
            videotime = Time.time;
        }

        // space: toggle pause and sync
        else if( Input.GetKeyDown(KeyCode.Space) )
        {
            pause = !pause;
            MJP.SetTime(Time.time);
            videotime = Time.time;
        }

        // Esc: main camera
        else if( Input.GetKeyDown(KeyCode.Escape) )
        {
            camindex = -1;
        }

        // load scene or set model camera
        else for( int i=0; i<=9; i++ )
            if( Input.GetKeyDown(i.ToString()) )
            {
                // load scene: runtime only
                if( Input.GetKey(KeyCode.LeftShift) || 
                    Input.GetKey(KeyCode.RightShift) )
                {
                    #if !UNITY_EDITOR
                    if( i<SceneManager.sceneCountInBuildSettings )
                        SceneManager.LoadScene(i, LoadSceneMode.Single);
                    #endif
                }

                // set camera
                else
                {
                    if( i<ncamera )
                        camindex = i;
                }
            }
    }


    // mouse input
    unsafe void ProcessMouse()
    {
        // get modifiers
        bool alt = Input.GetKey(KeyCode.LeftAlt) || Input.GetKey(KeyCode.RightAlt);
        bool shift = Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift);
        bool control = Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.RightControl);

        // get button pressed, swap left-right on alt
        int buttonpressed = 0;
        if( Input.GetMouseButton(0) )           // left
            buttonpressed = (alt ? 2 : 1);
        if( Input.GetMouseButton(1) )           // right
            buttonpressed = (alt ? 1 : 2);
        if( Input.GetMouseButton(2) )           // middle
            buttonpressed = 3;

        // get button click, swap left-right on alt
        int buttonclick = 0;
        if( Input.GetMouseButtonDown(0) )       // left
            buttonclick = (alt ? 2 : 1);
        if( Input.GetMouseButtonDown(1) )       // right
            buttonclick = (alt ? 1 : 2);
        if( Input.GetMouseButtonDown(2) )       // middle
            buttonclick = 3;

        // click
        if( buttonclick>0 )
        {
            // set perturbation state
            int newstate = 0;
            if( control )
            {
                // determine new perturbation state
                if( buttonclick==1 )
                    newstate = 2;              // rotate
                else if( buttonclick==2 )
                    newstate = 1;              // move

                // get old perturbation state
                MJP.TPerturb current;
                MJP.GetPerturb(&current);

                // syncronize if starting perturbation now
                if( newstate>0 && current.active==0 )
                    MJP.PerturbSynchronize();
            }
            MJP.PerturbActive(newstate);

            // process double-click
            if( buttonclick==lastbutton && Time.fixedUnscaledTime-lasttime<0.25 )
            {
                // relative screen position and aspect ratio
                float relx = Input.mousePosition.x / Screen.width;
                float rely = Input.mousePosition.y / Screen.height;
                float aspect = (float)Screen.width / (float)Screen.height;

                // left: select body
                if( buttonclick==1 )
                    MJP.PerturbSelect(relx, rely, aspect);

                // right: set lookat
                else if( buttonclick==2 )
                    MJP.CameraLookAt(relx, rely, aspect);
            }

            // save mouse state
            lastx = Input.mousePosition.x;
            lasty = Input.mousePosition.y;
            lasttime = Time.fixedUnscaledTime;
            lastbutton = buttonclick;
        }

        // left or right drag: manipulate camera or perturb
        if( buttonpressed==1 || buttonpressed==2 )
        {
            // compute relative displacement and modifier
            float reldx = (Input.mousePosition.x - lastx) / Screen.height;
            float reldy = (Input.mousePosition.y - lasty) / Screen.height;
            int modifier = (shift ? 1 : 0);

            // perturb
            if( control )
            {
                if (buttonpressed == 1)
                    MJP.PerturbRotate(reldx, -reldy, modifier);
                else
                    MJP.PerturbMove(reldx, -reldy, modifier);
            }

            // camera
            else
            {
                if( buttonpressed==1 )
                    MJP.CameraRotate(reldx, -reldy);
                else
                    MJP.CameraMove(reldx, -reldy, modifier);
            }
        }

        // middle drag: zoom camera
        if( buttonpressed==3 )
        {
            float reldy = (Input.mousePosition.y - lasty) / Screen.height;
            MJP.CameraZoom(-reldy);
        }

        // scroll: zoom camera
        if( Input.mouseScrollDelta.y!=0 )
            MJP.CameraZoom(-0.05f * Input.mouseScrollDelta.y);

        // save position
        lastx = Input.mousePosition.x;
        lasty = Input.mousePosition.y;

        // release left or right: stop perturb
        if( Input.GetMouseButtonUp(0) || Input.GetMouseButtonUp(1) )
            MJP.PerturbActive(0);
    }

    
    // per-frame update
    unsafe void Update ()
    {
        // process input
        ProcessKeyboard();
        if( Input.mousePresent )
            ProcessMouse();

        // update object states
        MJP.TTransform transform;
        for( int i=0; i<nobject; i++ )
            if( objects[i] )
            {
                int visible;
                int selected;
                MJP.GetObjectState(i, &transform, &visible, &selected);
                SetTransform(objects[i], transform);
                objects[i].SetActive(visible>0);

                // set emission color
                if( selected>0 )
                    objects[i].GetComponent<Renderer>().material.SetColor("_EmissionColor", selcolor);
                else
                    objects[i].GetComponent<Renderer>().material.SetColor("_EmissionColor", Color.black);
            }

        // update camera
        MJP.GetCameraState(camindex, &transform);
        SetCamera(thecamera, transform);
        thecamera.fieldOfView = camfov[camindex+1];
    }


    // cleanup
    void OnApplicationQuit()
    {
        // free plugin
        MJP.Close();

        // close file
        if( videofile!=null )
            videofile.Close();

        // free render texture
       
    }
}
