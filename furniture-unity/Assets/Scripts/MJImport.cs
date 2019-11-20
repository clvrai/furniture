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


using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;


using System.IO;
using System.Xml;


using System.Collections.Generic;
using System;

public class MJImport : MonoBehaviour
{

    // script options
    private string modelFile = null;
    private string fileName = "";
    public bool recomputeUV = false;
    public bool recomputeNormal = false;
    public bool importTexture = true;
    public bool enableRemote = false;
    public string tcpAddress = "127.0.0.1";
    public int tcpPort = 1050;
    public bool enable_rendering = false;

    private readonly bool noVSync = true;

    // Unity element arrays
    Texture2D[] textures;
    SortedDictionary<int, Material> materials; // (mujoco material id) -> Material
    GameObject[] objects;
    GameObject root = null;

    SortedDictionary<string, string> assetMap;
    SortedDictionary<string, Vector3> scaleMap;

    SortedDictionary<string, byte> segmentIdMap;
    SortedDictionary<string, byte> meshSegmentMap;

    Matrix4x4 importSwap;

    Vector3 eulerRotation = new Vector3(0,0,0);
    Vector3 mirror = new Vector3(1, 1, 1);

    Material default_material = null;

    void OnGUI()
    {
        /*
        if (GUI.Button(new Rect(10, 90, 100, 30), "Import"))
        {
            RunImport();
        }
        */
    }

    // present GUI, get options, run importer
    void Start()
    {
        default_material = Resources.Load<Material>("Materials/Default");
        //TODO: take IP arguments (beinding address? port number? from command line

        // tcpAddress = EditorGUILayout.TextField("TCP Address", tcpAddress);
        // tcpPort = EditorGUILayout.IntField("TCP Port", tcpPort);
        // noVSync = EditorGUILayout.Toggle("Disable V-Sync", noVSync);

        // For Furniture Assembly Environment: change the path of StreamingAssets
        modelFile = Application.streamingAssetsPath + "/default.xml";
        transform.localScale.Set(transform.localScale.x, transform.localScale.x, transform.localScale.x);
        assetMap = new SortedDictionary<string, string>();
        scaleMap = new SortedDictionary<string, Vector3>();
        materials = new SortedDictionary<int, Material>();

        meshSegmentMap = new SortedDictionary<string, byte>();
    
        DoorGym.Configuration.Initialize();

        Resources.Load("Packages/com.unity.postprocessing/PostProcessing/PostProcessResources.asset");

        RunImport();
    }




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

    // add camera
    private unsafe void AddCamera()
    {
        if (enable_rendering == false)
        {
            Destroy(GameObject.Find("PreviewCamera"));
        }
        else
        {
            Destroy(GameObject.Find("DummyCamera"));
        }


        // add camera under root
        GameObject camobj = new GameObject("camera");
        camobj.layer = LayerMask.NameToLayer("PostProcessing");

        camobj.transform.parent = root.transform;
        Camera thecamera = camobj.AddComponent<Camera>();

        // For Furniture Assembly Environment: remove SITE from the culling mask
        thecamera.cullingMask = 1 + (1 << 1) + (1 << 2) + (1 << 4) + (1 << 8);
        thecamera.backgroundColor = new Color(1f, 1f, 1f);
        thecamera.clearFlags = CameraClearFlags.SolidColor;

        Shader segshader = Shader.Find("Unlit/SegmentationColor");
        SegmentationShader shadersub = camobj.AddComponent<SegmentationShader>();

        DepthShader shaderdepth = camobj.AddComponent<DepthShader>();


        // For Furniture Assembly Environment: no post process
        //GameObject pp_obj = GameObject.Find("PostPocessing");
        //PostProcessLayer pp_layer = camobj.AddComponent<PostProcessLayer>();
        //var resources= Resources.FindObjectsOfTypeAll<PostProcessResources>();
        //pp_layer.Init(resources[0]);
        //pp_layer.volumeTrigger = camobj.transform;
        //pp_layer.volumeLayer = 1 << LayerMask.NameToLayer("PostProcessing");

        // set field of view, near, far
        MJP.TCamera cam;
        MJP.GetCamera(-1, &cam);
        thecamera.fieldOfView = cam.fov;

        // For Furniture Assembly Environment: set znear and zfar independent to model extent.
        thecamera.nearClipPlane = 0.01f;
        thecamera.farClipPlane = 10f;

        //thecamera.nearClipPlane = cam.znear * this.transform.localScale.x;
        //thecamera.farClipPlane = cam.zfar * this.transform.localScale.x;

        // thecamera.enabled = false;
        //camobj.SetActive(enable_rendering);
        // set transform
        MJP.TTransform transform;
        MJP.GetCameraState(-1, &transform);
        SetCamera(thecamera, transform);
    }


    // import materials
    private unsafe void ImportMaterials(int nmaterial)
    {
        if (default_material == null)
        {
            Debug.Log("Skipped assigning a material because material is missing");
        }
        else
        {
            materials.Add(-1, new Material(default_material));
        }

        // process materials
        for ( int i=0; i<nmaterial; i++ )
        {
            // get material name
            StringBuilder name = new StringBuilder(100);
            MJP.GetElementName(MJP.TElement.MATERIAL, i, name, 100);
            string matname =  name.ToString();

            Material matt = Resources.Load<Material>("Materials/" + matname);

            if ( matt == null )
            {
                Debug.Log("No material for " + matname + " in Resources");

                if (default_material == null)
                {
                    Debug.Log("Skipped assigning a material because material is missing");
                }
                else
                {
                    matt = new Material(default_material);
                }
            }

            if (matt != null)
            {
                materials.Add(i, matt);
            }
        }
    }

    private void BuildAssetDatabase(string file)
    {
        scaleMap.Clear();
        assetMap.Clear();
        meshSegmentMap.Clear();
        materials.Clear();

        Stack<byte> segments = new Stack<byte>();
        segments.Push(0);

        string cwd = Path.GetDirectoryName(file);
        {
            XmlReaderSettings settings = new XmlReaderSettings();
            settings.Async = true;

            using (XmlReader reader = XmlReader.Create(file, settings))
            {
                while (reader.Read())
                {
                    switch (reader.NodeType)
                    {
                        case XmlNodeType.Element:
                            switch(reader.Name)
                            {
                                // Look up the segmentation id by comparing body names to the segmentation id list. 
                                // We keep track of these ids in a stack, so that we can inherit segmentation from 
                                // parent objects without writing the segmentation term over and over again in the XML
                                case "body":
                                    if (reader.HasAttributes)
                                    {
                                        while (reader.MoveToNextAttribute())
                                        {
                                            if (reader.Name == "name")
                                            {
                                                byte segment = 0;
                                                int first_underscore = reader.Value.IndexOf('_');
                                                string segname = first_underscore >= 0 ? reader.Value.Substring(0, first_underscore).ToLower() : reader.Value.ToLower();
                                                //Check if this object's name is in the segmentation list
                                                if( DoorGym.Configuration.SegmentIDMap.TryGetValue(segname, out segment) )
                                                {
                                                    segments.Push(segment);
                                                }
                                                else // Inherit segmentation from the parent if we aren't in the segmentation list
                                                {
                                                    segments.Push(segments.Peek());
                                                }
                                                break;
                                            }
                                        }
                                        reader.MoveToElement();
                                    }
                                    break;

                                   
                                // 'geom' tags is always under 'body', so we can assign the segmentation id by just checking the top of the stack
                                case "geom":
                                    if (reader.HasAttributes)
                                    {
                                        while (reader.MoveToNextAttribute())
                                        {
                                            if (reader.Name == "name" || reader.Name == "mesh")
                                            { 
                                               meshSegmentMap[reader.Value] = segments.Peek();
                                               break;
                                            }
                                        }
                                        reader.MoveToElement();
                                    }
                                    break;

                                // Build a list of meshes here so we can load the mesh from an alternate source or a Unity asset, if needed.
                                // Initally added for testing, currently not used, left it anyway, enjoy.
                                case "mesh":

                                    if (reader.HasAttributes)
                                    {
                                        string meshPath = null;
                                        string meshName = null;
                                        Vector3 scale = new Vector3();
                                        while (reader.MoveToNextAttribute())
                                        {
                                            switch (reader.Name)
                                            {
                                                case "file":
                                                    meshPath = reader.Value;
                                                    break;
                                                case "name":
                                                    if(meshName == null) meshName = reader.Value; //"mesh" gets priority
                                                    break;
                                                case "mesh":
                                                    meshName = reader.Value;
                                                    break;
                                                case "scale":
                                                    string[] words = reader.Value.Split();
                                                    
                                                    scale.Set(System.Convert.ToSingle(words[0]), System.Convert.ToSingle(words[1]), System.Convert.ToSingle(words[2]));
                                                    break;
                                            }
                                        }
                                        if (meshPath != null && meshName != null)
                                        {
                                            assetMap.Add(meshName, meshPath);
                                        }
                                        if (scale.sqrMagnitude != 0 && meshName != null)
                                        {
                                            scaleMap.Add(meshName, scale);
                                        }
                                        reader.MoveToElement();
                                    }
                                    break;

                                case "include":
                                    if (reader.HasAttributes)
                                    {
                                        while (reader.MoveToNextAttribute())
                                        {
                                            if (reader.Name == "file")
                                            {
                                                BuildAssetDatabase(Path.Combine(cwd, reader.Value));
                                                break;
                                            }
                                        }
                                        reader.MoveToElement();
                                    }
                                    //
                                    break;
                            }


                            break;
                        case XmlNodeType.Text:
                        case XmlNodeType.EndElement:
                        default:
                            // Pop the segmentation ID off the stack once we leave the 'body' tag
                            if (reader.Name == "body")
                            {
                                segments.Pop();
                            }
                            break;
                    }
                }
            }
        }

    }

    // import renderable objects
    private unsafe void ImportObjects(int nobject)
    {
        string cwd = Path.GetDirectoryName(modelFile);

        // make primitives
        PrimitiveType[] ptypes = {
            PrimitiveType.Plane,
            PrimitiveType.Sphere,
            PrimitiveType.Cylinder,
            PrimitiveType.Cube
        };
        GameObject[] primitives = new GameObject[4];
        for( int i=0; i<4; i++)
            primitives[i] = GameObject.CreatePrimitive(ptypes[i]);

        // allocate array
        objects = new GameObject[nobject];

        // process objects
        for( int i=0; i<nobject; i++ )
        {
            // get object name
            StringBuilder name = new StringBuilder(100);
            MJP.GetObjectName(i, name, 100);
            
            // create new GameObject, place under root
            objects[i] = new GameObject(name.ToString());
            MeshFilter filt = objects[i].AddComponent<MeshFilter>();
            MeshRenderer rend = objects[i].AddComponent<MeshRenderer>();
            InstancedColor colors = objects[i].AddComponent<InstancedColor>();

            objects[i].transform.parent = root.transform;
            // get MuJoCo object descriptor
            MJP.TObject obj;
            MJP.GetObject(i, &obj);

            // For Furniture Assembly Environment: do not visualize site
            if (obj.category == (int)MJP.TCategory.SITE && !objects[i].name.Contains("conn")) {
                objects[i].layer = 9;
            }
            if (objects[i].name.StartsWith("noviz", StringComparison.Ordinal)) {
                objects[i].layer = 10;
            }
            if (objects[i].name.StartsWith("floor", StringComparison.Ordinal)) {
                objects[i].layer = 10;
            }

            // set mesh
            switch ( (MJP.TGeom)obj.geomtype )
            {
                case MJP.TGeom.PLANE:
                    filt.sharedMesh = primitives[0].GetComponent<MeshFilter>().sharedMesh;
                    break;

                case MJP.TGeom.SPHERE:
                    filt.sharedMesh = primitives[1].GetComponent<MeshFilter>().sharedMesh;
                    break;

                case MJP.TGeom.CYLINDER:
                    filt.sharedMesh = primitives[2].GetComponent<MeshFilter>().sharedMesh;
                    break;

                case MJP.TGeom.BOX:
                    filt.sharedMesh = primitives[3].GetComponent<MeshFilter>().sharedMesh;
                    break;

                case MJP.TGeom.HFIELD:
                    int nrow = obj.hfield_nrow;
                    int ncol = obj.hfield_ncol;
                    int r, c;

                    // allocate
                    Vector3[] hfvertices = new Vector3[nrow*ncol + 4*nrow+4*ncol];
                    Vector2[] hfuv = new Vector2[nrow*ncol + 4*nrow+4*ncol];
                    int[] hffaces0 = new int[3*2*(nrow-1)*(ncol-1)];
                    int[] hffaces1 = new int[3*(4*(nrow-1)+4*(ncol-1))];

                    // vertices and uv: surface
                    for( r=0; r<nrow; r++ )
                        for( c=0; c<ncol; c++ )
                        {
                            int k = r*ncol+c;
                            float wc = c / (float)(ncol-1);
                            float wr = r / (float)(nrow-1);

                            hfvertices[k].Set(-(wc-0.5f), obj.hfield_data[k], -(wr-0.5f));
                            hfuv[k].Set(wc, wr);
                        }

                    // vertices and uv: front and back
                    for( r=0; r<nrow; r+=(nrow-1) )
                        for( c=0; c<ncol; c++ )
                        {
                            int k = nrow*ncol + 2*((r>0?ncol:0)+c);
                            float wc = c / (float)(ncol-1);
                            float wr = r / (float)(nrow-1);

                            hfvertices[k].Set(-(wc-0.5f), -0.5f, -(wr-0.5f));
                            hfuv[k].Set(wc, 0);
                            hfvertices[k+1].Set(-(wc-0.5f), obj.hfield_data[r*ncol+c], -(wr-0.5f));
                            hfuv[k+1].Set(wc, 1);
                        }

                    // vertices and uv: left and right
                    for( c=0; c<ncol; c+=(ncol-1) )
                        for( r=0; r<nrow; r++ )
                        {
                            int k = nrow*ncol + 4*ncol + 2*((c>0?nrow:0)+r);
                            float wc = c / (float)(ncol-1);
                            float wr = r / (float)(nrow-1);

                            hfvertices[k].Set(-(wc-0.5f), -0.5f, -(wr-0.5f));
                            hfuv[k].Set(wr, 0);
                            hfvertices[k+1].Set(-(wc-0.5f), obj.hfield_data[r*ncol+c], -(wr-0.5f));
                            hfuv[k+1].Set(wr, 1);
                        }


                    // faces: surface
                    for( r=0; r<nrow-1; r++ )
                        for( c=0; c<ncol-1; c++ )
                        {
                            int f = r*(ncol-1)+c;
                            int k = r*ncol+c;

                            // first face in rectangle
                            hffaces0[3*2*f]   = k;
                            hffaces0[3*2*f+2] = k+1;
                            hffaces0[3*2*f+1] = k+ncol+1;

                            // second face in rectangle
                            hffaces0[3*2*f+3] = k;
                            hffaces0[3*2*f+5] = k+ncol+1;
                            hffaces0[3*2*f+4] = k+ncol;
                        }

                    // faces: front and back
                    for( r=0; r<2; r++ )
                        for( c=0; c<ncol-1; c++ )
                        {
                            int f = ((r>0?(ncol-1):0)+c);
                            int k = nrow*ncol + 2*((r>0?ncol:0)+c);

                            // first face in rectangle
                            hffaces1[3*2*f]   = k;
                            hffaces1[3*2*f+2] = k + (r>0 ? 1 : 3);
                            hffaces1[3*2*f+1] = k + (r>0 ? 3 : 1);

                            // second face in rectangle
                            hffaces1[3*2*f+3] = k;
                            hffaces1[3*2*f+5] = k + (r>0 ? 3 : 2);
                            hffaces1[3*2*f+4] = k + (r>0 ? 2 : 3);
                        }

                    // faces: left and right
                    for( c=0; c<2; c++ )
                        for( r=0; r<nrow-1; r++ )
                        {
                            int f = 2*(ncol-1) + ((c>0?(nrow-1):0)+r);
                            int k = nrow*ncol + 4*ncol + 2*((c>0?nrow:0)+r);

                            // first face in rectangle
                            hffaces1[3*2*f]   = k;
                            hffaces1[3*2*f+2] = k + (c>0 ? 3 : 1);
                            hffaces1[3*2*f+1] = k + (c>0 ? 1 : 3);

                            // second face in rectangle
                            hffaces1[3*2*f+3] = k;
                            hffaces1[3*2*f+5] = k + (c>0 ? 2 : 3);
                            hffaces1[3*2*f+4] = k + (c>0 ? 3 : 2);
                        }

                    Debug.Log(ncol);
                    Debug.Log(nrow);
                    Debug.Log(Mathf.Min(hffaces1));
                    Debug.Log(Mathf.Max(hffaces1));

                    // create mesh with automatic normals and tangents
                    filt.sharedMesh = new Mesh();
                    filt.sharedMesh.vertices = hfvertices;
                    filt.sharedMesh.uv = hfuv;
                    filt.sharedMesh.subMeshCount = 2;
                    filt.sharedMesh.SetTriangles(hffaces0, 0);
                    filt.sharedMesh.SetTriangles(hffaces1, 1);
                    filt.sharedMesh.RecalculateNormals();
                    filt.sharedMesh.RecalculateTangents();

                    // set name
                    StringBuilder hname = new StringBuilder(100);
                    MJP.GetElementName(MJP.TElement.HFIELD, obj.dataid, hname, 100);
                    filt.sharedMesh.name = hname.ToString();
                    break;

                case MJP.TGeom.CAPSULE:
                case MJP.TGeom.MESH:
                    // reuse shared mesh from earlier object
                    if( obj.mesh_shared>=0 )
                        filt.sharedMesh = objects[obj.mesh_shared].GetComponent<MeshFilter>().sharedMesh;

                    // create new mesh
                    else
                    {
                        string meshName;
                        // set name
                        if ((MJP.TGeom)obj.geomtype == MJP.TGeom.CAPSULE)
                            meshName = "Capsule mesh";
                        else
                        {
                            StringBuilder mname = new StringBuilder(100);
                            MJP.GetElementName(MJP.TElement.MESH, obj.dataid, mname, 100);
                           meshName = mname.ToString();
                        }

                        {
                            // copy vertices, normals, uv
                            Vector3[] vertices = new Vector3[obj.mesh_nvertex];
                            Vector3[] normals = new Vector3[obj.mesh_nvertex];
                            Vector2[] uv = new Vector2[obj.mesh_nvertex];
                            for (int k = 0; k < obj.mesh_nvertex; k++)
                            {
                                vertices[k].Set(-obj.mesh_position[3 * k],
                                                 obj.mesh_position[3 * k + 2],
                                                -obj.mesh_position[3 * k + 1]);

                                normals[k].Set(-obj.mesh_normal[3 * k],
                                                obj.mesh_normal[3 * k + 2],
                                               -obj.mesh_normal[3 * k + 1]);

                                uv[k].Set(obj.mesh_texcoord[2 * k],
                                          obj.mesh_texcoord[2 * k + 1]);
                            }

                            // copy faces
                            int[] faces = new int[3 * obj.mesh_nface];
                            for (int k = 0; k < obj.mesh_nface; k++)
                            {
                                faces[3 * k] = obj.mesh_face[3 * k];
                                faces[3 * k + 1] = obj.mesh_face[3 * k + 2];
                                faces[3 * k + 2] = obj.mesh_face[3 * k + 1];
                            }

                            // number of verices can be modified by uncompressed mesh
                            int nvert = obj.mesh_nvertex;

                            
                            // replace with uncompressed mesh when UV needs to be recomputed
                           // ( recomputeUV && (MJP.TGeom)obj.geomtype==MJP.TGeom.MESH )
                            {
                                // make temporary mesh
                                Mesh temp = new Mesh();
                                temp.vertices = vertices;
                                temp.normals = normals;
                                temp.triangles = faces;

                                // generate uncompressed UV unwrapping
                                /* Vector2[] UV = Unwrapping.GeneratePerTriangleUV(temp);
                                 int N = UV.GetLength(0)/3;
                                 if( N!=obj.mesh_nface )
                                     throw new System.Exception("Unexpected number of faces");
                                 nvert = 3*N;*/
                                int N = obj.mesh_nface;
                                nvert = 3 * N;
                                // create corresponding uncompressed vertices, normals, faces
                                Vector3[] Vertex = new Vector3[3*N];
                                Vector3[] Normal = new Vector3[3*N];
                                int[] Face = new int[3*N];                            
                                for( int k=0; k<N; k++ )
                                {
                                    Vertex[3*k]   = vertices[faces[3*k]];
                                    Vertex[3*k+1] = vertices[faces[3*k+1]];
                                    Vertex[3*k+2] = vertices[faces[3*k+2]];

                                    Normal[3*k]   = normals[faces[3*k]];
                                    Normal[3*k+1] = normals[faces[3*k+1]];
                                    Normal[3*k+2] = normals[faces[3*k+2]];

                                    Face[3*k]   = 3*k;
                                    Face[3*k+1] = 3*k+1;
                                    Face[3*k+2] = 3*k+2;
                                }

                                // create uncompressed mesh
                                filt.sharedMesh = new Mesh();
                                filt.sharedMesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
                                filt.sharedMesh.vertices = Vertex;
                                filt.sharedMesh.normals = Normal;
                                filt.sharedMesh.triangles = Face;
                                filt.sharedMesh.name = meshName;
                                
                                // filt.sharedMesh.uv = UV;
                            }

                            // otherwise create mesh directly
                            /*   else
                               {
                                   filt.sharedMesh = new Mesh();
                                   filt.sharedMesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
                                   filt.sharedMesh.vertices = vertices;
                                   filt.sharedMesh.normals = normals;
                                   filt.sharedMesh.triangles = faces;
                                   filt.sharedMesh.uv = uv;
                                   filt.sharedMesh.name = meshName;
                                   filt.sharedMesh.RecalculateNormals(30);
                               }*/

                            // optionally recompute normals for meshes
                            /*   if (recomputeNormal && (MJP.TGeom)obj.geomtype == MJP.TGeom.MESH)
                                   filt.sharedMesh.RecalculateNormals(60);*/

                            filt.sharedMesh.RecalculateNormals(25);
                            // always calculate tangents (MuJoCo does not support tangents)
                            filt.sharedMesh.RecalculateTangents();     
                        }

                        // print error if number of vertices or faces is over 65535
                       /* if( obj.mesh_nface>65535 || nvert>65535 )
                            Debug.LogError("MESH TOO BIG: " + filt.sharedMesh.name + 
                                           ", vertices " + nvert + ", faces " + obj.mesh_nface);*/
                    }
                    break;
            }

            //TODO: Set segmentation color with Material.SetColor("_SegColor")
            // existing material

            byte segmentation = 0;
            // Try to get segmentation id. If nothing is found, the object will remain background (0)
            if( meshSegmentMap.TryGetValue(name.ToString(), out segmentation) == false)
            {
                meshSegmentMap.TryGetValue(filt.sharedMesh.name, out segmentation);
            }
            Material base_material = null;

            if( materials.TryGetValue(obj.material, out base_material) )
            {
                rend.sharedMaterial = base_material;
                colors.Diffuse = new Color(obj.color[0], obj.color[1], obj.color[2], obj.color[3]);
                colors.Segmentation = new Color32(segmentation, segmentation, segmentation, segmentation);
            }
            else // Missing material (shouldn't be possible?)
            {
               Debug.Log("Couldn't find a Material for id:" + obj.material);
            }


            // get MuJoCo object transform and set in Unity
            MJP.TTransform transform;
            int visible;
            int selected;
            MJP.GetObjectState(i, &transform, &visible, &selected);
            SetTransform(objects[i], transform);
        }

        // delete primitives
        for( int i=0; i<4; i++ )
            Destroy(primitives[i]);
    }


    public unsafe void Import(string _file)
    {
        modelFile = _file;
        RunImport();
    }

    // run importer
    private unsafe void RunImport()
    {
        Resources.UnloadUnusedAssets();

        BuildAssetDatabase(modelFile);
            

        // adjust global settings
        Time.fixedDeltaTime = 0.005f;
        //PlayerSettings.runInBackground = true;
        if( enableRemote )
        {
            QualitySettings.vSyncCount = (noVSync ? 0 : 1);
        }
        else
            QualitySettings.vSyncCount = 1;

        // disable active cameras
       /* Camera[] activecam = FindObjectsOfType<Camera>();
        foreach( Camera ac in activecam )
            ac.gameObject.SetActive(false);
            */

        fileName = Path.GetFileName(modelFile);

        // initialize plugin and load model
        MJP.Close();
        MJP.Initialize();
        MJP.LoadModel(modelFile);

        // get model sizes
        MJP.TSize size;
        MJP.GetSize(&size);

        // import materials
        //if( size.nmaterial>0 )
        { 
        //    MakeDirectory("Assets", "Materials");
            ImportMaterials(size.nmaterial);
        }

        // create root, destroy old if present
        root = GameObject.Find("MuJoCo");
        if( root!=null )
        {
            Destroy(root);
        }
            
        root = new GameObject("MuJoCo");
        if( root==null )
            throw new System.Exception("Could not create root MuJoCo object");

        root.transform.localPosition = transform.localPosition;
        root.transform.localRotation = transform.localRotation;
        root.transform.localScale = transform.localScale;

        // add camera to root
        AddCamera();

        // import renderable objects under root
        ImportObjects(size.nobject);

        // ImportLights(size.nlight);

        // attach script to root
        if( enableRemote )
        {
            // add remote
            MJRemote extsim = root.GetComponent<MJRemote>();
            if(extsim == null )
                extsim = root.AddComponent<MJRemote>();

            extsim.root = root;
            extsim.modelFile = fileName + ".mjb";

            MJTCPInterface tcpif = this.gameObject.GetComponent<MJTCPInterface>();
            if (tcpif == null)
                tcpif = this.gameObject.AddComponent<MJTCPInterface>();

            tcpif.root = root;
            tcpif.tcpAddress = tcpAddress;
            tcpif.tcpPort = tcpPort;

            // destroy simulate if present
            if( root.GetComponent<MJInternalSimulation>() )
                Destroy(root.GetComponent<MJInternalSimulation>());
        }
        else
        {
            // add simulate
            MJInternalSimulation sim = root.GetComponent<MJInternalSimulation>();
            if( sim==null )
                sim = root.AddComponent<MJInternalSimulation>();
            sim.root = root;
            sim.modelFile = fileName + ".mjb";

            // destroy remote if present
            if( root.GetComponent<MJRemote>() )
                Destroy(root.GetComponent<MJRemote>());
            // destroy remote if present
            if (this.GetComponent<MJTCPInterface>())
                Destroy(root.GetComponent<MJTCPInterface>());
        }

        // close plugin
      //  MJP.Close();
    }

    private unsafe void ImportLights(int nlight)
    {
        // allocate array
        objects = new GameObject[nlight];

        // process objects
        for (int i = 0; i < nlight; i++)
        {
            // get object name
            StringBuilder name = new StringBuilder(100);
            MJP.GetElementName(MJP.TElement.LIGHT, i, name, 100);
            if (name.Length == 0) { name.Append("Light "); name.Append(i); }
            // create new GameObject, place under root
            objects[i] = new GameObject(name.ToString());
            Light lightComp = objects[i].AddComponent<Light>();

            objects[i].transform.parent = root.transform;
            // get MuJoCo object descriptor
            MJP.TLight obj;
            MJP.GetLight(i, &obj);
            //GetLightState
            // set mesh

            lightComp.type = obj.directional != 0 ? LightType.Directional : LightType.Point;
            lightComp.shadows = obj.castshadow == 0 ? LightShadows.None : LightShadows.Soft;
            lightComp.color = new Color(obj.diffuse[0], obj.diffuse[1], obj.diffuse[2]);
            lightComp.range = (float)Math.Sqrt(255 / obj.attenuation[2]);
            float[] pos = { 0, 0, 0 };
            float[] dir = { 0, 0, 0 };
            fixed (float* ppos = pos)
            fixed (float* pdir = dir)

            MJP.GetLightState(i, 0, ppos, pdir);

            objects[i].transform.localPosition = new Vector3(-pos[0], pos[2], -pos[1]);
            objects[i].transform.rotation = Quaternion.LookRotation(new Vector3(-dir[0], dir[2], -dir[1]), Vector3.up); 
        }
    }

    void OnApplicationQuit()
    {
        MJP.Close();
    }
}
