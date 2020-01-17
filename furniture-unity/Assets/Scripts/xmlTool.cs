//---------------------------------//
//  This file is part of MuJoCo    //
//  Written by Emo Todorov         //
//  Copyright (C) 2018 Roboti LLC  //
//---------------------------------//

#if UNITY_EDITOR

using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;
using UnityEditor;
using System.Xml;
using System.Xml.Schema;
using System.IO;


public class xmlTool : EditorWindow
{
    // script options
    public string modelFile = "<enter file or browse>";
    public string fileName = "";
    public bool recomputeUV = false;
    public bool recomputeNormal = false;
    public bool importTexture = true;
    public bool enableRemote = false;
    public string tcpAddress = "127.0.0.1";
    public int tcpPort = 1050;
    public bool noVSync = false;
    public Dictionary<string, int> partCount = new Dictionary<string, int>();

    // Unity element arrays
    Texture2D[] textures;
    Material[] materials;
    public GameObject[] objects;
    GameObject root = null;

    // Xml things
    public XmlDocument doc = new XmlDocument();
    public string site_size = "";
    public List<string> bodyNames = new List<string>();
    public List<GameObject> connSiteObjs = new List<GameObject>();
    public int selBody1 = 0;
    public int selBody2 = 0;
    public string geomNameToDuplicate = "";
    public int dupGeomBodyIndex = 0; // body tag for duplicate
    public string groupName1 = "";
    public string groupName2 = "";
    public string angles = "";
    public List<string> geomTypes = new List<string>(){"cube", "cylinder"};
    public int geomTypeIndex = 0;
    public int colGeomBodyIndex;
    public List<GameObject> colGeoms = new List<GameObject>();
    public Dictionary<GameObject, string> geomType = new Dictionary<GameObject, string>();
    public Dictionary<string, int> colGeomCount = new Dictionary<string, int>();
    // create menu item
    [MenuItem("Window/xmlTool")]
    public static void ShowWindow()
    {
        // show existing window instance, or make one
        EditorWindow.GetWindow(typeof(xmlTool), false, "xmlTool");
    }

    // present GUI, get options, run importer
    void OnGUI()
    {
        // get file name
        EditorGUILayout.Space();
        GUILayout.BeginHorizontal();
        if (GUILayout.Button("Browse ...")) {
            // get directory of current model, otherwise use Assets
            string dir = "Assets";
            int lastsep = modelFile.LastIndexOf('/');
            if (lastsep >= 0)
                dir = modelFile.Substring(0, lastsep);

            // file open dialog
            string temp = EditorUtility.OpenFilePanel("Select file", dir, "xml,urdf,mjb");
            if (temp.Length != 0) {
                modelFile = temp;
                this.Repaint();
            }
        }

        modelFile = EditorGUILayout.TextField(modelFile);
        GUILayout.EndHorizontal();

        // options
        EditorGUILayout.Space();
        recomputeNormal = EditorGUILayout.Toggle("Recompute Normals", recomputeNormal);
        recomputeUV = EditorGUILayout.Toggle("Recompute UVs", recomputeUV);
        importTexture = EditorGUILayout.Toggle("Import Textures", importTexture);
        enableRemote = EditorGUILayout.Toggle("Enable Remote", enableRemote);
        using (new EditorGUI.DisabledScope(enableRemote == false)) {
            tcpAddress = EditorGUILayout.TextField("TCP Address", tcpAddress);
            tcpPort = EditorGUILayout.IntField("TCP Port", tcpPort);
            noVSync = EditorGUILayout.Toggle("Disable V-Sync", noVSync);
        }

        // run importer or clear
        EditorGUILayout.Space();
        if (GUILayout.Button("Import Model", GUILayout.Height(25), GUILayout.Width(200)))
            RunImport();
        GUILayout.BeginHorizontal();
        selBody1 = EditorGUILayout.Popup("Body1 (top)", selBody1, bodyNames.ToArray());
        selBody2 = EditorGUILayout.Popup("Body2 (bot)", selBody2, bodyNames.ToArray());
        GUILayout.EndHorizontal();
        GUILayout.BeginHorizontal();
        groupName1 = EditorGUILayout.TextField("groupName1", groupName1);
        groupName2 = EditorGUILayout.TextField("groupName2", groupName2);
        GUILayout.EndHorizontal();
        angles = EditorGUILayout.TextField("Connection Angles", angles);
        if (GUILayout.Button("Add Connection Site", GUILayout.Height(25), GUILayout.Width(200)))
            AddSite();

        GUILayout.BeginHorizontal();
        colGeomBodyIndex = EditorGUILayout.Popup("collider geom body", colGeomBodyIndex, bodyNames.ToArray());
        geomTypeIndex = EditorGUILayout.Popup("Geom Type", geomTypeIndex, geomTypes.ToArray());
        GUILayout.EndHorizontal();

        if( GUILayout.Button("Add Collider Geom", GUILayout.Height(25), GUILayout.Width(200)) )
            AddColGeom();
        // duplication gui
        EditorGUILayout.Space();
        GUILayout.BeginHorizontal();
        geomNameToDuplicate = EditorGUILayout.TextField("Name of geom to duplicate:", geomNameToDuplicate);
        dupGeomBodyIndex = EditorGUILayout.Popup("Duplicate to Body:", dupGeomBodyIndex, bodyNames.ToArray());
        GUILayout.EndHorizontal();
        EditorGUILayout.Space();

        if( GUILayout.Button("Duplicate Selected Collider Geom", GUILayout.Height(25), GUILayout.Width(200)) )
            DuplicateSelectedColGeom();
        //if (GUILayout.Button("Test Quat Conversion", GUILayout.Height(25)))
        //    TestQuat();
        if (GUILayout.Button("Save Model", GUILayout.Height(25), GUILayout.Width(100)))
            SaveModel();
    }

    private void TestQuat()
    {
        GameObject temp = GameObject.Find("Debug");
        if (temp == null) temp = new GameObject("Debug");

        //temp.transform.rotation = Random.rotation;
        temp.transform.rotation = new Quaternion(-0.7f, 0f, 0f, 0.7f);

        Debug.Log(GameObject.Find("frontlegs-lsidesupport,0,180,conn_site").transform.rotation.ToString("F4"));

        Quaternion qNew = TransformQuaternion(temp.transform.rotation);
        Debug.Log("New " + qNew.ToString("F4"));

        Quaternion q = new Quaternion(0, 0, 0, 1);
        Vector3 zaxis = temp.transform.up;
        Vector3 yaxis = temp.transform.forward;

        q.SetLookRotation(
            new Vector3(yaxis[0], -yaxis[2], yaxis[1]),
            new Vector3(-zaxis[0], zaxis[2], -zaxis[1])
        );

        temp.transform.rotation = q;

        Vector3 newz = new Vector3(-temp.transform.up[0], -temp.transform.up[2], temp.transform.up[1]);
        Vector3 newy = new Vector3(temp.transform.forward[0], temp.transform.forward[2], -temp.transform.forward[1]);

        q.SetLookRotation(newy, newz);
    }

    private Quaternion TransformQuaternion(Quaternion q)
    {
        GameObject temp = GameObject.Find("Debug");
        if (temp == null) temp = new GameObject("Debug");
        temp.transform.rotation = q;
        Vector3 newz = new Vector3(-temp.transform.up[0], -temp.transform.up[2], temp.transform.up[1]);
        Vector3 newy = new Vector3(temp.transform.forward[0], temp.transform.forward[2], -temp.transform.forward[1]);
        return Quaternion.LookRotation(newz, newy);
    }

    private void DuplicateSelectedColGeom() {
        GameObject geomToDuplicate = GameObject.Find(geomNameToDuplicate);
        GameObject dup = Object.Instantiate(geomToDuplicate);
        dup.name = "noviz_collision_" + bodyNames[dupGeomBodyIndex] + "_";
        colGeoms.Add(dup);
        geomType.Add(dup, geomType[geomToDuplicate]);
    }

    private unsafe void AddColGeom(){
        GameObject parentbody, cornersite, colGeom;
        float x_scale, y_scale, z_scale, x, y, z;
        colGeom = null;
        //GameObject.gameObject.tag="cube";
        //GameObject.gameObject.tag="cylinder";

        if(geomTypes[geomTypeIndex] == "cube"){
            colGeom = GameObject.CreatePrimitive(PrimitiveType.Cube);
            geomType.Add(colGeom, "cube");
        }
        else if(geomTypes[geomTypeIndex] == "cylinder"){
            colGeom = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            geomType.Add(colGeom, "cylinder");
        }
        if(colGeom != null){
            // access body
            parentbody = GameObject.Find("MuJoCo/" + bodyNames[colGeomBodyIndex] + "_mesh");
            // access corner of this body
            cornersite = GameObject.Find("MuJoCo/" + bodyNames[colGeomBodyIndex] + "_corner_site1");
            // set default scales, pos
            if(parentbody != null){
                x = parentbody.transform.localPosition.x;
                y = parentbody.transform.localPosition.y;
                z = parentbody.transform.localPosition.z;
                x_scale = 1;
                y_scale = 1;
                z_scale = 1;
                if(cornersite != null){
                    x_scale = 2*System.Math.Abs(x - cornersite.transform.localPosition.x);
                    y_scale = 2*System.Math.Abs(y - cornersite.transform.localPosition.y);
                    z_scale = 2*System.Math.Abs(z - cornersite.transform.localPosition.z);
                }
                colGeom.transform.localPosition = new Vector3(x, y, z);
                colGeom.transform.localScale = new Vector3(x_scale, y_scale, z_scale);
                //update name
                string parentName = "noviz_collision_" + bodyNames[colGeomBodyIndex];
                string geomName = parentName + "_";
                int count = 0;
                if (colGeomCount.TryGetValue(parentName, out count)) {
                    colGeomCount[parentName] += 1;
                } else {
                    colGeomCount[parentName] = 1;
                }
                geomName += count;
                Debug.Log("Geom Name: " + geomName);
                colGeom.name = geomName;
            }
            colGeoms.Add(colGeom);
        }
    }

    private unsafe void AddSite()
    {
        GameObject body1, body2, connSiteObj;
        float x, y, z, size;
        size = float.Parse(site_size);
        string bodyname1 = bodyNames[selBody1];
        string bodyname2 = bodyNames[selBody2];
        string curGroup1 = groupName1;
        string curGroup2 = groupName2;
        string curAngles = angles;
        string siteName = "";
        body1 = GameObject.Find("MuJoCo/" + bodyname1 + "_bottom_site");
        body2 = GameObject.Find("MuJoCo/" + bodyname2 + "_top_site");
        x = (body1.transform.localPosition.x + body2.transform.localPosition.x) / 2;
        y = (body1.transform.localPosition.y + body2.transform.localPosition.y) / 2;
        z = (body1.transform.localPosition.z + body2.transform.localPosition.z) / 2;
        if (curAngles == "") {
            siteName = bodyname1 + "-" + bodyname2 + "," + curGroup1 + "-"
            + curGroup2 + ",conn_site";
        } else {
            siteName = bodyname1 + "-" + bodyname2 + "," + curGroup1 + "-"
                + curGroup2 + "," + curAngles + ",conn_site";
        }
        // Debug.Log("here x=" + x + ", y=" + y + ", z=" + z);
        connSiteObj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        connSiteObj.transform.parent = root.transform;
        connSiteObj.transform.localPosition = new Vector3(x, y, z);
        connSiteObj.transform.localScale = new Vector3(size, size, size);
        connSiteObj.name = siteName;
        connSiteObjs.Add(connSiteObj);
        try {
            partCount[curGroup1 + "-" + curGroup2] += 1;
            partCount[curGroup2 + "-" + curGroup1] += 1;
        }
        catch (KeyNotFoundException) {
            partCount.Add(curGroup1 + "-" + curGroup2, 1);
            partCount.Add(curGroup2 + "-" + curGroup1, 1);
        }
        // make new site active selection in unity
        Selection.activeGameObject = connSiteObj;
    }
//  <geom density="50" euler="0 0 0" name="noviz_collision_1_table_leg1_1" pos="-0.015 0.17 0.0" rgba="1 0 0 1" size="0.012 0.012 0.16" solref="0.001 1" type="box" />

    private unsafe XmlElement createcolGeom(string name, string pos, string size, Quaternion quat, string type){
        XmlElement elem = doc.CreateElement("geom");
        XmlAttribute densityAttr = doc.CreateAttribute("density");
        densityAttr.Value = "50";
        XmlAttribute quatAttr = doc.CreateAttribute("euler");
        Quaternion mujocoQ = TransformQuaternion(quat);
        quatAttr.Value = mujocoQ.w + " " + mujocoQ.x + " " + mujocoQ.y + " " + mujocoQ.z;
        XmlAttribute nameAttr = doc.CreateAttribute("name");
        nameAttr.Value = name;
        XmlAttribute posAttr = doc.CreateAttribute("pos");
        posAttr.Value = pos;
        XmlAttribute rgbaAttr = doc.CreateAttribute("rgba");
        rgbaAttr.Value = "1 0 0 1";
        XmlAttribute sizeAttr = doc.CreateAttribute("size");
        sizeAttr.Value = size;
        XmlAttribute solrefAttr = doc.CreateAttribute("solref");
        solrefAttr.Value = "0.001 1";
        XmlAttribute typeAttr = doc.CreateAttribute("type");
        typeAttr.Value = type;
        elem.Attributes.Append(densityAttr);
        elem.Attributes.Append(quatAttr);
        elem.Attributes.Append(nameAttr);
        elem.Attributes.Append(posAttr);
        elem.Attributes.Append(rgbaAttr);
        elem.Attributes.Append(sizeAttr);
        elem.Attributes.Append(solrefAttr);
        elem.Attributes.Append(typeAttr);
        return elem;


    }

    private unsafe XmlElement createConnSite(string group1, string group2, string connAngles, string pos, Quaternion quat)
    {
        XmlElement elem = doc.CreateElement("site");
        XmlAttribute nameAttr = doc.CreateAttribute("name");
        nameAttr.Value = group1 + "-" + group2 + connAngles + ",conn_site" + partCount[group1 + "-" + group2].ToString();
        partCount[group1 + "-" + group2] -= 1;
        XmlAttribute posAttr = doc.CreateAttribute("pos");
        posAttr.Value = pos;
        XmlAttribute quatAttr = doc.CreateAttribute("quat");
        Quaternion mujocoQ = TransformQuaternion(quat);
        quatAttr.Value = mujocoQ.w + " " + mujocoQ.x + " " + mujocoQ.y + " " + mujocoQ.z;
        XmlAttribute rgbaAttr = doc.CreateAttribute("rgba");
        rgbaAttr.Value = "0 0 1 0.3";
        XmlAttribute sizeAttr = doc.CreateAttribute("size");
        sizeAttr.Value = site_size;
        elem.Attributes.Append(nameAttr);
        elem.Attributes.Append(posAttr);
        elem.Attributes.Append(quatAttr);
        elem.Attributes.Append(rgbaAttr);
        elem.Attributes.Append(sizeAttr);
        return elem;
    }

    private XmlNode getBody(string name)
    {
        string query = "//body[@name='" + name + "']";
        XmlNodeList nodes = doc.SelectNodes(query);
        return nodes.Item(0);
    }

    private unsafe string getMJSize(GameObject unityObj){
        string mj_relsize = "";
        float mj_size_x, mj_size_y, mj_size_z;
        mj_size_x = unityObj.transform.localPosition.x;
        mj_size_y = unityObj.transform.localPosition.z;
        mj_size_z = unityObj.transform.localPosition.y;
        // round to 5 decimal places
        mj_size_x = (float)System.Math.Round((double)mj_size_x, 5);
        mj_size_y = (float)System.Math.Round((double)mj_size_y, 5);
        mj_size_z = (float)System.Math.Round((double)mj_size_z, 5);
        mj_relsize = mj_size_x.ToString() + " " + mj_size_y.ToString() + " " + mj_size_z.ToString();
        return mj_relsize;
    }

    private unsafe string getMjRelCoords(GameObject unityObj, XmlNode parentBody){
        string mj_relpos = "";
        string [] parent_coords;
        float mj_abs_x, mj_abs_y, mj_abs_z, mj_rel_x, mj_rel_y, mj_rel_z;
        //convert unity pos to mj pos
        mj_abs_x = -unityObj.transform.localPosition.x;
        mj_abs_y = -unityObj.transform.localPosition.z;
        mj_abs_z = unityObj.transform.localPosition.y;
        parent_coords = parentBody.Attributes["pos"].Value.Split(' ');
        //subtract body pos from unity conn pos since mujoco is relative coords
        mj_rel_x = mj_abs_x - float.Parse(parent_coords[0]);
        mj_rel_y = mj_abs_y - float.Parse(parent_coords[1]);
        mj_rel_z = mj_abs_z - float.Parse(parent_coords[2]);
        // round to 5 decimal places
        mj_rel_x = (float)System.Math.Round((double)mj_rel_x, 5);
        mj_rel_y = (float)System.Math.Round((double)mj_rel_y, 5);
        mj_rel_z = (float)System.Math.Round((double)mj_rel_z, 5);
        mj_relpos = mj_rel_x.ToString() + " " + mj_rel_y.ToString() + " " + mj_rel_z.ToString();
        return mj_relpos;
    }

    // import renderable objects
    private unsafe void SaveModel()
    {
        string[] connsiteName, bodyNames, groupNames;
        string b1_sitepos, b2_sitepos, connAngles, parentBodyName;
        HashSet<string> validAngles = new HashSet<string>() { "0", "45", "90", "135", "180", "215", "270", "315" };
        // connsites
        foreach (GameObject connSiteObj in connSiteObjs) {
            if (connSiteObj != null) {
                connsiteName = connSiteObj.name.Split(',');
                bodyNames = connsiteName[0].Split('-');
                groupNames = connsiteName[1].Split('-');
                XmlNode body1 = getBody(bodyNames[0]);
                XmlNode body2 = getBody(bodyNames[1]);
                b1_sitepos = getMjRelCoords(connSiteObj, body1);
                b2_sitepos = getMjRelCoords(connSiteObj, body2);
                // get conn angles
                connAngles = "";
                foreach (string s in connsiteName) {
                    if (validAngles.Contains(s)) {
                        connAngles = connAngles + "," + s;
                    }
                }
                Quaternion b1_site_quat = connSiteObj.transform.rotation, b2_site_quat = connSiteObj.transform.rotation;
                XmlElement b1connSite = createConnSite(groupNames[0], groupNames[1], connAngles, b1_sitepos, b1_site_quat);
                XmlElement b2connSite = createConnSite(groupNames[1], groupNames[0], connAngles, b2_sitepos, b2_site_quat);
                body1.AppendChild(b1connSite);
                body2.AppendChild(b2connSite);
            }
        }
        // colGeom geoms
        string colGeomPos, colGeomSize;
        foreach (GameObject colGeom in colGeoms) {
            if (colGeom != null) {
                parentBodyName = colGeom.name.Substring(16);
                var words = parentBodyName.Split('_');
                parentBodyName = words[0] + '_' + words[1]; // remove collision id

                Debug.Log("colGeom parent body " + parentBodyName);
                Debug.Log("colGeom name: " + colGeom.name);
                XmlNode parentBody = getBody(parentBodyName);
                colGeomPos = getMjRelCoords(colGeom, parentBody);
                colGeomSize = getMJSize(colGeom);
                Quaternion colGeomQuat = colGeom.transform.rotation;
                XmlElement geom = createcolGeom(colGeom.name, colGeomPos, colGeomSize, colGeomQuat, geomType[colGeom]);
                parentBody.AppendChild(geom);
            }
        }

        char[] seperator = { '\\' };
        int count = 1;
        string[] subpath = modelFile.Split('/');
        string sourcedir = string.Join("/", subpath, 0, subpath.Length - 1);
        string savedir = string.Join("/", subpath, 0, subpath.Length - 3) + "/complete/" + fileName + "/";
        Directory.CreateDirectory(savedir);
        string curFileName = "";
        string destFile = "";
        if (System.IO.Directory.Exists(sourcedir)) {
            string[] files = System.IO.Directory.GetFiles(sourcedir);
            foreach (string s in files) {
                if (s.EndsWith(".stl")) {
                    curFileName = System.IO.Path.GetFileName(s);
                    destFile = System.IO.Path.Combine(savedir, curFileName);
                    System.IO.File.Copy(s, destFile, true);
                }
            }
        }
        Debug.Log("saving to" + savedir);
        doc.Save(savedir + fileName + ".xml");
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


    // make scene-speficic directories for materials and textures
    private void MakeDirectory(string parent, string directory)
    {
        // check subdirectories of parent
        string[] subdir = AssetDatabase.GetSubFolders(parent);
        bool found = false;
        string fullpath = parent + "/" + directory;
        foreach (string str in subdir)
            if (str == fullpath) {
                found = true;
                break;
            }

        // create if not found
        if (!found)
            AssetDatabase.CreateFolder(parent, directory);
    }


    // adjust material given object color
    private void AdjustMaterial(Material m, float r, float g, float b, float a)
    {
        // set main color,
        m.SetColor("_Color", new Color(r, g, b, a));

        // prepare for emission (used for highlights at runtime)
        m.EnableKeyword("_EMISSION");
        m.SetColor("_EmissionColor", new Color(0, 0, 0, 1));

        // set transparent mode (magic needed to convince Unity to do it)
        if (a < 1) {
            m.SetFloat("_Mode", 2);
            m.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            m.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            m.SetInt("_ZWrite", 0);
            m.DisableKeyword("_ALPHATEST_ON");
            m.EnableKeyword("_ALPHABLEND_ON");
            m.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            m.renderQueue = 3000;
        }
    }


    // Add camera
    private unsafe void AddCamera()
    {
        // Add camera under root
        GameObject camobj = new GameObject("camera");
        camobj.transform.parent = root.transform;
        Camera thecamera = camobj.AddComponent<Camera>();

        // set field of view, near, far
        MJP.TCamera cam;
        MJP.GetCamera(-1, &cam);
        thecamera.fieldOfView = cam.fov;
        thecamera.nearClipPlane = cam.znear;
        thecamera.farClipPlane = cam.zfar;

        // set transform
        MJP.TTransform transform;
        MJP.GetCameraState(-1, &transform);
        SetCamera(thecamera, transform);
    }


    // import textures
    private unsafe void ImportTextures(int ntexture)
    {
        // allocate array, find existing
        textures = new Texture2D[ntexture];
        Object[] alltextures = Resources.FindObjectsOfTypeAll(typeof(Texture2D));

        // process textures
        for (int i = 0; i < ntexture; i++) {
            // get texture name
            StringBuilder name = new StringBuilder(100);
            MJP.GetElementName(MJP.TElement.TEXTURE, i, name, 100);
            string texname = fileName + "_" + name.ToString();

            // get texture descriptor and save
            MJP.TTexture tex;
            MJP.GetTexture(i, &tex);

            // MuJoCo cube texture: use only top piece
            if (tex.cube > 0)
                tex.height = tex.width;

            // find existing texture
            foreach (Object texx in alltextures)
                if (texx.name == texname) {
                    textures[i] = (Texture2D)texx;

                    // resize if different
                    if (textures[i].width != tex.width || textures[i].height != tex.height)
                        textures[i].Resize(tex.width, tex.height);

                    break;
                }

            // not found: create new texture
            if (textures[i] == null)
                textures[i] = new Texture2D(tex.width, tex.height);

            // copy array
            Color32[] color = new Color32[tex.width * tex.height];
            for (int k = 0; k < tex.width * tex.height; k++) {
                color[k].r = tex.rgb[3 * k];
                color[k].g = tex.rgb[3 * k + 1];
                color[k].b = tex.rgb[3 * k + 2];
                color[k].a = 255;
            }

            // load data and apply
            textures[i].SetPixels32(color);
            textures[i].Apply();

            // create asset in database if not aleady there
            if (!AssetDatabase.Contains(textures[i]))
                AssetDatabase.CreateAsset(textures[i], "Assets/Textures/" + texname + ".asset");
        }

        AssetDatabase.Refresh();
    }


    // import materials
    private unsafe void ImportMaterials(int nmaterial)
    {
        // allocate array, find all existing
        materials = new Material[nmaterial];
        Object[] allmaterials = Resources.FindObjectsOfTypeAll(typeof(Material));

        // process materials
        for (int i = 0; i < nmaterial; i++) {
            // get material name
            StringBuilder name = new StringBuilder(100);
            MJP.GetElementName(MJP.TElement.MATERIAL, i, name, 100);
            string matname = fileName + "_" + name.ToString();

            // find existing material
            foreach (Object matt in allmaterials)
                if (matt != null && matt.name == matname) {
                    materials[i] = (Material)matt;
                    break;
                }

            // not found: create new material
            materials[i] = new Material(Shader.Find("Standard"));

            // get material descriptor and save
            MJP.TMaterial mat;
            MJP.GetMaterial(i, &mat);

            // set properties
            materials[i].name = matname;
            materials[i].EnableKeyword("_EMISSION");
            materials[i].SetColor("_Color", new Color(mat.color[0], mat.color[1], mat.color[2], mat.color[3]));
            materials[i].SetColor("_EmissionColor", new Color(mat.emission, mat.emission, mat.emission, 1));
            if (mat.color[3] < 1)
                materials[i].SetFloat("_Mode", 3.0f);

            // set texture if present
            if (mat.texture >= 0 && importTexture) {
                materials[i].mainTexture = textures[mat.texture];
                materials[i].mainTextureScale = new Vector2(mat.texrepeat[0], mat.texrepeat[1]);
            }

            // create asset in database if not aleady there
            if (!AssetDatabase.Contains(materials[i]))
                AssetDatabase.CreateAsset(materials[i], "Assets/Materials/" + matname + ".mat");
        }

        AssetDatabase.Refresh();
    }


    // import renderable objects
    private unsafe void ImportObjects(int nobject)
    {
        // make primitives
        PrimitiveType[] ptypes = {
            PrimitiveType.Plane,
            PrimitiveType.Sphere,
            PrimitiveType.Cylinder,
            PrimitiveType.Cube
        };
        GameObject[] primitives = new GameObject[4];
        for (int i = 0; i < 4; i++)
            primitives[i] = GameObject.CreatePrimitive(ptypes[i]);

        // allocate array
        objects = new GameObject[nobject];

        // process objects
        for (int i = 0; i < nobject; i++) {
            // get object name
            StringBuilder name = new StringBuilder(100);
            MJP.GetObjectName(i, name, 100);
            //            Debug.Log("Part " + i + " " + name);
            // create new GameObject, place under root
            objects[i] = new GameObject(name.ToString());
            objects[i].AddComponent<MeshFilter>();
            objects[i].AddComponent<MeshRenderer>();
            objects[i].transform.parent = root.transform;

            // get components
            MeshFilter filt = objects[i].GetComponent<MeshFilter>();
            MeshRenderer rend = objects[i].GetComponent<MeshRenderer>();

            // get MuJoCo object descriptor
            MJP.TObject obj;
            MJP.GetObject(i, &obj);

            // set mesh
            switch ((MJP.TGeom)obj.geomtype) {
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
                    Vector3[] hfvertices = new Vector3[nrow * ncol + 4 * nrow + 4 * ncol];
                    Vector2[] hfuv = new Vector2[nrow * ncol + 4 * nrow + 4 * ncol];
                    int[] hffaces0 = new int[3 * 2 * (nrow - 1) * (ncol - 1)];
                    int[] hffaces1 = new int[3 * (4 * (nrow - 1) + 4 * (ncol - 1))];

                    // vertices and uv: surface
                    for (r = 0; r < nrow; r++)
                        for (c = 0; c < ncol; c++) {
                            int k = r * ncol + c;
                            float wc = c / (float)(ncol - 1);
                            float wr = r / (float)(nrow - 1);

                            hfvertices[k].Set(-(wc - 0.5f), obj.hfield_data[k], -(wr - 0.5f));
                            hfuv[k].Set(wc, wr);
                        }

                    // vertices and uv: front and back
                    for (r = 0; r < nrow; r += (nrow - 1))
                        for (c = 0; c < ncol; c++) {
                            int k = nrow * ncol + 2 * ((r > 0 ? ncol : 0) + c);
                            float wc = c / (float)(ncol - 1);
                            float wr = r / (float)(nrow - 1);

                            hfvertices[k].Set(-(wc - 0.5f), -0.5f, -(wr - 0.5f));
                            hfuv[k].Set(wc, 0);
                            hfvertices[k + 1].Set(-(wc - 0.5f), obj.hfield_data[r * ncol + c], -(wr - 0.5f));
                            hfuv[k + 1].Set(wc, 1);
                        }

                    // vertices and uv: left and right
                    for (c = 0; c < ncol; c += (ncol - 1))
                        for (r = 0; r < nrow; r++) {
                            int k = nrow * ncol + 4 * ncol + 2 * ((c > 0 ? nrow : 0) + r);
                            float wc = c / (float)(ncol - 1);
                            float wr = r / (float)(nrow - 1);

                            hfvertices[k].Set(-(wc - 0.5f), -0.5f, -(wr - 0.5f));
                            hfuv[k].Set(wr, 0);
                            hfvertices[k + 1].Set(-(wc - 0.5f), obj.hfield_data[r * ncol + c], -(wr - 0.5f));
                            hfuv[k + 1].Set(wr, 1);
                        }


                    // faces: surface
                    for (r = 0; r < nrow - 1; r++)
                        for (c = 0; c < ncol - 1; c++) {
                            int f = r * (ncol - 1) + c;
                            int k = r * ncol + c;

                            // first face in rectangle
                            hffaces0[3 * 2 * f] = k;
                            hffaces0[3 * 2 * f + 2] = k + 1;
                            hffaces0[3 * 2 * f + 1] = k + ncol + 1;

                            // second face in rectangle
                            hffaces0[3 * 2 * f + 3] = k;
                            hffaces0[3 * 2 * f + 5] = k + ncol + 1;
                            hffaces0[3 * 2 * f + 4] = k + ncol;
                        }

                    // faces: front and back
                    for (r = 0; r < 2; r++)
                        for (c = 0; c < ncol - 1; c++) {
                            int f = ((r > 0 ? (ncol - 1) : 0) + c);
                            int k = nrow * ncol + 2 * ((r > 0 ? ncol : 0) + c);

                            // first face in rectangle
                            hffaces1[3 * 2 * f] = k;
                            hffaces1[3 * 2 * f + 2] = k + (r > 0 ? 1 : 3);
                            hffaces1[3 * 2 * f + 1] = k + (r > 0 ? 3 : 1);

                            // second face in rectangle
                            hffaces1[3 * 2 * f + 3] = k;
                            hffaces1[3 * 2 * f + 5] = k + (r > 0 ? 3 : 2);
                            hffaces1[3 * 2 * f + 4] = k + (r > 0 ? 2 : 3);
                        }

                    // faces: left and right
                    for (c = 0; c < 2; c++)
                        for (r = 0; r < nrow - 1; r++) {
                            int f = 2 * (ncol - 1) + ((c > 0 ? (nrow - 1) : 0) + r);
                            int k = nrow * ncol + 4 * ncol + 2 * ((c > 0 ? nrow : 0) + r);

                            // first face in rectangle
                            hffaces1[3 * 2 * f] = k;
                            hffaces1[3 * 2 * f + 2] = k + (c > 0 ? 3 : 1);
                            hffaces1[3 * 2 * f + 1] = k + (c > 0 ? 1 : 3);

                            // second face in rectangle
                            hffaces1[3 * 2 * f + 3] = k;
                            hffaces1[3 * 2 * f + 5] = k + (c > 0 ? 2 : 3);
                            hffaces1[3 * 2 * f + 4] = k + (c > 0 ? 3 : 2);
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
                    if (obj.mesh_shared >= 0)
                        filt.sharedMesh = objects[obj.mesh_shared].GetComponent<MeshFilter>().sharedMesh;

                    // create new mesh
                    else {
                        // copy vertices, normals, uv
                        Vector3[] vertices = new Vector3[obj.mesh_nvertex];
                        Vector3[] normals = new Vector3[obj.mesh_nvertex];
                        Vector2[] uv = new Vector2[obj.mesh_nvertex];
                        for (int k = 0; k < obj.mesh_nvertex; k++) {
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
                        for (int k = 0; k < obj.mesh_nface; k++) {
                            faces[3 * k] = obj.mesh_face[3 * k];
                            faces[3 * k + 1] = obj.mesh_face[3 * k + 2];
                            faces[3 * k + 2] = obj.mesh_face[3 * k + 1];
                        }

                        // number of verices can be modified by uncompressed mesh
                        int nvert = obj.mesh_nvertex;

                        // replace with uncompressed mesh when UV needs to be recomputed
                        if (recomputeUV && (MJP.TGeom)obj.geomtype == MJP.TGeom.MESH) {
                            // make temporary mesh
                            Mesh temp = new Mesh();
                            temp.vertices = vertices;
                            temp.normals = normals;
                            temp.triangles = faces;

                            // generate uncompressed UV unwrapping
                            Vector2[] UV = Unwrapping.GeneratePerTriangleUV(temp);
                            int N = UV.GetLength(0) / 3;
                            if (N != obj.mesh_nface)
                                throw new System.Exception("Unexpected number of faces");
                            nvert = 3 * N;

                            // create corresponding uncompressed vertices, normals, faces
                            Vector3[] Vertex = new Vector3[3 * N];
                            Vector3[] Normal = new Vector3[3 * N];
                            int[] Face = new int[3 * N];
                            for (int k = 0; k < N; k++) {
                                Vertex[3 * k] = vertices[faces[3 * k]];
                                Vertex[3 * k + 1] = vertices[faces[3 * k + 1]];
                                Vertex[3 * k + 2] = vertices[faces[3 * k + 2]];

                                Normal[3 * k] = normals[faces[3 * k]];
                                Normal[3 * k + 1] = normals[faces[3 * k + 1]];
                                Normal[3 * k + 2] = normals[faces[3 * k + 2]];

                                Face[3 * k] = 3 * k;
                                Face[3 * k + 1] = 3 * k + 1;
                                Face[3 * k + 2] = 3 * k + 2;
                            }

                            // create uncompressed mesh
                            filt.sharedMesh = new Mesh();
                            filt.sharedMesh.vertices = Vertex;
                            filt.sharedMesh.normals = Normal;
                            filt.sharedMesh.triangles = Face;
                            filt.sharedMesh.uv = UV;
                        }

                        // otherwise create mesh directly
                        else {
                            filt.sharedMesh = new Mesh();
                            filt.sharedMesh.vertices = vertices;
                            filt.sharedMesh.normals = normals;
                            filt.sharedMesh.triangles = faces;
                            filt.sharedMesh.uv = uv;
                        }

                        // optionally recompute normals for meshes
                        if (recomputeNormal && (MJP.TGeom)obj.geomtype == MJP.TGeom.MESH)
                            filt.sharedMesh.RecalculateNormals();

                        // always calculate tangents (MuJoCo does not support tangents)
                        filt.sharedMesh.RecalculateTangents();

                        // set name
                        if ((MJP.TGeom)obj.geomtype == MJP.TGeom.CAPSULE)
                            filt.sharedMesh.name = "Capsule mesh";
                        else {
                            StringBuilder mname = new StringBuilder(100);
                            MJP.GetElementName(MJP.TElement.MESH, obj.dataid, mname, 100);
                            filt.sharedMesh.name = mname.ToString();
                        }

                        // print error if number of vertices or faces is over 65535
                        if (obj.mesh_nface > 65535 || nvert > 65535)
                            Debug.LogError("MESH TOO BIG: " + filt.sharedMesh.name +
                                           ", vertices " + nvert + ", faces " + obj.mesh_nface);
                    }
                    break;
            }

            // existing material
            if (obj.material >= 0) {
                // not modified
                if (obj.color[0] == 0.5f && obj.color[1] == 0.5f && obj.color[2] == 0.5f && obj.color[3] == 1)
                    rend.sharedMaterial = materials[obj.material];

                // color override
                else {
                    rend.sharedMaterial = new Material(materials[obj.material]);
                    AdjustMaterial(rend.sharedMaterial, obj.color[0], obj.color[1], obj.color[2], obj.color[3]);
                }
            }

            // new material
            else {
                rend.sharedMaterial = new Material(Shader.Find("Standard"));
                AdjustMaterial(rend.sharedMaterial, obj.color[0], obj.color[1], obj.color[2], obj.color[3]);
            }

            // get MuJoCo object transform and set in Unity
            MJP.TTransform transform;
            int visible;
            int selected;
            MJP.GetObjectState(i, &transform, &visible, &selected);
            SetTransform(objects[i], transform);

        }

        // delete primitives
        for (int i = 0; i < 4; i++)
            DestroyImmediate(primitives[i]);

        AssetDatabase.Refresh();
    }


    // run importer
    private unsafe void RunImport()
    {
        // adjust global settings
        Time.fixedDeltaTime = 0.005f;
        PlayerSettings.runInBackground = true;
        if (enableRemote) {
            QualitySettings.vSyncCount = (noVSync ? 0 : 1);
        } else
            QualitySettings.vSyncCount = 1;

        // disable active cameras
        Camera[] activecam = FindObjectsOfType<Camera>();
        foreach (Camera ac in activecam)
            ac.gameObject.SetActive(false);

        // get filename only (not path or extension)
        int i1 = modelFile.LastIndexOf('/');
        int i2 = modelFile.LastIndexOf('.');
        if (i1 >= 0 && i2 > i1)
            fileName = modelFile.Substring(i1 + 1, i2 - i1 - 1);
        else
            throw new System.Exception("Unexpected model file format");

        // initialize plugin and load model
        MJP.Initialize();
        MJP.LoadModel(modelFile);

        bodyNames.Clear();
        connSiteObjs.Clear();
        // remove existing collision geoms
        colGeomCount.Clear();
        foreach(GameObject g in colGeoms) {
            Object.DestroyImmediate(g);
        }
        // initialize xml-related stuff
        doc.PreserveWhitespace = false;
        doc.Load(modelFile);
        site_size = doc.GetElementsByTagName("site")[0].Attributes["size"].Value;
        // site_size = "20";
        XmlNodeList bodies = doc.SelectNodes(".//body");
        foreach (XmlNode body in bodies) {
            bodyNames.Add(body.Attributes["name"].Value);
        }

        //        MJP.PrintModel(const mjModel* m, const char* filename);

        // get model sizes
        MJP.TSize size;
        MJP.GetSize(&size);

        // save binary model
        MakeDirectory("Assets", "StreamingAssets");
        MJP.SaveMJB("Assets/StreamingAssets/" + fileName + ".mjb");

        // import textures
        if (size.ntexture > 0 && importTexture) {
            MakeDirectory("Assets", "Textures");
            ImportTextures(size.ntexture);
        }

        // import materials
        if (size.nmaterial > 0) {
            MakeDirectory("Assets", "Materials");
            ImportMaterials(size.nmaterial);
        }

        // create root, destroy old if present
        root = GameObject.Find("MuJoCo");
        if (root != null)
            DestroyImmediate(root);
        root = new GameObject("MuJoCo");
        if (root == null)
            throw new System.Exception("Could not create root MuJoCo object");

        // Add camera to root
        AddCamera();

        // import renderable objects under root
        ImportObjects(size.nobject);

        /*     // attach script to root
             if( enableRemote )
             {
                 // Add remote
                 MJRemote rem = root.GetComponent<MJRemote>();
                 if( rem==null )
                     rem = root.AddComponent<MJRemote>();
                 rem.modelFile = fileName + ".mjb";
                 rem.tcpAddress = tcpAddress;
                 rem.tcpPort = tcpPort;

                 // destroy simulate if present
                 if( root.GetComponent<MJSimulate>() )
                     DestroyImmediate(root.GetComponent<MJSimulate>());
             }
             else
             {
                 // Add simulate
                 MJSimulate sim = root.GetComponent<MJSimulate>();
                 if( sim==null )
                     sim = root.AddComponent<MJSimulate>();
                 sim.modelFile = fileName + ".mjb";

                 // destroy remote if present
                 if( root.GetComponent<MJRemote>() )
                     DestroyImmediate(root.GetComponent<MJRemote>());
             } */

        // close plugin
        //MJP.Close();
    }
}

#endif
