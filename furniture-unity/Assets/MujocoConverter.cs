using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Parabox.Stl;
using UnityEditor;
using System.IO;
using System.Xml;

public class MujocoConverter : MonoBehaviour {

    private EnvironmentInfo environmentInfo;
    private Dictionary<string, FurnitureInfo> furnitureList;
    private string rootFolder;
    private string folder;
    private string furnitureName;

    void Start() {
        LoadEnvironmentModels();

        int i = 0;
        foreach (var item in furnitureList) {
            FurnitureInfo info = item.Value;
            // Create directory for the furniture
            furnitureName = item.Key;
            furnitureName = "0189_TABLE_BILLSTA ROUND";
            rootFolder = Directory.GetCurrentDirectory() + "/MujocoOutput/";
            folder = rootFolder + furnitureName + "/";
            //GenerateSTL();
            CreateXML();
            i += 1;
            if (i == 1) {
                break;
            }
        }

        //EditorApplication.isPlaying = false;
    }
    private static string Vector3String(Vector3 data) {
        return data.x.ToString("F8") + " " + data.y.ToString("F8") + " " + data.z.ToString("F8");
    }

    private static string QuaternionString(Quaternion data) {
        // mujoco is w x y z, unity is x y z w
        return data.w.ToString("F8") + " " +  data.x.ToString("F8") + " " + data.y.ToString("F8") + " " + data.z.ToString("F8");
    }
    private static XmlElement GetElement(string xml) {
        XmlDocument doc = new XmlDocument();
        doc.LoadXml(xml);
        return doc.DocumentElement;
    }
    private void CreateXML() {
        FurnitureInfo furnitureInfo = furnitureList[furnitureName];
        GameObject original = Resources.Load<GameObject>(furnitureInfo.path);
        if (original == null) {
            print(furnitureInfo + " does not have prefab, skipping XML generation");
            return;
        }
        GameObject clone = Instantiate(original, new Vector3(100f, 0f, 0f), new Quaternion(1f, 0f, 0f, 0f));
        // get top level children (parts)
        List<GameObject> children = new List<GameObject>();
        foreach (Transform child in clone.transform) {
            GameObject c = child.gameObject;
            if (c.name.Contains("part")) {
                children.Add(c);
            }
        }

        // now STL files are created, generate Mujoco XML
        XmlDocument Document = new XmlDocument();
        XmlElement mujocoElement = Document.CreateElement("mujoco");
        mujocoElement.SetAttribute("model", furnitureName);


        XmlElement assetElement = Document.CreateElement("asset");
        XmlElement equalityElement = Document.CreateElement("equality");
        XmlElement worldbodyElement = Document.CreateElement("worldbody");

        // Generate assets tag, with mesh as children
        string[] meshPaths = Directory.GetFiles(folder, "*.stl");
        foreach (GameObject child in children) { // for each part
            string partName = child.name;
            string meshPath = furnitureName + "/" + partName + ".stl";
            XmlElement mesh = Document.CreateElement("mesh");
            mesh.SetAttribute("name", partName);
            mesh.SetAttribute("file", meshPath);
            Vector3 scale = GameObject.Find(partName).transform.localScale;
            Vector3 mujScale = new Vector3(scale[0], scale[2], scale[1]);
            mesh.SetAttribute("scale", Vector3String(scale));
            assetElement.AppendChild(mesh);
        } 

        XmlElement texture = Document.CreateElement("texture");
        texture.SetAttribute("file", "textures/light-wood.png");
        texture.SetAttribute("name", "tex-light-wood");
        texture.SetAttribute("type", "2d");

        XmlElement material = Document.CreateElement("material");
        material.SetAttribute("name", "light-wood");
        material.SetAttribute("reflectance", "0.5");
        material.SetAttribute("texrepeat", "20 20");
        material.SetAttribute("texture", "tex-light-wood");
        material.SetAttribute("texuniform", "true");

        assetElement.AppendChild(texture);
        assetElement.AppendChild(material);

        // Generate Equality tags


        // Worldbody tags


        foreach (GameObject c in children) {
            XmlElement body = Document.CreateElement("body");
            // set name, pos, quat of body
            body.SetAttribute("name", c.name);
            //Use opposite of setTransform to get mujoco pos, rot
            Vector3 scale = c.transform.localScale;
            Vector3 pos = c.transform.localPosition;
            pos.x *= scale.x;
            pos.y *= scale.y;
            pos.z *= scale.z;

            Vector3 mujPos = new Vector3(pos[0], -pos[2], pos[1]);
            body.SetAttribute("quat", "0 0.707 -0.707 0");
            body.SetAttribute("pos", Vector3String(mujPos));

            // Create geom, sites for body tag
            XmlElement geom = Document.CreateElement("geom");
            geom.SetAttribute("mass", "5");
            geom.SetAttribute("material", "light-wood");
            geom.SetAttribute("mesh", c.name);
            geom.SetAttribute("name", c.name);
            geom.SetAttribute("rgba", "0.82 0.71 0.55 1");
            geom.SetAttribute("type", "mesh");
            geom.SetAttribute("density", "100");
            geom.SetAttribute("solref", "0.001 1");

            // generate joint sites
            print("getting joints for " + c.name);

            foreach (Transform child in c.transform) {
                if (child.name.Contains("joint")) {
                    print(child.name);
                }
            }


            body.AppendChild(geom);
            worldbodyElement.AppendChild(body);
        }

        string filePath = rootFolder + furnitureName + ".xml";
        mujocoElement.AppendChild(assetElement);
        mujocoElement.AppendChild(equalityElement);
        mujocoElement.AppendChild(worldbodyElement);
        Document.AppendChild(mujocoElement);
        Document.Save(filePath);
        print(filePath);
        DestroyImmediate(clone);
    }

    /*
     Quaternion q = new Quaternion(0, 0, 0, 1);
        q.SetLookRotation(
            new Vector3(transform.yaxis[0], -transform.yaxis[2], transform.yaxis[1]),
            new Vector3(-transform.zaxis[0], transform.zaxis[2], -transform.zaxis[1])
        );

        obj.transform.localPosition = new Vector3(-transform.position[0], transform.position[2], -transform.position[1]);
        obj.transform.localRotation = q;
        obj.transform.localScale = new Vector3(transform.scale[0], transform.scale[2], transform.scale[1]);
    */
    private void LoadEnvironmentModels() {
        environmentInfo = EnvironmentInfo.Read();
        furnitureList = new Dictionary<string, FurnitureInfo>();
        HashSet<string> shapeSet = new HashSet<string>();
        int idx = 0;
        foreach (string furniture in environmentInfo.furnitureNameList) {
            furnitureList[furniture] = ModelSpec.Read(furniture);
            furnitureList[furniture].taskID = idx;
            foreach (var part in furnitureList[furniture].partList)
                shapeSet.Add(part.shapeDesc);
            idx++;
        }
    }

    private void GenerateSTL() {
       
        FurnitureInfo furnitureInfo = furnitureList[furnitureName];
        GameObject original = Resources.Load<GameObject>(furnitureInfo.path);
        if (original == null) {
            print(furnitureInfo + " does not have prefab, skipping STL conversion");
            return;
        }
        if (!Directory.Exists(folder)) {
            Directory.CreateDirectory(folder);
        }
        GameObject clone = Instantiate(original, new Vector3(100f, 0f, 0f), new Quaternion(1f, 0f, 0f, 0f));
        // get top level children (parts)
        List<GameObject> children = new List<GameObject>();
        foreach (Transform child in clone.transform) {
            GameObject c = child.gameObject;
            if (c.name.Contains("part"))
                children.Add(c);
        }
        // create stl files for each part
        foreach (GameObject child in children) {
            MeshFilter[] mfs = child.GetComponentsInChildren<MeshFilter>();
            List<GameObject> partObjects = new List<GameObject>();
            foreach (MeshFilter m in mfs) {
                if (!m.name.Contains("joint"))
                    partObjects.Add(m.gameObject);
            }
            // create unified STL file for all meshes in the part
            GameObject[] partChildren = partObjects.ToArray();
            string path = folder + child.name + ".stl";
            if (partChildren.Length > 0)
                Exporter.Export(path, partChildren, FileType.Binary);
        }
        print(furnitureName + " generated STL.");
        //DestroyImmediate(clone);
    }
}