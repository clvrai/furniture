using UnityEngine;
using System;
using System.IO;
using System.Collections.Generic;
using System.Xml;


public class FurnitureInfo
{
    public string name;
    public int numPart;
    public string version;
    public string path;
    public int taskID;
    public List<PartInfo> partList;
    public List<ManualInfo> manualList;
    public float scale;
}

public class PartInfo
{
    public string name;
    public string path;
    public string shapeDesc;
    public Vector3 scale;
    public Vector3 position;
    public Vector3 rotation;
    public List<JointInfo> joints;
}

public class JointInfo
{
    public string group;
    public string opponent;
    public string path;
    public Vector3 scale;
    public Vector3 position;
    public Vector3 rotation;
    public string type;
}

[Serializable]
public class ManualInfo
{
    public int m1;
    public string group1;
    public int m2;
    public string group2;
    public float scale1;
    public float scale2;
}

public sealed class EnvironmentInfo
{
    public List<string> furnitureNameList = null;
    public List<string> furniturePathList = null;
    public Color[] colorList = null;
    public string[] colorDescList = null;

    public static EnvironmentInfo Read()
    {
        // Read furniture model information
        string folderPath = Application.dataPath + "/Models/";
        string[] fileNames = Directory.GetFiles(folderPath, "*.xml");
        EnvironmentInfo environmentInfo = new EnvironmentInfo
        {
            furnitureNameList = new List<string>(),
            furniturePathList = new List<string>()
        };
        foreach (string fileName in fileNames) {
            string modelName = fileName.Replace(".xml", "").Replace(folderPath, "");
            environmentInfo.furnitureNameList.Add(modelName);
            environmentInfo.furniturePathList.Add(fileName);
        }

        environmentInfo.colorList = new Color[] {
            new Color (200, 140, 100),
            new Color (80, 66, 53),
            new Color (255, 216, 168),
            new Color (163, 218, 255),
            new Color (255, 201, 201),
            new Color (238, 190, 250),
            new Color (252, 194, 215),
            new Color (153, 233, 242),
            new Color (150, 242, 215),
            new Color (178, 242, 187),
            new Color (216, 245, 162),
            new Color (255, 236, 153),
        };

        environmentInfo.colorDescList = new String[] {
            "wood",
            "dark wood",
            "orange 2", //      new Color (255, 216, 168),
            "blue 2", //      new Color (163, 218, 255),
            "red 2", //      new Color(255, 201, 201),
            "grape 2", //      new Color (238, 190, 250),
            "pink 2", //     new Color(252, 194, 215),
            "cyan 2", //      new Color (153, 233, 242),
            "teal 2", //      new Color (150, 242, 215),
            "green 2", //      new Color (178, 242, 187),
            "lime 2", //      new Color (216, 245, 162),
            "yellow 2", //      new Color (255, 236, 153),
        };

        if (environmentInfo.colorDescList.Length != environmentInfo.colorList.Length) {
            Debug.LogError(String.Format(
                "colorDescList ({0}) and colorList ({1}) have different # of colors",
                environmentInfo.colorDescList.Length, environmentInfo.colorList.Length
            ));
        }
        return environmentInfo;
    }
}

public static class ModelSpec
{
    private static void WriteVector3(XmlElement element, string type, Vector3 data)
    {
        element.SetAttribute(type + "x", data.x.ToString("F8"));
        element.SetAttribute(type + "y", data.y.ToString("F8"));
        element.SetAttribute(type + "z", data.z.ToString("F8"));
    }

    public static void Write(FurnitureInfo furnitureInfo, string fileName, string folderName = "/Models/")
    {
        List<PartInfo> partList = furnitureInfo.partList;
        string filePath = Application.dataPath + folderName + fileName + ".xml";
        XmlDocument Document = new XmlDocument();
        XmlElement furnitureElement = Document.CreateElement("FurnitureInfo");
        furnitureElement.SetAttribute("version", "1");
        furnitureElement.SetAttribute("path", furnitureInfo.path);
        furnitureElement.SetAttribute("name", fileName);
        Document.AppendChild(furnitureElement);

        // Write unit info
        XmlElement partListElement = Document.CreateElement("UnitList");
        furnitureElement.AppendChild(partListElement);
        foreach (PartInfo unit in partList) {
            XmlElement unitElement = Document.CreateElement("Unit");
            unitElement.SetAttribute("name", unit.name);
            unitElement.SetAttribute("path", unit.path);
            unitElement.SetAttribute("shape", unit.shapeDesc);
            WriteVector3(unitElement, "s", unit.scale);
            WriteVector3(unitElement, "r", unit.rotation);
            WriteVector3(unitElement, "p", unit.position);

            foreach (JointInfo joint in unit.joints) {
                XmlElement jointElement = Document.CreateElement("Joint");
                jointElement.SetAttribute("group", joint.group);
                jointElement.SetAttribute("opponent", joint.opponent);
                jointElement.SetAttribute("path", joint.path);
                WriteVector3(jointElement, "s", joint.scale);
                WriteVector3(jointElement, "r", joint.rotation);
                WriteVector3(jointElement, "p", joint.position);
                jointElement.SetAttribute("type", joint.type);
                unitElement.AppendChild(jointElement);
            }
            partListElement.AppendChild(unitElement);
        }

        // Write manual info
        XmlElement manualListElement = Document.CreateElement("ManualList");
        furnitureElement.AppendChild(manualListElement);
        foreach (ManualInfo manual in furnitureInfo.manualList) {
            XmlElement manualElement = Document.CreateElement("Manual");
            manualElement.SetAttribute("m1", manual.m1.ToString());
            manualElement.SetAttribute("m2", manual.m2.ToString());
            manualElement.SetAttribute("group1", manual.group1);
            manualElement.SetAttribute("group2", manual.group2);
            manualListElement.AppendChild(manualElement);
        }

        Document.Save(filePath);
    }

    private static Vector3 ReadVector3(XmlElement element, string type)
    {
        float x = System.Convert.ToSingle(element.GetAttribute(type + "x"));
        float y = System.Convert.ToSingle(element.GetAttribute(type + "y"));
        float z = System.Convert.ToSingle(element.GetAttribute(type + "z"));
        return new Vector3(x, y, z);
    }

    public static FurnitureInfo Read(string fileName, string folderName = "/Models/")
    {
        FurnitureInfo furnitureInfo = new FurnitureInfo();
        List<PartInfo> unitList = new List<PartInfo>();
        List<ManualInfo> manualList = new List<ManualInfo>();

        string filePath = Application.dataPath + folderName + fileName + ".xml";
        XmlDocument document = new XmlDocument();
        document.Load(filePath);
        XmlElement furnitureElement = document["FurnitureInfo"];
        furnitureInfo.version = furnitureElement.GetAttribute("version");
        furnitureInfo.path = furnitureElement.GetAttribute("path");
        furnitureInfo.name = furnitureElement.GetAttribute("name");
        string scale = furnitureElement.GetAttribute("scale");
        furnitureInfo.scale = scale.Equals("") ? 1f : System.Convert.ToSingle(scale);

        XmlNode unitListElement = furnitureElement.ChildNodes[0];
        furnitureInfo.numPart = 0;
        foreach (XmlElement unitElement in unitListElement.ChildNodes) {
            PartInfo unit = new PartInfo();
            unit.name = unitElement.GetAttribute("name");
            unit.path = unitElement.GetAttribute("path");
            unit.shapeDesc = unitElement.GetAttribute("shape");
            unit.scale = ReadVector3(unitElement, "s");
            unit.scale *= furnitureInfo.scale;
            // TODO: rotation and position are used for checking correct assembly result.
            unit.rotation = ReadVector3(unitElement, "r");
            unit.position = ReadVector3(unitElement, "p");
            unit.position *= furnitureInfo.scale;

            unit.joints = new List<JointInfo>();
            foreach (XmlElement jointElement in unitElement.ChildNodes) {
                JointInfo joint = new JointInfo();
                joint.group = jointElement.GetAttribute("group");
                joint.opponent = jointElement.GetAttribute("opponent");
                joint.path = jointElement.GetAttribute("path");
                joint.scale = ReadVector3(jointElement, "s");
                if (furnitureInfo.version != "1") {
                    joint.scale.x = joint.scale.x * 3f;
                    joint.scale.z = joint.scale.z * 3f;
                }
                joint.rotation = ReadVector3(jointElement, "r");
                joint.position = ReadVector3(jointElement, "p");
                joint.type = jointElement.GetAttribute("type");
                unit.joints.Add(joint);
            }
            unitList.Add(unit);
            furnitureInfo.numPart++;
        }
        furnitureInfo.partList = unitList;

        // Write manual info
        XmlNode manualListElement = furnitureElement.ChildNodes[1];
        foreach (XmlElement manualElement in manualListElement.ChildNodes) {
            ManualInfo manual = new ManualInfo {
                m1 = System.Convert.ToInt32(manualElement.GetAttribute("m1")),
                m2 = System.Convert.ToInt32(manualElement.GetAttribute("m2")),
                group1 = manualElement.GetAttribute("group1"),
                group2 = manualElement.GetAttribute("group2")
            };
            string scale1 = manualElement.GetAttribute("scale1");
            string scale2 = manualElement.GetAttribute("scale2");
            manual.scale1 = scale1.Equals("") ? 1f : System.Convert.ToSingle(scale1);
            manual.scale2 = scale2.Equals("") ? 1f : System.Convert.ToSingle(scale2);
            manualList.Add(manual);
        }
        furnitureInfo.manualList = manualList;

        return furnitureInfo;
    }
}
