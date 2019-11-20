using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Xml;

namespace DoorGym
{
    public class Configuration
    {
        private static Configuration instance = null;

        private SortedDictionary<string, byte> segmentIdMap;
        private SortedDictionary<string, RandomColor> material_random_parameters;
        public static void Initialize()
        {
            if (instance == null)
            {
                instance = new Configuration();
            }
        }

        // Use this for initialization
        protected Configuration()
        {
            segmentIdMap = new SortedDictionary<string, byte>();
            material_random_parameters = new SortedDictionary<string, RandomColor>();
            // YW: change the path of StreamingAssets
            string settings_path = Application.streamingAssetsPath + "/settings.xml";

            XmlDocument settings = new XmlDocument();

            settings.Load(settings_path);

            XmlNodeList segments = settings.DocumentElement.SelectNodes("segments/segment");
            XmlNodeList materials = settings.DocumentElement.SelectNodes("materials/material");

            foreach(XmlNode segment in segments)
            {
                XmlAttributeCollection attr = segment.Attributes;
                XmlAttribute nameNode = attr["name"];
                XmlAttribute idNode = attr["id"];
                if(nameNode != null && idNode != null)
                {
                    string name = nameNode.Value;
                    if (name.Length == 0)
                    {
                        throw new XmlException("Segment node must specify a name:\n " + segment.ToString());
                    }
                    else
                    {
                        byte id = System.Convert.ToByte(idNode.Value);
                        segmentIdMap.Add(name, id);
                    }
                }
                else
                {
                    throw new XmlException("Segment node name or id is missing or invalid:\n " + segment.ToString());
                }
            }

            foreach (XmlNode material in materials)
            {
                XmlAttributeCollection attr = material.Attributes;
                XmlAttribute nameNode = attr["name"];
                if (nameNode != null && nameNode.Value.Length > 0)
                {
                    material_random_parameters[nameNode.Value] = new RandomColor(material);            
                }
                else
                {
                    throw new XmlException("Material node has missing or invalid name:\n " + material.ToString());
                }
            }
        }

        public static SortedDictionary<string, byte> SegmentIDMap
        {
            get
            {
                return instance.segmentIdMap;
            }
        }

        public static SortedDictionary<string, RandomColor> MaterialParameters
        {
            get
            {
                return instance.material_random_parameters;
            }
        }

        public static Configuration Instance
        {
            get
            {
                return instance;
            }
        }
    }
}