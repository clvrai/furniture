/*
Copyright 2019 Panasonic Beta, a division of Panasonic Corporation of North America

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, 
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation 
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.
*/

using UnityEngine;
using System.Collections;
using System.Xml;


public class RandomColor
{
    public float hue_minimum;
    public float saturation_minimum;
    public float value_minimum;

    public float hue_maximum;
    public float saturation_maximum;
    public float value_maximum;

    public RandomColor()
    {
        hue_minimum = 0;
        saturation_minimum = 0;
        value_minimum = 0;

        hue_maximum = 1;
        saturation_maximum = 1;
        value_maximum = 1;
    }

    public RandomColor(float hl, float hh, float sl, float sh, float vl, float vh)
    {
        hue_minimum = hl;
        saturation_minimum = sl;
        value_minimum = vl;

        hue_maximum = hh;
        saturation_maximum = sh;
        value_maximum = vh;
    }


    private float Sanitize(XmlAttribute attr, float default_value, float min, float max)
    {
        float val = default_value;

        try
        {
            val = System.Convert.ToSingle(attr.Value);
        }
        catch (System.FormatException e)
        {
            Debug.LogError("Attribute " + attr.Value + " is not a floating point number");
            throw e;
        }
        catch (System.OverflowException e)
        {
            Debug.LogError("Attribute " + attr.Value + " cannot be represented as a 32-bit floating point number");
            throw e;
        }

        float clamped = Mathf.Clamp(val, min, max);

        if(clamped != val)
        {
            Debug.LogWarning("Attribute " + attr.Value + " was clamped to the range [" + min + " , " + max + "]");
        }

        return clamped;
    }

    public RandomColor(XmlNode node)
    {
        foreach (XmlNode param in node.ChildNodes)
        {
            XmlAttributeCollection chattr = param.Attributes;
            XmlAttribute min_attr = chattr["min"];
            XmlAttribute max_attr = chattr["max"];

            if (min_attr != null && max_attr != null)
            {
                switch (param.Name)
                {
                    case "hue":
                        hue_minimum = Sanitize(min_attr, 0, 0, 1);
                        hue_maximum = Sanitize(max_attr, 1, 0, 1);
                        break;
                    case "saturation":
                        saturation_minimum = Sanitize(min_attr, 0, 0, 1); ;
                        saturation_maximum = Sanitize(max_attr, 1, 0, 1); ;
                        break;
                    case "lightness":
                    case "value":
                        value_minimum = Sanitize(min_attr, 0, 0, 1); ;
                        value_maximum = Sanitize(max_attr, 1, 0, 1); ;
                        break;
                }
            }
            else
            {
                throw new XmlException("Missing min or max attrube in material color tag");
            }

        }
    }


        public Color Next()
    {
        return Color.HSVToRGB(  Random.Range(hue_minimum, hue_maximum), 
                                Random.Range(saturation_minimum, saturation_maximum),
                                Random.Range(value_minimum, value_maximum)  );
    }
}
