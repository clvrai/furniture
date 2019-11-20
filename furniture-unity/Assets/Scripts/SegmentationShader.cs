/*
Copyright 2019 Panasonic Beta, a division of Panasonic Corporation of North America

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

using UnityEngine;
using System.Collections;

public class SegmentationShader : MonoBehaviour
{
    private Shader shader;
    private bool hdr;
    private bool aa;
    private Color bgcolor;
    private CameraClearFlags flags;
    bool active;
    // Use this for initialization
    void Start()
    {
        shader = Shader.Find("Unlit/SegmentationColor");
        active = false;
    }

    public void Activate()
    {
        if (active == false)
        {
            Camera this_camera = GetComponent<Camera>();

            hdr = this_camera.allowHDR;
            aa = this_camera.allowMSAA;
            bgcolor = this_camera.backgroundColor;
            flags = this_camera.clearFlags;
        
            this_camera.allowHDR = false;
            this_camera.allowMSAA = false;
            this_camera.backgroundColor = new Color(0, 0, 0, 0);
            this_camera.clearFlags = CameraClearFlags.Color;
            this_camera.SetReplacementShader(shader, null);
            active = true;
        }
    }

    public void Deactivate()
    {
        if (active == true)
        {
            Camera this_camera = GetComponent<Camera>();
            this_camera.allowHDR = hdr;
            this_camera.allowMSAA = aa;
            this_camera.backgroundColor = bgcolor;
            this_camera.clearFlags = flags;
            this_camera.SetReplacementShader(null, null);
            active = false;
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
