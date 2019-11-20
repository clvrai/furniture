using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DepthShader : MonoBehaviour
{
    private Shader shader;
    private bool hdr;
    private bool aa;
    private Color bgcolor;
    private CameraClearFlags flags;
    bool active;

    // Start is called before the first frame update
    void Start()
    {
        shader = Shader.Find("Custom/DepthGrayscale");
        active = false;
    }

    public void Activate()
    {
        if (active == false) {
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
        if (active == true) {
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
