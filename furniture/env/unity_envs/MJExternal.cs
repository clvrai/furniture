using System.Xml
using UnityEngine

public static class MJExternal{
    
    // load model from xml.
    // params: modelFile: Path of model file
    public int loadModel(string modelFile){
        XmlDocument Xdoc = null;
        
        try{
            Xdoc = new XmlDocument();
            Xdoc.Load(modelFile);
        }
        catch (Exception e){
            Debug.LogException(e, this);
        }
        return Xdoc;   
    }


    public int getMainCamera(Tcamera* camera){
        Camera MainCamera = Camera.main;
        if (MainCamera==NULL) return -1;
        camera = new Tcamera;
        camera.fov = MainCamera.fieldOfView;
        camera.znear = MainCamera.nearClipPlane;                
        camera.zfar = MainCamera.farClipPlane;              
        camera.width = MainCamera.width;                  
        camera.height = MainCamera.height;
        return 0;
    }

}