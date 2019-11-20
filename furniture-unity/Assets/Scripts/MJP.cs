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

using System.Runtime.InteropServices;
using System.Text;


public static class MJP
{

    // check for plugin error
    static void check(int code)
    {
        if( code!=0 )
        {
            // get error description from plugin
            StringBuilder err = new StringBuilder(1000);
            mjp_getErrorText(err, 1000);

            // throw exception with error message
            throw new System.Exception("MuJoCo Plugin Error " + code + " : " + err.ToString());
        }
    }


    //------------------------------ Plugin type definitions ---------------------------------------

    enum TResult : int                      // code returned by API functions
    {
        OK = 0,                             // success
        MUJOCOERROR,                        // MuJoCo error (see mjp_getErrorText)
        EXCEPTION,                          // unknown exception
        NOINIT,                             // plugin not initialized
        NOMODEL,                            // model not present
        BADMODEL,                           // failed to load model
        BADINDEX,                           // index out of range
        NULLPOINTER,                        // null pointer
        SMALLBUFFER,                        // buffer size too small
        NOFILENAME                          // file name null or empty
    };
    

    public enum TElement : int			    // element type (subset of mjtObj)
    {
        CAMERA = 7,                         // camera
        LIGHT,					            // light
        MESH,					            // mesh
        HFIELD,				                // height field
        TEXTURE,				            // texture
        MATERIAL				            // material
    };


    public enum TCategory  				    // renderable object category
    {
    	PERTBAND = 0,		                // perturbation band
    	PERTCUBE,				            // perturbation cube
    	GEOM,					            // geom
    	SITE,					            // site
    	TENDON					            // tendon
    };


    public enum TGeom : int                 // type of geometric shape (same as mjtGeom)
    {
        PLANE = 0,                          // plane
        HFIELD,                             // height field
        SPHERE,                             // sphere
        CAPSULE,                            // capsule
        ELLIPSOID,                          // ellipsoid
        CYLINDER,                           // cylinder
        BOX,                                // box
        MESH                                // mesh
    };


    public struct TOption		            // subset of physics options
    {
	    public int gravity;		            // enable gravity
	    public int equality;				// enable equality constraints
	    public int limit;					// enable joint and tendon limits
    };


    public struct TSize                     // number of model elements by type
    {
	    // copied from mjModel
	    public int nbody;					// number of bodies
	    public int nqpos;					// number of position coordinates
	    public int nmocap;					// number of mocap bodies
	    public int nmaterial;				// number of materials
	    public int ntexture;				// number of textures
	    public int nlight;					// number of lights (in addition to headlight)
	    public int ncamera;				    // number of cameras (in addition to main camera)
	    public int nkeyframe;				// number of keyframes

        // computed by plugin
        public int nobject;                 // number of renderable objects
    };


    public unsafe struct TObject            // renderable object descriptor
    {
        // common
        public int category;                // geom type (TCategory)
        public int geomtype;                // geom type (TGeom)
        public int material;                // material id; -1: none
        public int dataid;                  // mesh or hfield id; -1: none
        public fixed float color[4];        // override material color if different from (.5 .5 .5 1)

        // mesh specific
        public int mesh_shared;             // index of object containing shared mesh; -1: none
        public int mesh_nvertex;            // number of mesh vertices
        public int mesh_nface;              // number of mesh triangles
        public float* mesh_position;        // vertex position data             (nvertex x 3)
        public float* mesh_normal;          // vertex normal data               (nvertex x 3)
        public float* mesh_texcoord;        // vertex texture coordinate data   (nvertex x 2)
        public int* mesh_face;              // face index data                  (nface x 3)

        // height field specific
        public int hfield_nrow;             // number of rows (corresponding to x-axis)
        public int hfield_ncol;             // number of columns (corresponding to y-axis)
        public float* hfield_data;          // elevation data                   (ncol x nrow)
    };


    public unsafe struct TMaterial	        // material descriptor
    {
	    public int texture;				    // texture id; -1: none
	    public int texuniform;				// uniform texture cube mapping
	    public fixed float texrepeat[2];	// repetition number of 2d textures (x,y)
	    public fixed float color[4];		// main color
	    public float emission;				// emission coefficient
	    public float specular;				// specular coefficient
	    public float shininess;			    // shininess coefficient
	    public float reflectance;			// reflectance coefficient

    };


    public unsafe struct TTexture			// texture descriptor
    {
    	public int cube;					// is cube texture (as opposed to 2d)
    	public int skybox;					// is cube texture used as skybox
    	public int width;					// width in pixels
    	public int height;					// height in pixels
    	public byte* rgb;			        // RGB24 texture data
    };


    public unsafe struct TLight				// light descriptor
    {
    	public int directional;			    // is light directional
    	public int castshadow;				// does ligth cast shadows
    	public fixed float ambient[3];		// ambient rgb color
    	public fixed float diffuse[3];		// diffuse rgb color
    	public fixed float specular[3];		// specular rgb color
    	public fixed float attenuation[3];	// OpenGL quadratic attenuation model
    	public float cutoff;				// OpenGL cutoff angle for spot lights
    	public float exponent;				// OpenGL exponent
    };


    public struct TCamera			        // camera descriptor
    {
        public float fov;                   // field of view
        public float znear;                 // near depth plane
        public float zfar;                  // far depth plane
        public int width;                   // offscreen width
        public int height;					// offscreen height
    };


    public unsafe struct TPerturb	        // perturbation state
    {
        public int select;					// selected body id; non-positive: none
        public int active;					// bitmask: 1: translation, 2: rotation
        public fixed float refpos[3];		// desired position for selected object
        public fixed float refquat[4];		// desired orientation for selected object
    };


    public unsafe struct TTransform         // spatial transform
    {
        public fixed float position[3];     // position
        public fixed float xaxis[3];        // x-axis (right in MuJoCo)
        public fixed float yaxis[3];        // y-axis (forward in MuJoCo)
        public fixed float zaxis[3];        // z-axis (up in MuJoCo)
        public fixed float scale[3];        // scaling
    };


    //------------------------------ Initializaion and Simulation ---------------------------

    const string plugin = "mjplugin155";

    // initialize plugin
    [DllImport(plugin)] static extern int mjp_initialize();
    public static void Initialize() {check(mjp_initialize());}

    // close plugin
    [DllImport(plugin)] static extern int mjp_close();
    public static void Close() {check(mjp_close());}

    // load model from file in XML (MJCF or URDF) or MJB format
    [DllImport(plugin)] static extern int mjp_loadModel(string modelfile);
    public static void LoadModel(string modelfile) {check(mjp_loadModel(modelfile));}

    // save model as MJB file
    [DllImport(plugin)] static extern int mjp_saveMJB(string modelfile);
    public static void SaveMJB(string modelfile) {check(mjp_saveMJB(modelfile));}

    // reset simulation
    [DllImport(plugin)] static extern int mjp_reset();
    public static void Reset() {check(mjp_reset());}

    // reset simulation to specified keyframe
    [DllImport(plugin)] static extern int mjp_resetKeyframe(int index);
    public static void ResetKeyframe(int index) {check(mjp_resetKeyframe(index));}

    // compute forward kinematics (sufficient for rendering)
    [DllImport(plugin)] static extern int mjp_kinematics();
    public static void Kinematics() {check(mjp_kinematics());}

    // advance simulation until time marker is reached or internal reset
    [DllImport(plugin)] static extern unsafe int mjp_simulate(float marker, int paused, int* reset);
    public static unsafe void Simulate(float marker, int paused, int* reset) 
        {check(mjp_simulate(marker, paused, reset));}


    //------------------------------ Get and Set --------------------------------------------

    // (const) get model sizes
    [DllImport(plugin)] static extern unsafe int mjp_getSize(TSize* size);
    public static unsafe void GetSize(TSize* size) {check(mjp_getSize(size));}

    // (const) get name of renderable object
    [DllImport(plugin)] static extern int mjp_getObjectName(int index, StringBuilder buffer, int buffersize);
    public static void GetObjectName(int index, StringBuilder buffer, int buffersize) 
        {check(mjp_getObjectName(index, buffer, buffersize));}

    // (const) get name of mode element with specified type and index
    [DllImport(plugin)] static extern int mjp_getElementName(TElement type, int index, StringBuilder buffer, int buffersize);
    public static void GetElementName(TElement type, int index, StringBuilder buffer, int buffersize) 
        {check(mjp_getElementName(type, index, buffer, buffersize));}

    // (const) get index of body with specified name; -1: not found
    [DllImport(plugin)] static extern unsafe int mjp_getBodyIndex(string name, int* index);
    public static unsafe void GetBodyIndex(string name, int* index) {check(mjp_getBodyIndex(name, index));}

    // (const) get descriptor of specified renderable object
    [DllImport(plugin)] static extern unsafe int mjp_getObject(int index, TObject* obj);
    public static unsafe void GetObject(int index, TObject* obj) {check(mjp_getObject(index, obj));}

    // (const) get descriptor of specified material
    [DllImport(plugin)] static extern unsafe int mjp_getMaterial(int index, TMaterial* material);
    public static unsafe void GetMaterial(int index, TMaterial* material) {check(mjp_getMaterial(index, material));}

    // (const) get descriptor of specified texture
    [DllImport(plugin)] static extern unsafe int mjp_getTexture(int index, TTexture* texture);
    public static unsafe void GetTexture(int index, TTexture* texture) {check(mjp_getTexture(index, texture));}

    // (const) get descriptor of specified material
    [DllImport(plugin)] static extern unsafe int mjp_getLight(int index, TLight* light);
    public static unsafe void GetLight(int index, TLight* light) {check(mjp_getLight(index, light));}

    // (const) get descriptor of specified camera; -1: main camera
    [DllImport(plugin)] static extern unsafe int mjp_getCamera(int index, TCamera* camera);
    public static unsafe void GetCamera(int index, TCamera* camera) {check(mjp_getCamera(index, camera));}

    // get state of specified renderable object
    [DllImport(plugin)] static extern unsafe int mjp_getObjectState(int index, TTransform* transform, int* visible, int* selected);
    public static unsafe void GetObjectState(int index, TTransform* transform, int* visible, int* selected) 
        {check(mjp_getObjectState(index, transform, visible, selected));}

    // get state of specified light; -1: head light (use camera index)
    [DllImport(plugin)] static extern unsafe int mjp_getLightState(int index, int cameraindex, float* position, float* direction);
    public static unsafe void GetLightState(int index, int cameraindex, float* position, float* direction) 
        {check(mjp_getLightState(index, cameraindex, position, direction));}

    // get state of specified camera; -1: main camera
    [DllImport(plugin)] static extern unsafe int mjp_getCameraState(int index, TTransform* transform);
    public static unsafe void GetCameraState(int index, TTransform* transform) {check(mjp_getCameraState(index, transform));}

    // get state of specified body relative to parent body
    [DllImport(plugin)] static extern unsafe int mjp_getBodyRelativeState(int index, TTransform* transform);
    public static unsafe void GetBodyRelativeState(int index, TTransform* transform) {check(mjp_getBodyRelativeState(index, transform));}

    // get text description of last error
    [DllImport(plugin)] static extern int mjp_getErrorText(StringBuilder buffer, int buffersize);
    public static void GetErrorText(StringBuilder buffer, int buffersize) {check(mjp_getErrorText(buffer, buffersize));}

    // get text description of last warning
    [DllImport(plugin)] static extern int mjp_getWarningText(StringBuilder buffer, int buffersize);
    public static void GetWarningText(StringBuilder buffer, int buffersize) {check(mjp_getWarningText(buffer, buffersize));}

    // get number of warnings since last load or reset
    [DllImport(plugin)] static extern unsafe int mjp_getWarningNumber(int* number);
    public static unsafe void GetWarningNumber(int* number) {check(mjp_getWarningNumber(number));}

    // set system position vector; size(qpos) = nqpos
    [DllImport(plugin)] static extern unsafe int mjp_setQpos(float* qpos);
    public static unsafe void SetQpos(float* qpos) {check(mjp_setQpos(qpos));}

    // set all mocap body poses; size(pos) = 3*nmocap, size(quat) = 4*nmocap
    [DllImport(plugin)] static extern unsafe int mjp_setMocap(float* pos, float* quat);
    public static unsafe void SetMocap(float* pos, float* quat) {check(mjp_setMocap(pos, quat));}

    // get simulation time
    [DllImport(plugin)] static extern unsafe int mjp_getTime(float* time);
    public static unsafe void GetTime(float* time) {check(mjp_getTime(time));}

    // set simulation time
    [DllImport(plugin)] static extern int mjp_setTime(float time);
    public static void SetTime(float time) {check(mjp_setTime(time));}

    // get options
    [DllImport(plugin)] static extern unsafe int mjp_getOption(TOption* option);
    public static unsafe void GetOption(TOption* option) {check(mjp_getOption(option));}

    // set options
    [DllImport(plugin)] static extern unsafe int mjp_setOption(TOption* option);
    public static unsafe void SetOption(TOption* option) {check(mjp_setOption(option));}

    // get perturbation state
    [DllImport(plugin)] static extern unsafe int mjp_getPerturb(TPerturb* perturb);
    public static unsafe void GetPerturb(TPerturb* perturb) {check(mjp_getPerturb(perturb));}

    // set perturbation state
    [DllImport(plugin)] static extern unsafe int mjp_setPerturb(TPerturb* perturb);
    public static unsafe void SetPerturb(TPerturb* perturb) {check(mjp_setPerturb(perturb));}


    //------------------------------ Camera and Perturbation --------------------------------

    // set main camera lookat point; aspect = width/height
    [DllImport(plugin)] static extern int mjp_cameraLookAt(float x, float y, float aspect);
    public static void CameraLookAt(float x, float y, float aspect) {check(mjp_cameraLookAt(x, y, aspect));}

    // zoom main camera
    [DllImport(plugin)] static extern int mjp_cameraZoom(float zoom);
    public static void CameraZoom(float zoom) {check(mjp_cameraZoom(zoom));}

    // move main camera (i.e. lookat point)
    [DllImport(plugin)] static extern int mjp_cameraMove(float dx, float dy, int modified);
    public static void CameraMove(float dx, float dy, int modified) {check(mjp_cameraMove(dx, dy, modified));}

    // rotate main camera around lookat point
    [DllImport(plugin)] static extern int mjp_cameraRotate(float dx, float dy);
    public static void CameraRotate(float dx, float dy) {check(mjp_cameraRotate(dx, dy));}

    // set active bitmask only (use setPerturb for full access)
    [DllImport(plugin)] static extern int mjp_perturbActive(int state);
    public static void PerturbActive(int state) {check(mjp_perturbActive(state));}

    // set perturb object pose equal to selected body pose
    [DllImport(plugin)] static extern int mjp_perturbSynchronize();
    public static void PerturbSynchronize() {check(mjp_perturbSynchronize());}

    // select body for perturbation
    [DllImport(plugin)] static extern int mjp_perturbSelect(float x, float y, float aspect);
    public static void PerturbSelect(float x, float y, float aspect) {check(mjp_perturbSelect(x, y, aspect));}

    // move perturbation object
    [DllImport(plugin)] static extern int mjp_perturbMove(float dx, float dy, int modified);
    public static void PerturbMove(float dx, float dy, int modified) {check(mjp_perturbMove(dx, dy, modified));}

    // rotate perturbation object
    [DllImport(plugin)] static extern int mjp_perturbRotate(float dx, float dy, int modified);
    public static void PerturbRotate(float dx, float dy, int modified) {check(mjp_perturbRotate(dx, dy, modified));}
}
