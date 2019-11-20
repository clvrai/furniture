// Standard shader with triplanar mapping
// https://github.com/keijiro/StandardTriplanar

using UnityEngine;
using UnityEditor;

public class StandardTriplanarInspector : ShaderGUI
{
    static class Styles
    {
        static public readonly GUIContent albedo = new GUIContent("Albedo", "Albedo (RGB)");
        static public readonly GUIContent normalMap = new GUIContent("Normal Map", "Normal Map");
        static public readonly GUIContent occlusion = new GUIContent("Occlusion", "Occlusion (G)");
    }

    bool _initialized;

    public override void OnGUI(MaterialEditor editor, MaterialProperty[] props)
    {
        EditorGUI.BeginChangeCheck();

        editor.TexturePropertySingleLine(
            Styles.albedo, FindProperty("_MainTex", props), FindProperty("_Color", props)
        );

        editor.ShaderProperty(FindProperty("_Metallic", props), "Metallic");
        editor.ShaderProperty(FindProperty("_Glossiness", props), "Smoothness");

        var normal = FindProperty("_BumpMap", props);
        editor.TexturePropertySingleLine(
            Styles.normalMap, normal,
            normal.textureValue ? FindProperty("_BumpScale", props) : null
        );

        var occ = FindProperty("_OcclusionMap", props);
        editor.TexturePropertySingleLine(
            Styles.occlusion, occ,
            occ.textureValue ? FindProperty("_OcclusionStrength", props) : null
        );

        editor.ShaderProperty(FindProperty("_MapScale", props), "Texture Scale");

        if (EditorGUI.EndChangeCheck() || !_initialized)
            foreach (Material m in editor.targets)
                SetMaterialKeywords(m);

        _initialized = true;
    }

    static void SetMaterialKeywords(Material material)
    {
        SetKeyword(material, "_NORMALMAP", material.GetTexture("_BumpMap"));
        SetKeyword(material, "_OCCLUSIONMAP", material.GetTexture("_OcclusionMap"));
    }

    static void SetKeyword(Material m, string keyword, bool state)
    {
        if (state)
            m.EnableKeyword(keyword);
        else
            m.DisableKeyword(keyword);
    }
}
