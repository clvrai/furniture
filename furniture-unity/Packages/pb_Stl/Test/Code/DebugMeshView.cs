#if MESH_DEBUG_VIEW

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

namespace Parabox.Stl.Test
{
	[InitializeOnLoad]
	static class DebugMeshView
	{
		static string meshInfo = "";

		static class Styles
		{
			static bool s_Initialized;
			public static GUIStyle box;

			public static void Init()
			{
				if(s_Initialized)
					return;

				s_Initialized = true;

				box = new GUIStyle(GUI.skin.box);
				box.margin.left += 12;
				box.alignment = TextAnchor.UpperLeft;
				box.normal.textColor = new Color(.9f, .9f, .9f, 1f);
				box.fontSize = 16;
				box.fontStyle = FontStyle.Bold;
			}
		}

		static DebugMeshView()
		{
			SceneView.onSceneGUIDelegate += SceneGUI;
			Selection.selectionChanged += SelectionChanged;
		}

		static void SceneGUI(SceneView view)
		{
			Styles.Init();

			Handles.BeginGUI();

			GUILayout.Box(meshInfo, Styles.box);

			Handles.EndGUI();
		}

		static void SelectionChanged()
		{
			var meshes = Selection.GetFiltered(typeof(MeshFilter), SelectionMode.Deep);

			int vertexCount = 0;
			int triangleCount = 0;
			int meshCount = 0;

			foreach(var mesh in meshes)
			{
				var m = ((MeshFilter) mesh).sharedMesh;

				if(m == null)
					continue;

				meshCount++;
				vertexCount += m.vertexCount;
				triangleCount += m.triangles.Length;
			}

			meshInfo = "Mesh Count: " + meshes.Length +
				"\nVertex Count: " + vertexCount +
				"\nTriangle Count: " + triangleCount;
		}
	}
}
#endif
