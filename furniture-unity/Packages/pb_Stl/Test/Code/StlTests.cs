using UnityEngine;
using UnityEditor;
using NUnit.Framework;
using System.Collections.Generic;
using Parabox.Stl;
using System.IO;

namespace Parabox.Stl.Test
{
	/// <summary>
	/// Editor tests for Stl lib.
	/// </summary>
	sealed class StlTests
	{
		const string k_TempDir = "Assets/Stl-Editor-Tests-Temp";
		const string k_TestModelsDir = "Packages/co.parabox.stl/Test/Models/";

		[Test]
		public void VerifyWriteASCII()
		{
			DoVerifyWriteString(k_TestModelsDir + "Cylinder_ASCII_RH.stl", GameObject.CreatePrimitive(PrimitiveType.Cylinder));
			DoVerifyWriteString(k_TestModelsDir + "Sphere_ASCII_RH.stl", GameObject.CreatePrimitive(PrimitiveType.Sphere));
		}

		[Test]
		public void VerifyWriteBinary()
		{
			if(!Directory.Exists(k_TempDir))
				Directory.CreateDirectory(k_TempDir);

			DoVerifyWriteBinary(k_TestModelsDir + "Cylinder_BINARY_RH.stl", GameObject.CreatePrimitive(PrimitiveType.Cylinder));
			DoVerifyWriteBinary(k_TestModelsDir + "Sphere_BINARY_RH.stl", GameObject.CreatePrimitive(PrimitiveType.Sphere));

			Directory.Delete(k_TempDir, true);
		}

		[Test]
		public void TestExportMultiple()
		{
			GameObject a = GameObject.CreatePrimitive(PrimitiveType.Cube);
			GameObject b = GameObject.CreatePrimitive(PrimitiveType.Cube);

			a.transform.position = Vector3.right;
			b.transform.position = new Vector3(3f, 5f, 2.4f);
			b.transform.localRotation = Quaternion.Euler( new Vector3(45f, 45f, 10f) );

			if(!Directory.Exists(k_TempDir))
				Directory.CreateDirectory(k_TempDir);

			string temp_model_path = string.Format("{0}/multiple.stl", k_TempDir);
			Exporter.Export(temp_model_path, new GameObject[] { a, b }, FileType.Binary );

			// Comparing binary files isn't great
			// Assert.IsTrue(CompareFiles(string.Format("{0}/CompositeCubes_BINARY.stl", k_TestModelsDir), temp_model_path));
			Mesh[] expected = Importer.Import(string.Format("{0}/CompositeCubes_BINARY.stl", k_TestModelsDir));
			Mesh[] results = Importer.Import(temp_model_path);

			Assert.IsTrue(expected != null);
			Assert.IsTrue(results != null);

			Assert.IsTrue(expected.Length == 1);
			Assert.IsTrue(results.Length == 1);

			Assert.AreEqual(expected[0].vertexCount, results[0].vertexCount);
			Assert.AreEqual(expected[0].triangles, results[0].triangles);

			// Can't use Assert.AreEqual(positions, normals, uvs) because Vec3 comparison is subject to floating point inaccuracy
			for(int i = 0; i < expected[0].vertexCount; i++)
			{
				Assert.Less( Vector3.Distance(expected[0].vertices[i], results[0].vertices[i]), .00001f );
				Assert.Less( Vector3.Distance(expected[0].normals[i], results[0].normals[i]), .00001f );
			}

			GameObject.DestroyImmediate(a);
			GameObject.DestroyImmediate(b);

			Directory.Delete(k_TempDir, true);
		}

		[Test]
		public void TestImportAscii()
		{
			Mesh[] meshes = Importer.Import(string.Format("{0}/Cylinder_ASCII_RH.stl", k_TestModelsDir));
			Assert.IsTrue(meshes != null);
			Assert.AreEqual(1, meshes.Length);
			Assert.AreEqual(240, meshes[0].triangles.Length);
			Assert.AreEqual(240, meshes[0].vertexCount);
		}

		[Test]
		public void TestImportBinary()
		{
			Mesh[] meshes = Importer.Import(string.Format("{0}/Cylinder_BINARY_RH.stl", k_TestModelsDir));
			Assert.IsTrue(meshes != null);
			Assert.AreEqual(1, meshes.Length);
			Assert.AreEqual(240, meshes[0].triangles.Length);
			Assert.AreEqual(240, meshes[0].vertexCount);
		}

		[Test]
		public void TestImportBinaryWithHeaders()
		{
			Mesh[] meshes = Importer.Import(string.Format("{0}/CubedShape_BINARY_H.stl", k_TestModelsDir));
			Assert.IsTrue(meshes != null);
			Assert.AreEqual(1, meshes.Length);
			Assert.AreEqual(204, meshes[0].triangles.Length);
			Assert.AreEqual(204, meshes[0].vertexCount);
		}

		private void DoVerifyWriteBinary(string expected_path, GameObject go)
		{
			string temp_model_path = string.Format("{0}/binary_file.stl", k_TempDir);

			Assert.IsTrue(Exporter.WriteFile(temp_model_path, go.GetComponent<MeshFilter>().sharedMesh, FileType.Binary));
			Assert.IsTrue(CompareFiles(temp_model_path, expected_path));

			GameObject.DestroyImmediate(go);
		}

		private void DoVerifyWriteString(string path, GameObject go)
		{
			string ascii = Exporter.WriteString(go.GetComponent<MeshFilter>().sharedMesh, true);
			// Replace Windows line endings with Unix
			// @todo Does STL spec care about line endings?
			ascii = ascii.Replace("\r\n", "\n");
			Assert.AreNotEqual(ascii, null);
			Assert.AreNotEqual(ascii, "");
			string expected = File.ReadAllText(path);
			Assert.AreNotEqual(expected, null);
			Assert.AreNotEqual(expected, "");
			Assert.AreEqual(ascii, expected);
			GameObject.DestroyImmediate(go);
		}

		private bool CompareFiles(string left, string right)
		{
			if(left == null || right == null)
				return false;

			// http://stackoverflow.com/questions/968935/compare-binary-files-in-c-sharp
			FileInfo a = new FileInfo(left);
			FileInfo b = new FileInfo(right);

			if(a.Length != b.Length)
				return false;

			using(FileStream f0 = a.OpenRead())
			using(FileStream f1 = b.OpenRead())
			using(BufferedStream bs0 = new BufferedStream(f0))
			using(BufferedStream bs1 = new BufferedStream(f1))
			{
				for(long i = 0; i < a.Length; i++)
				{
					if(bs0.ReadByte() != bs1.ReadByte())
					{
						return false;
					}
				}
			}

			return true;
		}
	}
}
