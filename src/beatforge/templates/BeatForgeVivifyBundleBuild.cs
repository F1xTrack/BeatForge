using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEngine;
using VivifyTemplate.Exporter.Editor.Build;
using VivifyTemplate.Exporter.Editor.Build.Structures;
using VivifyTemplate.Exporter.Editor.PlayerPrefs;
using VivifyTemplate.Exporter.Editor.Project;

public static class BeatForgeVivifyBundleBuild
{
    [Serializable]
    private sealed class BuildSpec
    {
        public string unityVersion;
        public string buildVersion;
        public string bundleName;
        public string outputDirectory;
        public string resultPath;
        public AssetSpec[] assets;
    }

    [Serializable]
    private sealed class AssetSpec
    {
        public string asset_id;
        public string source_model_path;
        public string prefab_asset_path;
        public string vivify_asset_path;
        public string source_archive_sha256;
    }

    private static string Argument(string name)
    {
        string[] args = Environment.GetCommandLineArgs();
        for (int i = 0; i + 1 < args.Length; i++)
        {
            if (args[i] == name)
                return args[i + 1];
        }
        throw new ArgumentException("Missing command line argument: " + name);
    }

    public static void Build()
    {
        string resultPath = null;
        try
        {
            string specPath = Argument("-beatforgeSpec");
            BuildSpec spec = JsonUtility.FromJson<BuildSpec>(File.ReadAllText(specPath));
            if (spec == null || spec.assets == null || spec.assets.Length == 0)
                throw new InvalidDataException("Build spec contains no model assets.");
            resultPath = spec.resultPath;
            if (Application.unityVersion != spec.unityVersion || spec.unityVersion != "2021.3.16f1")
                throw new InvalidOperationException("Expected Unity 2021.3.16f1; found " + Application.unityVersion);
            if (spec.buildVersion != "Windows2021" || spec.bundleName != "beatforge")
                throw new InvalidOperationException("Unsupported BeatForge Vivify build target.");

            if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.StandaloneWindows64)
                throw new InvalidOperationException("Launch Unity with -buildTarget Win64.");

            SetupProject.Setup();
            foreach (AssetSpec asset in spec.assets)
                CreatePrefab(asset, spec.bundleName);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Directory.CreateDirectory(spec.outputDirectory);
            ProjectBundle.Value = spec.bundleName;

            var settings = new BuildSettings
            {
                OutputDirectory = spec.outputDirectory,
                ProjectBundle = spec.bundleName,
                WorkingVersion = BuildVersion.Windows2021,
                ShouldExportBundleInfo = false,
                ShouldPrettifyBundleInfo = true,
            };
            var logger = new AccumulatingLogger();
            logger.OnLog += Debug.Log;
            BuildReport report = BuildAssetBundles.Build(
                settings, BuildAssetBundleOptions.None, BuildVersion.Windows2021, logger,
                task => task.GetLogger().OnLog += Debug.Log).GetAwaiter().GetResult();
            if (!File.Exists(report.OutputBundlePath) || new FileInfo(report.OutputBundlePath).Length == 0)
                throw new IOException("VivifyTemplate returned without a non-empty bundle.");

            string assets = string.Join(",", spec.assets.Select(asset =>
                "{\"assetId\":\"" + Escape(asset.asset_id) +
                "\",\"vivifyAssetPath\":\"" + Escape(asset.vivify_asset_path) + "\"}"));
            string json = "{\"unityVersion\":\"" + Escape(Application.unityVersion) +
                "\",\"buildVersion\":\"Windows2021\",\"bundleFile\":\"bundleWindows2021.vivify\"" +
                ",\"crc\":\"" + report.CRC.ToString(CultureInfo.InvariantCulture) +
                "\",\"assets\":[" + assets + "]}";
            File.WriteAllText(resultPath, json, new UTF8Encoding(false));
            Debug.Log("BeatForge Vivify bundle build completed: " + report.OutputBundlePath);
            EditorApplication.Exit(0);
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            if (!string.IsNullOrEmpty(resultPath))
            {
                try
                {
                    File.WriteAllText(resultPath,
                        "{\"error\":\"" + Escape(exception.ToString()) + "\"}",
                        new UTF8Encoding(false));
                }
                catch (Exception writeException)
                {
                    Debug.LogException(writeException);
                }
            }
            EditorApplication.Exit(1);
        }
    }

    private static void CreatePrefab(AssetSpec asset, string bundleName)
    {
        if (string.IsNullOrWhiteSpace(asset.asset_id) || asset.asset_id.Length != 20)
            throw new InvalidDataException("Invalid asset ID in build spec.");
        if (string.IsNullOrWhiteSpace(asset.source_model_path)
            || !asset.source_model_path.StartsWith("Assets/BeatForgeImported/", StringComparison.Ordinal))
            throw new InvalidDataException("Model path is outside staged Unity Assets.");
        if (!string.Equals(asset.prefab_asset_path,
                "Assets/assets/" + asset.asset_id + "/" + asset.asset_id + ".prefab",
                StringComparison.Ordinal))
            throw new InvalidDataException("Prefab path does not match the deterministic asset path.");

        GameObject model = AssetDatabase.LoadAssetAtPath<GameObject>(asset.source_model_path);
        if (model == null)
            throw new FileNotFoundException("Unity could not import staged model: " + asset.source_model_path);
        string prefabFullPath = Path.GetFullPath(Path.Combine(
            Directory.GetCurrentDirectory(), asset.prefab_asset_path));
        Directory.CreateDirectory(Path.GetDirectoryName(prefabFullPath));

        GameObject root = new GameObject(asset.asset_id);
        try
        {
            GameObject modelInstance = PrefabUtility.InstantiatePrefab(model) as GameObject;
            if (modelInstance == null)
                throw new InvalidOperationException("Unity could not instantiate imported model " + asset.asset_id);
            modelInstance.transform.SetParent(root.transform, false);
            GameObject prefab = PrefabUtility.SaveAsPrefabAsset(root, asset.prefab_asset_path);
            if (prefab == null)
                throw new IOException("Unity failed to save prefab " + asset.prefab_asset_path);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(root);
        }

        AssetImporter importer = AssetImporter.GetAtPath(asset.prefab_asset_path);
        if (importer == null)
            throw new InvalidOperationException("Unity did not import generated prefab " + asset.prefab_asset_path);
        importer.assetBundleName = bundleName;
        importer.assetBundleVariant = string.Empty;
        importer.SaveAndReimport();
    }

    private static string Escape(string value)
    {
        return (value ?? string.Empty).Replace("\\", "\\\\").Replace("\"", "\\\"")
            .Replace("\r", "\\r").Replace("\n", "\\n");
    }
}
