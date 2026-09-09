using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;

internal class CommandScript : IRunCommand
{
    static void ForceRealtime(PlayModeStateChange state)
    {
        if (state != PlayModeStateChange.EnteredPlayMode) return;
        Time.timeScale = 1f;
        EditorApplication.playModeStateChanged -= ForceRealtime;
    }

    public void Execute(ExecutionResult result)
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode) { result.LogError("Editor must be fully outside Play Mode"); return; }
        EditorSceneManager.OpenScene("Assets/Scenes/SampleScene.unity", OpenSceneMode.Single);
        var controller = Object.FindAnyObjectByType<NativeEcosystemController>();
        if (controller == null) { result.LogError("Native controller missing from SampleScene"); return; }
        controller.loadCheckpointOnStart = false;
        Time.timeScale = 1f;
        EditorApplication.playModeStateChanged += ForceRealtime;
        EditorApplication.isPlaying = true;
        result.Log("Entered a fresh, unsaved 50 Hz validation run");
    }
}
