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
        if (EditorApplication.isPlaying) EditorApplication.isPlaying = false;
        Time.timeScale = 1f;
        EditorSceneManager.OpenScene("Assets/Scenes/SampleScene.unity", OpenSceneMode.Single);
        EditorApplication.playModeStateChanged += ForceRealtime;
        EditorApplication.isPlaying = true;
        result.Log("Opened SampleScene and entered Play Mode for locomotion validation");
    }
}
