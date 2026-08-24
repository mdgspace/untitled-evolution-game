using UnityEngine;

[DisallowMultipleComponent]
[RequireComponent(typeof(NativeEcosystemController), typeof(EnvironmentSpawner))]
[RequireComponent(typeof(EcosystemTelemetryHUD))]
// The surviving scene object is the composition root for the live ecosystem.
// The native controller owns the runtime ecosystem. Python components remain
// in the project for legacy inspection but are not required by this scene.
public class FixedSpawner : MonoBehaviour
{
    private void Awake()
    {
        Application.runInBackground = true;
        PythonBridge legacyBridge = GetComponent<PythonBridge>();
        if (legacyBridge != null) legacyBridge.enabled = false;
        PythonProcessManager legacyProcess = GetComponent<PythonProcessManager>();
        if (legacyProcess != null) legacyProcess.enabled = false;
        Physics2D.simulationMode = SimulationMode2D.FixedUpdate;
        if (GetComponent<EcosystemTelemetryHUD>() == null) gameObject.AddComponent<EcosystemTelemetryHUD>();
        if (GetComponent<NativeEcosystemController>() == null) gameObject.AddComponent<NativeEcosystemController>();
        if (FindAnyObjectByType<EnvironmentSpawner>() == null) gameObject.AddComponent<EnvironmentSpawner>();
        Camera observerCamera = FindAnyObjectByType<Camera>();
        if (observerCamera != null && observerCamera.GetComponent<ObserverCamera>() == null)
            observerCamera.gameObject.AddComponent<ObserverCamera>();
    }
}
