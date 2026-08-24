using UnityEngine;

public class EcosystemTelemetryHUD : MonoBehaviour
{
    private GUIStyle style;
    private PythonBridge bridge;
    private PythonProcessManager process;
    private NativeEcosystemController native;
    private string cachedDetails = "Starting telemetry…";
    private float nextRefresh;

    private void Awake()
    {
        bridge = GetComponent<PythonBridge>();
        process = GetComponent<PythonProcessManager>();
        native = GetComponent<NativeEcosystemController>();
    }

    private void Update()
    {
        if (native == null) native = GetComponent<NativeEcosystemController>();
        if (Time.unscaledTime < nextRefresh) return;
        nextRefresh = Time.unscaledTime + .25f;
        if (native != null && native.isActiveAndEnabled) { cachedDetails = string.Format("NATIVE ECOSYSTEM\n{0}\nRender {1:F1} FPS   Physics {2}\nTick {3}   Control interval {4} ticks   Generation {5}\nPopulation {6}/{7}   Nested body species {8}\nLatest M2 goal loss {9:F5}   Dynamics loss {10:F5}\nM1 energy fitness {11:F2}   M3 replay {12}\nMean action {13:F2} degrees   Bounds {14:F2}..{15:F2}\nControls {16}/{6}   Moving {17}   Root {18:F2}   Joint {19:F2}\nControl faults {20}   Snapshot {21}\nNext checkpoint {22:F1}s / interval {23:F1}s\nLast checkpoint {24}   Status {25}\n{26}", native.Status, native.RenderRateHz, native.PhysicsMode, native.Tick, native.controlIntervalTicks, native.Generation, native.Population, native.populationTarget, native.SpeciesCount, native.LastM2Loss, native.LastM3Loss, native.MeanM1EnergyFitness, native.DynamicsReplaySamples, native.LastActionMagnitude, NativeCreatureModel.MinimumActionDegrees, NativeCreatureModel.MaximumActionDegrees, native.ActiveControlCount, native.MovingCreatureCount, native.MeanRootSpeed, native.MeanJointSpeed, native.ControlFailures, native.SnapshotVersion, native.SecondsUntilCheckpoint, native.checkpointIntervalSeconds, native.LastCheckpointSavedAt, native.CheckpointStatus, string.IsNullOrEmpty(native.LastCheckpointError) ? native.LastControlError : native.LastCheckpointError); return; }
        string transport = bridge == null ? "No bridge" : bridge.ConnectionStatus;
        string processState = process == null ? "No process manager" : process.Status;
        string details = bridge == null ? "" : string.Format(
            "SIMULATION: {0}\nFrames {1} ({2:F1} Hz)   Physics {3} ({4:F1} Hz)   scale {5:F2}\nFrame age {6:F2}s   Step age {7:F2}s   Step total {8:F2} ms\nCONTROL: {9}\nObservation age {10:F2}s   Response age {11:F2}s   Round trip {12:F1} ms\nCurrent tick {13}   Response tick {14}   gap {15}   Actions applied {16}\nPopulation {17}/{18}   Species {19}   Dropped obs {20}   Events {21}\nPer-creature: age / source / command / root speed / joint speed / mode / state\n(Fixed 50 Hz; Python every 4 physics ticks; physics never waits):\n{23}\nPYTHON: observe {24:F1} ms @ tick {25}   inference {26:F1} ms @ tick {27}\nLearner {28:F1} ms @ tick {29} ({30})   updates {31}   lag {32}   dropped {33}   replay {34}/{35} moving\nM3 loss {36:F5}   Checkpoint {37}\n{38}",
            bridge.RuntimeWaitStatus, bridge.FrameCount, bridge.RenderRateHz, bridge.PhysicsStepCount, bridge.PhysicsRateHz, Time.timeScale,
            bridge.LastFrameAgeSeconds, bridge.LastFixedStepAgeSeconds, bridge.LastPhysicsStepMilliseconds,
            bridge.PipelineStatus, bridge.LastObservationAgeSeconds, bridge.LastResponseAgeSeconds, bridge.LastRoundTripMilliseconds,
            bridge.PhysicsStepCount, bridge.LastResponseSourceTick, bridge.ResponseTickGap, bridge.LastActionsApplied,
            bridge.Population, bridge.populationTarget, bridge.SpeciesCount, bridge.DroppedObservations, bridge.PendingEventCount,
            bridge.ActionStarvationThresholdTicks, bridge.CreatureControlSummary,
            bridge.ServerObserveMilliseconds, bridge.ObserveMetricsTick, bridge.ServerInferenceMilliseconds, bridge.InferenceMetricsTick,
            bridge.LearnerUpdateMilliseconds, bridge.LearnerMetricsTick, bridge.LearnerBusy ? "busy" : "idle",
            bridge.LearnerUpdates, bridge.LearnerLag, bridge.DroppedLearning, bridge.ReplaySize, bridge.MotionReplaySize,
            bridge.M3Error, bridge.CheckpointStatus,
            string.IsNullOrEmpty(bridge.LastServerError) ? (process == null ? "" : process.LastError) : bridge.LastServerError);
        cachedDetails = transport + " / " + processState + "\n" + details;
    }

    private void OnGUI()
    {
        if (style == null) style = new GUIStyle(GUI.skin.box) { alignment = TextAnchor.UpperLeft, fontSize = 14,
                                                               wordWrap = true, padding = new RectOffset(10, 10, 8, 8) };
        GUI.Box(new Rect(10, 10, 680, 440), cachedDetails, style);
    }
}
