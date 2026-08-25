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
        if (native != null && native.isActiveAndEnabled) { string fault = string.IsNullOrEmpty(native.LastCheckpointError) ? native.LastControlError : native.LastCheckpointError; cachedDetails = $"NATIVE ECOSYSTEM\n{native.Status}\nRender {native.RenderRateHz:F1} FPS   Physics {native.PhysicsMode}\nTick {native.Tick}   Action cadence {native.controlIntervalTicks} ticks   Generation {native.Generation}\nPopulation {native.Population}/{native.populationTarget}   Nested body species {native.SpeciesCount}\nM2 horizon loss {native.LastM2Loss:F5}   M3 transition loss {native.LastM3Loss:F5}\nGoal window {native.GoalTicksRemaining}/{NativeCreatureModel.GoalWindowTicks} ticks   Plan rollouts {native.PlannerRollouts}   Planner {native.PlannerMilliseconds:F2} ms   Refinement gain {native.RefinementGain:F4}\nPredicted ground/predator {native.PredictedGroundRisk:F2}/{native.PredictedPredatorRisk:F2}   Actual {native.ActualGroundRisk:F0}/{native.ActualPredatorRisk:F0}\nMean action {native.LastActionMagnitude:F2} degrees   Bounds {NativeCreatureModel.MinimumActionDegrees:F2}..{NativeCreatureModel.MaximumActionDegrees:F2}\nControls {native.ActiveControlCount}/{native.Population}   Dead {native.DeadCreatureCount}   Moving sustained {native.MovingCreatureCount}\nReplacements {native.ImmediateReplacementCount}   Last replacement tick {native.LastReplacementTick}\nRoot speed {native.MeanRootSpeed:F2}   Joint speed {native.MeanJointSpeed:F2}   Control faults {native.ControlFailures}   Snapshot {native.SnapshotVersion}\nNext checkpoint {native.SecondsUntilCheckpoint:F1}s / interval {native.checkpointIntervalSeconds:F1}s\nLast checkpoint {native.LastCheckpointSavedAt}   Status {native.CheckpointStatus}\n{fault}"; return; }
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
        GUI.Box(new Rect(10, 10, 760, 530), cachedDetails, style);
    }
}
