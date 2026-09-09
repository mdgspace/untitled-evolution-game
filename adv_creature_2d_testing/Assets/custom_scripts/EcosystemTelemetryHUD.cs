using UnityEngine;

public class EcosystemTelemetryHUD : MonoBehaviour
{
    private GUIStyle style;
    private PythonBridge bridge;
    private PythonProcessManager process;
    private NativeEcosystemController native;
    private string cachedDetails = "Starting telemetry…";
    private float nextRefresh;
    private GUIContent telemetryContent;

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
        if (native != null && native.isActiveAndEnabled)
        {
            string fault = string.IsNullOrEmpty(native.LastCheckpointError) ? native.LastControlError : native.LastCheckpointError;
            cachedDetails = $"NATIVE CONTINUAL M1/M2/M3  {native.ExperimentId}\n{native.Status}\nRender {native.RenderRateHz:F1} FPS   p95 {native.FrameP95Milliseconds:F1} ms   Physics {native.PhysicsMode} {native.PhysicsRateHz:F0} Hz\nTick {native.Tick}   Sequence {native.SequenceStep}/5   hold {NativeCreatureModel.ActionTicks} ticks   M1 {(native.M1Enabled ? "rtNEAT" : "curriculum")}   practice {native.PracticeCreatureCount}\nPopulation {native.Population}/{native.populationTarget}   Species {native.SpeciesCount}   Births {native.Generation}\nM2 total/pose/energy {native.LastM2Loss:F4}/{native.LastM2GoalLoss:F4}/{native.LastM2EnergyLoss:F4}   {(native.M2WarmupComplete ? "training" : "waiting for M3 warmup")}\nM2 L/R {native.LeftProposalLoss:F3}/{native.RightProposalLoss:F3} n {native.LeftProposalSamples}/{native.RightProposalSamples}   {native.DirectionState}\nM3 total/pose/anti0 {native.LastM3Loss:F4}/{native.LastM3Mse:F4}/{native.LastM3RelativeLoss:F4}\nM3 fresh/exposures/steps {native.FreshSequenceSamples}/{native.SharedM3Samples}/{native.SharedM3OptimizerSteps}   batch {native.LastM3BatchSize}\nReplay {native.ReplayCount}/{NativeReplayBuffer.Capacity}   learner {(native.LearnerBusy ? "training" : "idle")}   queue {native.LearnerQueueDepth}   drops {native.LearnerQueueDrops}   snapshot age {native.SnapshotAgeTicks} ticks\nGoal successes/failures {native.GoalSuccesses}/{native.GoalFailures}\n{native.CurriculumTelemetry}\nGoal {native.GoalTicksRemaining}/{NativeCreatureModel.GoalWindowTicks}   recenter {native.TemporaryGoalRecenters}   bias {native.BiasStrength:P0}   noise {native.ExplorationSigma:F3}°\nPlan {native.PlannerMilliseconds:F1} ms   learner {native.M3TrainingMilliseconds:F1} ms   checkpoint {native.CheckpointMilliseconds:F1} ms\nAction {native.LastActionMagnitude:F2}°   Controls {native.ActiveControlCount}/{native.Population}   Settling {native.SettlingCreatureCount}   Moving {native.MovingCreatureCount}\nRoot {native.MeanRootSpeed:F2}   Joint {native.MeanJointSpeed:F2}   Faults {native.ControlFailures}\nSave in {native.SecondsUntilCheckpoint:F0}s   {native.CheckpointStatus}\n{fault}";
            return;
        }
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
        if (style == null) style = new GUIStyle(GUI.skin.box) { alignment = TextAnchor.UpperLeft, fontSize = 12,
                                                               wordWrap = true, padding = new RectOffset(7, 7, 6, 6) };
        telemetryContent ??= new GUIContent(); telemetryContent.text=cachedDetails;float width=Mathf.Clamp(style.CalcSize(telemetryContent).x+14f,100f,Mathf.Max(100f,Screen.width-20f));float height=style.CalcHeight(telemetryContent,width)+12f;Rect telemetryRect=new Rect(10,10,width,height);GUI.Box(telemetryRect,telemetryContent,style);
        if(native!=null&&!native.InteractiveFeaturesEnabled&&GUI.Button(new Rect(10,telemetryRect.yMax+8f,270f,28f),"Enable Food, Predators, M1, Births & Deaths"))native.EnableInteractiveFeaturesFromUser();
    }
}
