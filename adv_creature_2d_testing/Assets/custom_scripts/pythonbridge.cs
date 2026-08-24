using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;
using Stopwatch = System.Diagnostics.Stopwatch;

[DefaultExecutionOrder(-50)]
public class PythonBridge : MonoBehaviour
{
    public const int ProtocolVersion = 4;
    private const int MaxMessageBytes = 8 * 1024 * 1024;
    private const int NominalControlTicks = 4;
    private const int MaxActionAgeTicks = 8;
    private const int HeldActionLimitTicks = 50;
    private const float FixedStepSeconds = .02f;

    public string host = "127.0.0.1";
    public int port = 9999;
    [Tooltip("Retained for scene compatibility. Control is fixed at four physics ticks.")]
    public int pythonCallInterval = NominalControlTicks;
    [Tooltip("Retained for scene compatibility. The asynchronous grace window is four ticks.")]
    public int slowPythonCallInterval = 5;
    public float slowRoundTripThresholdMilliseconds = 40f;
    public int populationTarget = 4;
    [SerializeField] private int maximumCatchUpSteps = 4;

    public string ConnectionStatus { get; private set; } = "Starting";
    public string LastServerError { get; private set; } = "";
    public string CheckpointStatus { get; private set; } = "Unknown";
    public int Population => creatures.Count;
    public int SpeciesCount { get; private set; }
    public int DroppedObservations { get; private set; }
    public int PendingEventCount => lifecycleEvents.Count;
    public int LearnerLag { get; private set; }
    public float M3Error { get; private set; }
    public int OldestActionAge { get; private set; }
    public int FrameCount { get; private set; }
    public int PhysicsStepCount { get; private set; }
    public float RenderRateHz { get; private set; }
    public float PhysicsRateHz { get; private set; }
    public float LastPhysicsStepMilliseconds { get; private set; }
    public int LastResponseSourceTick { get; private set; } = -1;
    public int ResponseTickGap => LastResponseSourceTick < 0 ? -1 : Mathf.Max(0, tick - LastResponseSourceTick);
    public int LastActionsApplied { get; private set; }
    public float LastRoundTripMilliseconds { get; private set; }
    public float ServerObserveMilliseconds { get; private set; }
    public float ServerInferenceMilliseconds { get; private set; }
    public float LearnerUpdateMilliseconds { get; private set; }
    public int LearnerUpdates { get; private set; }
    public int DroppedLearning { get; private set; }
    public int ReplaySize { get; private set; }
    public int MotionReplaySize { get; private set; }
    public bool LearnerBusy { get; private set; }
    public int ObserveMetricsTick { get; private set; } = -1;
    public int InferenceMetricsTick { get; private set; } = -1;
    public int LearnerMetricsTick { get; private set; } = -1;
    public string PipelineStatus { get; private set; } = "Starting asynchronous transport";
    public bool ControllerFaultPaused { get; private set; }
    public int CurrentControlInterval => NominalControlTicks;
    public int FramesUntilPythonCall => outstandingObservationId == null ? Mathf.Max(0, nextObservationTick - tick) : 0;
    public int ActionStarvationThresholdTicks => MaxActionAgeTicks;
    public float LastFrameAgeSeconds => AgeSince(lastFrameTimestamp);
    public float LastFixedStepAgeSeconds => AgeSince(lastFixedTimestamp);
    public float LastObservationAgeSeconds => AgeSince(lastObservationTimestamp);
    public float LastResponseAgeSeconds => AgeSince(lastResponseTimestamp);
    public string RuntimeWaitStatus {
        get {
            if (!isActiveAndEnabled) return "Bridge disabled: manual 2D physics is not being stepped";
            if (ControllerFaultPaused) return "Controller fault: simulation paused after one second of held action";
            if (!receivedInitialWorld) return "Waiting for asynchronous Python bootstrap";
            if (outstandingObservationId != null) return "Physics continues while Python evaluates tick " + outstandingObservationTick;
            return "Asynchronous hybrid control: fixed physics never waits for Python";
        }
    }
    public string CreatureControlSummary {
        get {
            StringBuilder summary = new StringBuilder();
            foreach (CreatureBrain brain in creatures) {
                if (brain == null || brain.IsDead || brain.torso == null || brain.torso.bodyPart == null) continue;
                int age = brain.ActionAge(tick); Rigidbody2D root = brain.torso.Rigidbody;
                if (summary.Length > 0) summary.Append("\n");
                string id = brain.torso.bodyPart.identity.creatureId;
                summary.Append(id.Substring(0, Math.Min(6, id.Length))).Append(": ").Append(age).Append(" ticks / source ").Append(brain.LastActionSourceTick).Append(" / cmd ").Append(brain.LastActionMagnitude.ToString("F1")).Append("° / v ").Append(root == null ? "?" : root.linearVelocity.magnitude.ToString("F2")).Append(" / joint ").Append(brain.MeanJointSpeed.ToString("F1")).Append("°/s / ").Append(brain.ControlMode).Append(age > ActionStarvationThresholdTicks ? "  HELD" : "  active");
            }
            return summary.Length == 0 ? "No living creature control state" : summary.ToString();
        }
    }

    private readonly List<CreatureBrain> creatures = new List<CreatureBrain>();
    private readonly Dictionary<string, JObject> lifecycleEvents = new Dictionary<string, JObject>();
    private readonly HashSet<string> handledCommands = new HashSet<string>();
    private readonly Queue<string> handledCommandOrder = new Queue<string>();
    private readonly Dictionary<string, int> spawnSlots = new Dictionary<string, int>();
    private readonly Dictionary<string, JObject> reportedSegments = new Dictionary<string, JObject>();
    private readonly ConcurrentQueue<JObject> incoming = new ConcurrentQueue<JObject>();
    private readonly ConcurrentQueue<string> transportEvents = new ConcurrentQueue<string>();
    private readonly ConcurrentQueue<JObject> outgoing = new ConcurrentQueue<JObject>();
    private TransportWorker transport;
    private int tick, nextObservationTick = 1, outstandingObservationTick = -1, outstandingSentTick = -1, heldActionStartedTick = -1, activeActionStartedTick;
    private string outstandingObservationId, activeActionId, unitySessionId, serverSessionId;
    private bool receivedInitialWorld, activeActionHadTransportGap;
    private float accumulator, rateWindowStarted;
    private int framesAtRateWindow, stepsAtRateWindow;
    private long lastFrameTimestamp, lastFixedTimestamp, lastObservationTimestamp, lastResponseTimestamp;
    private static long TimestampNow => Stopwatch.GetTimestamp();
    private static float AgeSince(long timestamp) => timestamp <= 0 ? float.PositiveInfinity : (float)((TimestampNow - timestamp) / (double)Stopwatch.Frequency);

    private void Awake() { if (!enabled) return; foreach (PythonBridge other in FindObjectsByType<PythonBridge>(FindObjectsInactive.Include)) if (other != this) { enabled = false; Destroy(this); return; } Physics2D.simulationMode = SimulationMode2D.Script; unitySessionId = Guid.NewGuid().ToString("N"); }
    private void Start() { transport = new TransportWorker(host, port, incoming, outgoing, transportEvents); transport.Start(); PipelineStatus = "Asynchronous transport starting"; }
    private void Update()
    {
        FrameCount++; lastFrameTimestamp = TimestampNow; UpdateRates(); DrainTransport();
        if (ControllerFaultPaused || Time.timeScale <= 0f || !receivedInitialWorld) return;
        accumulator += Time.unscaledDeltaTime;
        for (int steps = 0; accumulator >= FixedStepSeconds && steps < Mathf.Max(1, maximumCatchUpSteps); steps++) { accumulator -= FixedStepSeconds; RunFixedSimulationStep(); DrainTransport(); if (ControllerFaultPaused) break; }
    }
    private void UpdateRates() { float now = Time.unscaledTime; if (rateWindowStarted <= 0f) { rateWindowStarted = now; framesAtRateWindow = FrameCount; stepsAtRateWindow = PhysicsStepCount; return; } float elapsed = now - rateWindowStarted; if (elapsed < .5f) return; RenderRateHz = (FrameCount - framesAtRateWindow) / elapsed; PhysicsRateHz = (PhysicsStepCount - stepsAtRateWindow) / elapsed; rateWindowStarted = now; framesAtRateWindow = FrameCount; stepsAtRateWindow = PhysicsStepCount; }
    public void RegisterCreature(CreatureBrain brain) { if (brain != null && !creatures.Contains(brain)) creatures.Add(brain); }
    public void UnregisterCreature(CreatureBrain brain) { creatures.Remove(brain); }

    private void RunFixedSimulationStep()
    {
        if (outstandingObservationId != null && tick - outstandingSentTick >= MaxActionAgeTicks) BeginRecovery("Python action exceeded the 8-tick deadline");
        if (heldActionStartedTick >= 0 && tick - heldActionStartedTick >= HeldActionLimitTicks) { ControllerFaultPaused = true; PipelineStatus = "Controller fault: held action exceeded one second"; return; }
        if (outstandingObservationId == null && tick >= nextObservationTick) QueueObservation();
        long started = TimestampNow; lastFixedTimestamp = started; PhysicsStepCount++; tick++;
        int oldest = 0;
        foreach (CreatureBrain brain in creatures.ToArray()) { if (brain == null) { creatures.Remove(brain); continue; } brain.FixedStep(); oldest = Mathf.Max(oldest, brain.ActionAge(tick)); }
        OldestActionAge = oldest; Physics2D.Simulate(FixedStepSeconds); LastPhysicsStepMilliseconds = (float)((TimestampNow - started) * 1000d / Stopwatch.Frequency); CollectDeaths();
    }
    private void QueueObservation() { string id = Guid.NewGuid().ToString("N"); outstandingObservationId = id; outstandingObservationTick = tick; outstandingSentTick = tick; outgoing.Enqueue(BuildObservation(id)); lastObservationTimestamp = TimestampNow; PipelineStatus = "Asynchronous: sent observation " + tick; }
    private JObject BuildObservation(string observationId)
    {
        JObject states = new JObject(); List<CreatureIdentity> population = new List<CreatureIdentity>();
        foreach (CreatureBrain brain in creatures) if (brain != null && !brain.IsDead && brain.torso != null && brain.torso.bodyPart != null) population.Add(brain.torso.bodyPart.identity);
        foreach (CreatureBrain brain in creatures) if (brain != null && !brain.IsDead && brain.torso != null && brain.torso.bodyPart != null) states[brain.torso.bodyPart.identity.creatureId] = brain.ToObservation(population);
        JArray events = new JArray(); foreach (JObject value in lifecycleEvents.Values) events.Add(value.DeepClone());
        JArray segments = new JArray(); foreach (JObject value in reportedSegments.Values) segments.Add(value.DeepClone());
        return new JObject { ["protocol_version"] = ProtocolVersion, ["session_id"] = unitySessionId, ["observation_id"] = observationId, ["tick_id"] = tick, ["physics_tick"] = tick, ["simulated_seconds"] = tick * FixedStepSeconds, ["fixed_delta_time"] = FixedStepSeconds, ["control_ticks"] = NominalControlTicks, ["previous_action_id"] = activeActionId ?? "", ["creatures"] = states, ["lifecycle_events"] = events, ["executed_segments"] = segments };
    }
    private void DrainTransport()
    {
        while (transportEvents.TryDequeue(out string status)) { ConnectionStatus = status.StartsWith("Connected") ? "Connected" : status.StartsWith("Connecting") ? "Connecting" : "Disconnected"; if (!status.StartsWith("Connected") && !status.StartsWith("Connecting")) LastServerError = status; }
        while (incoming.TryDequeue(out JObject response)) ApplyResponse(response);
    }
    private void ApplyResponse(JObject response)
    {
        if ((response.Value<int?>("protocol_version") ?? 0) != ProtocolVersion) { BeginRecovery("Python protocol mismatch"); return; }
        string incomingSession = response.Value<string>("session_id") ?? "";
        if (!string.IsNullOrEmpty(serverSessionId) && !string.IsNullOrEmpty(incomingSession) && incomingSession != serverSessionId) ResetPhysicalWorldForServerSession();
        if (!string.IsNullOrEmpty(incomingSession)) serverSessionId = incomingSession;
        receivedInitialWorld = true; ConnectionStatus = response.Value<string>("server_status") ?? "Connected"; lastResponseTimestamp = TimestampNow; CheckpointStatus = response.Value<string>("checkpoint_status") ?? CheckpointStatus; UpdateTelemetry(response["telemetry"] as JObject);
        if (response["acknowledged_event_ids"] is JArray acknowledged) foreach (JToken token in acknowledged) lifecycleEvents.Remove(token.Value<string>());
        if (response["acknowledged_segment_ids"] is JArray segmentAcks) foreach (JToken token in segmentAcks) reportedSegments.Remove(token.Value<string>());
        if (response["commands"] is JArray commands) foreach (JToken token in commands) if (token is JObject command) ApplyCommand(command);
        string observationId = response.Value<string>("source_observation_id") ?? ""; int sourceTick = response.Value<int?>("source_tick") ?? -1;
        if (sourceTick < 0 || observationId.Length == 0) {
            // A successful reconnect only restores the transport.  While the
            // controlled world is fault-paused, immediately request a fresh
            // action from its frozen state; only that action may resume it.
            if (ControllerFaultPaused && outstandingObservationId == null) QueueObservation();
            return;
        }
        LastResponseSourceTick = sourceTick;
        if (observationId != outstandingObservationId || tick - sourceTick > MaxActionAgeTicks) { BeginRecovery("Discarded stale or unmatched Python response"); return; }
        LastRoundTripMilliseconds = (float)((TimestampNow - lastObservationTimestamp) * 1000d / Stopwatch.Frequency); outstandingObservationId = null; outstandingObservationTick = -1; nextObservationTick = tick + NominalControlTicks; activeActionHadTransportGap |= heldActionStartedTick >= 0; heldActionStartedTick = -1; ControllerFaultPaused = false; ApplyActions(response, sourceTick);
    }
    private void UpdateTelemetry(JObject telemetry) { if (telemetry == null) return; SpeciesCount = telemetry.Value<int?>("species") ?? SpeciesCount; LearnerLag = telemetry.Value<int?>("learner_lag") ?? LearnerLag; M3Error = telemetry.Value<float?>("m3_error") ?? M3Error; LearnerUpdates = telemetry.Value<int?>("learner_updates") ?? LearnerUpdates; DroppedLearning = telemetry.Value<int?>("dropped_learning") ?? DroppedLearning; ReplaySize = telemetry.Value<int?>("replay_size") ?? ReplaySize; MotionReplaySize = telemetry.Value<int?>("motion_replay_size") ?? MotionReplaySize; LearnerBusy = telemetry.Value<bool?>("learner_busy") ?? LearnerBusy; ServerObserveMilliseconds = telemetry.Value<float?>("observe_ms") ?? ServerObserveMilliseconds; ServerInferenceMilliseconds = telemetry.Value<float?>("inference_ms") ?? ServerInferenceMilliseconds; LearnerUpdateMilliseconds = telemetry.Value<float?>("learner_update_ms") ?? LearnerUpdateMilliseconds; ObserveMetricsTick = telemetry.Value<int?>("observe_tick") ?? ObserveMetricsTick; InferenceMetricsTick = telemetry.Value<int?>("inference_tick") ?? InferenceMetricsTick; LearnerMetricsTick = telemetry.Value<int?>("learner_tick") ?? LearnerMetricsTick; }
    private void ApplyActions(JObject response, int sourceTick)
    {
        JObject actions = response["actions"] as JObject; if (actions == null) return;
        if (!string.IsNullOrEmpty(activeActionId)) RecordCompletedActionSegment("replaced");
        string actionId = response.Value<string>("action_id") ?? Guid.NewGuid().ToString("N"); int applied = 0;
        foreach (CreatureBrain brain in creatures) { if (brain == null || brain.torso == null || brain.torso.bodyPart == null) continue; string id = brain.torso.bodyPart.identity.creatureId; if (!(actions[id] is JObject action) || !(action["deltas"] is JObject deltas)) continue; JArray goal = action["goal"] as JArray; Vector2 target = goal != null && goal.Count >= 2 ? new Vector2(goal[0].Value<float>(), goal[1].Value<float>()) : Vector2.zero; if (brain.ApplyActions(deltas, target, NominalControlTicks, sourceTick, tick)) applied++; if (action["telemetry"] is JObject creatureTelemetry) brain.UpdateTelemetry(creatureTelemetry); }
        activeActionId = actionId; activeActionStartedTick = tick; LastActionsApplied = applied; PipelineStatus = "Asynchronous: applied learned action " + actionId.Substring(0, Math.Min(6, actionId.Length));
    }
    private void BeginRecovery(string reason) { if (heldActionStartedTick < 0) heldActionStartedTick = tick; LastServerError = reason; PipelineStatus = "Controller recovering: holding last learned action"; outstandingObservationId = null; outstandingObservationTick = -1; nextObservationTick = tick; transport?.ResetConnection(); }
    private void RecordCompletedActionSegment(string reason) { if (string.IsNullOrEmpty(activeActionId) || activeActionStartedTick >= tick) return; string id = Guid.NewGuid().ToString("N"); reportedSegments[id] = new JObject { ["segment_id"] = id, ["action_id"] = activeActionId, ["start_tick"] = activeActionStartedTick, ["end_tick"] = tick, ["reason"] = reason, ["transport_gap"] = activeActionHadTransportGap }; activeActionHadTransportGap = false; }
    private void ApplyCommand(JObject command)
    {
        string commandId = command.Value<string>("command_id"); if (string.IsNullOrWhiteSpace(commandId) || handledCommands.Contains(commandId)) return; string type = command.Value<string>("type"); bool success = false; string error = "";
        try { if (type == "spawn" && command["creature"] is JObject creature) SpawnTransactional(creature); else if (type == "despawn") Despawn(command.Value<string>("creature_id")); else throw new InvalidOperationException("unknown command type"); success = true; } catch (Exception exception) { error = exception.Message; LastServerError = error; UnityEngine.Debug.LogError("Lifecycle command failed: " + exception); }
        if (handledCommandOrder.Count >= 4096) handledCommands.Remove(handledCommandOrder.Dequeue()); handledCommands.Add(commandId); handledCommandOrder.Enqueue(commandId); AddLifecycleEvent(new JObject { ["type"] = type == "spawn" ? "spawn_result" : "despawn_result", ["command_id"] = commandId, ["success"] = success, ["error"] = error, ["creature_id"] = type == "spawn" ? command["creature"]?.Value<string>("creature_id") : command.Value<string>("creature_id") });
    }
    private void SpawnTransactional(JObject command)
    {
        string id = command.Value<string>("creature_id"), species = command.Value<string>("species_id"); if (string.IsNullOrWhiteSpace(id) || string.IsNullOrWhiteSpace(species)) throw new ArgumentException("spawn identity is incomplete"); foreach (CreatureBrain existing in creatures) if (existing != null && existing.torso.bodyPart.identity.creatureId == id) return; if (!(command["body"] is JObject body)) throw new ArgumentException("spawn body is missing"); BodyGenomeDto genome = BodyGenomeDto.FromJson(body); genome.Validate(); int slot = AllocateSpawnSlot(id); Vector2 position = WorldLayout.CreatureSpawnPosition(slot); GameObject root = new GameObject("Creature_" + id.Substring(0, Math.Min(8, id.Length))); root.SetActive(false); root.transform.position = position;
        try { CreatureIdentity identity = root.AddComponent<CreatureIdentity>(); identity.creatureId = id; identity.speciesId = species; GameObject torsoObject = new GameObject("Torso"); torsoObject.transform.SetParent(root.transform); torsoObject.transform.position = position; Torso torso = torsoObject.AddComponent<Torso>(); torso.InitFromGenome(identity, genome); identity.torso = torso; if (torso.childLimbs.Count != genome.limbs.FindAll(limb => limb.enabled).Count || torso.childLimbs.Count == 0) throw new InvalidOperationException("phenotype actuator topology is incomplete"); CreatureBrain brain = torsoObject.AddComponent<CreatureBrain>(); brain.Init(torso, torso.GetAllLimbs()); root.SetActive(true); RegisterCreature(brain); } catch { spawnSlots.Remove(id); if (Application.isPlaying) Destroy(root); else DestroyImmediate(root); throw; }
    }
    private void CollectDeaths() { foreach (CreatureBrain brain in creatures.ToArray()) { if (brain == null || !brain.IsDead) continue; Dictionary<string, float> counters = brain.torso.GetGlobalInputs(); AddLifecycleEvent(new JObject { ["type"] = "death", ["creature_id"] = brain.torso.bodyPart.identity.creatureId, ["reason"] = brain.DeathReason, ["age_seconds"] = Time.time - brain.BornTime, ["tick"] = tick, ["energy"] = brain.torso.energy, ["food_gain"] = counters["food_gain"], ["baseline_cost"] = counters["baseline_cost"], ["movement_cost"] = counters["movement_cost"], ["idle_cost"] = counters["idle_cost"] }); spawnSlots.Remove(brain.torso.bodyPart.identity.creatureId); UnregisterCreature(brain); Destroy(brain.transform.parent.gameObject); nextObservationTick = tick; } }
    private void AddLifecycleEvent(JObject value) { string id = Guid.NewGuid().ToString("N"); value["event_id"] = id; lifecycleEvents[id] = value; }
    private void Despawn(string creatureId) { foreach (CreatureBrain brain in creatures.ToArray()) if (brain != null && brain.torso != null && brain.torso.bodyPart.identity.creatureId == creatureId) { brain.Kill("evolution"); spawnSlots.Remove(creatureId); UnregisterCreature(brain); Destroy(brain.transform.parent.gameObject); return; } spawnSlots.Remove(creatureId); }
    private void ResetPhysicalWorldForServerSession() { foreach (CreatureBrain brain in creatures.ToArray()) if (brain != null) { brain.Kill("server_recovery"); Destroy(brain.transform.parent.gameObject); } creatures.Clear(); spawnSlots.Clear(); lifecycleEvents.Clear(); handledCommands.Clear(); handledCommandOrder.Clear(); reportedSegments.Clear(); }
    private int AllocateSpawnSlot(string creatureId) { if (spawnSlots.TryGetValue(creatureId, out int existing)) return existing; HashSet<int> occupied = new HashSet<int>(spawnSlots.Values); int slot = 0; while (occupied.Contains(slot)) slot++; spawnSlots[creatureId] = slot; return slot; }
    public void RequestOwnedServerShutdown() { transport?.Shutdown(); }
    private void OnDestroy() { transport?.Dispose(); }

    private sealed class TransportWorker : IDisposable
    {
        private readonly string host; private readonly int port; private readonly ConcurrentQueue<JObject> incoming; private readonly ConcurrentQueue<JObject> outgoing; private readonly ConcurrentQueue<string> status; private readonly AutoResetEvent signal = new AutoResetEvent(false); private readonly CancellationTokenSource cancellation = new CancellationTokenSource(); private Thread thread; private TcpClient client; private NetworkStream stream; private int generation;
        public TransportWorker(string host, int port, ConcurrentQueue<JObject> incoming, ConcurrentQueue<JObject> outgoing, ConcurrentQueue<string> status) { this.host = host; this.port = port; this.incoming = incoming; this.outgoing = outgoing; this.status = status; }
        public void Start() { thread = new Thread(Run) { IsBackground = true, Name = "LiveEcosystemTransport" }; thread.Start(); }
        public void ResetConnection() { Interlocked.Increment(ref generation); Close(); signal.Set(); }
        public void Shutdown() { cancellation.Cancel(); Close(); signal.Set(); }
        private void Run() { while (!cancellation.IsCancellationRequested) { try { status.Enqueue("Connecting"); client = new TcpClient(); client.Connect(host, port); stream = client.GetStream(); incoming.Enqueue(ReadMessage(stream)); status.Enqueue("Connected"); int localGeneration = generation; while (!cancellation.IsCancellationRequested && localGeneration == generation) { if (!outgoing.TryDequeue(out JObject request)) { signal.WaitOne(10); continue; } WriteMessage(stream, request); JObject response = ReadMessage(stream); if (localGeneration == generation) incoming.Enqueue(response); } } catch (Exception exception) when (!cancellation.IsCancellationRequested) { status.Enqueue("Disconnected: " + exception.Message); Thread.Sleep(50); } finally { Close(); } } }
        private void Close() { try { stream?.Dispose(); } catch { } try { client?.Close(); } catch { } stream = null; client = null; }
        private static void WriteMessage(NetworkStream stream, JObject value) { byte[] body = Encoding.UTF8.GetBytes(value.ToString(Formatting.None)); if (body.Length > MaxMessageBytes) throw new InvalidOperationException("message too large"); byte[] header = BitConverter.GetBytes(body.Length); if (BitConverter.IsLittleEndian) Array.Reverse(header); stream.Write(header, 0, 4); stream.Write(body, 0, body.Length); stream.Flush(); }
        private static JObject ReadMessage(NetworkStream stream) { byte[] header = ReadExact(stream, 4); if (BitConverter.IsLittleEndian) Array.Reverse(header); int length = BitConverter.ToInt32(header, 0); if (length <= 0 || length > MaxMessageBytes) throw new InvalidOperationException("invalid message size"); return JObject.Parse(Encoding.UTF8.GetString(ReadExact(stream, length))); }
        private static byte[] ReadExact(NetworkStream stream, int count) { byte[] data = new byte[count]; int offset = 0; while (offset < count) { int read = stream.Read(data, offset, count - offset); if (read <= 0) throw new InvalidOperationException("socket closed"); offset += read; } return data; }
        public void Dispose() { Shutdown(); if (thread != null && thread.IsAlive) thread.Join(1000); signal.Dispose(); cancellation.Dispose(); }
    }
}
