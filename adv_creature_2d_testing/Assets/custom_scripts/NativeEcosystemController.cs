using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using UnityEngine;

[DefaultExecutionOrder(-80)]
[DisallowMultipleComponent]
public sealed class NativeEcosystemController : MonoBehaviour
{
    private const float MinimumSpawnSeparation = 20f;
    public int populationTarget = 4;
    public int controlIntervalTicks = 2;
    public float evolutionIntervalSeconds = 10f;
    public float checkpointIntervalSeconds = 30f;
    public bool loadCheckpointOnStart = true;
    public string checkpointFileName = "native_ecosystem_v1.json";
    public string Status { get; private set; } = "Starting native ecosystem";
    public int Tick { get; private set; }
    public int Generation { get; private set; }
    public int Population => agents.Count;
    public int SpeciesCount => agents.Select(a => a.identity.speciesId).Distinct().Count();
    public float LastM3Loss { get; private set; }
    public float LastM2Loss { get; private set; }
    public float LastActionMagnitude { get; private set; }
    public float MeanM1EnergyFitness => agents.Count == 0 ? 0f : agents.Average(agent => agent.m1EnergyFitness);
    public int DynamicsReplaySamples => agents.Sum(agent => agent.model.DynamicsReplayCount);
    public float RenderRateHz { get; private set; }
    public string PhysicsMode => Physics2D.simulationMode.ToString();
    public int SnapshotVersion { get; private set; }
    public string CheckpointStatus { get; private set; } = "Not saved";
    public string LastCheckpointError { get; private set; } = "";
    public float SecondsUntilCheckpoint => Mathf.Max(0f, nextCheckpoint - Time.time);
    public string LastCheckpointSavedAt { get; private set; } = "Never";
    public int ActiveControlCount => agents.Count(agent => agent.brain != null && !agent.brain.IsDead && agent.brain.LastAppliedTick >= 0);
    public int MovingCreatureCount => agents.Count(agent => agent.identity != null && agent.identity.torso != null && !agent.brain.IsDead && agent.identity.torso.Rigidbody != null && agent.identity.torso.Rigidbody.linearVelocity.magnitude >= .05f);
    public float MeanRootSpeed => agents.Where(agent => agent.identity != null && agent.identity.torso != null && agent.identity.torso.Rigidbody != null && !agent.brain.IsDead).Select(agent => agent.identity.torso.Rigidbody.linearVelocity.magnitude).DefaultIfEmpty(0f).Average();
    public float MeanJointSpeed => agents.Where(agent => agent.brain != null && !agent.brain.IsDead).Select(agent => agent.brain.MeanJointSpeed).DefaultIfEmpty(0f).Average();
    public int ControlFailures { get; private set; }
    public string LastControlError { get; private set; } = "";

    private readonly List<NativeAgent> agents = new List<NativeAgent>();
    private List<NativeBodySpecies> species = new List<NativeBodySpecies>();
    private NativeBrainWeights sharedWeights;
    private System.Random random;
    private float nextEvolution;
    private float nextCheckpoint;
    private string checkpointPath;
    private int framesAtRateWindow;
    private float rateWindowStarted;
    private int controlStep;

    private sealed class NativeAgent
    {
        public CreatureIdentity identity;
        public CreatureBrain brain;
        public BodyGenomeDto body;
        public NativeCreatureModel model;
        public float deathPenalty;
        public Vector2 previousPosition;
        public float m1EnergyFitness;
        public float previousEnergy;
        public bool deathRecorded;
    }

    private void Awake()
    {
        Application.runInBackground = true;
        Physics2D.simulationMode = SimulationMode2D.FixedUpdate;
        checkpointPath = Path.Combine(Application.persistentDataPath, checkpointFileName);
        random = new System.Random(1337);
        sharedWeights = NativeBrainWeights.Create(1337);
        if (loadCheckpointOnStart) TryLoad();
    }

    private void Start()
    {
        Physics2D.simulationMode = SimulationMode2D.FixedUpdate;
        if (agents.Count == 0) SeedPopulation();
        RebuildSpeciation();
        nextEvolution = Time.time + evolutionIntervalSeconds;
        nextCheckpoint = Time.time + checkpointIntervalSeconds;
        Status = "Native recurrent ecosystem running";
    }

    private void Update()
    {
        if (rateWindowStarted <= 0f) { rateWindowStarted = Time.unscaledTime; framesAtRateWindow = Time.frameCount; return; }
        float elapsed = Time.unscaledTime - rateWindowStarted;
        if (elapsed >= .5f) { RenderRateHz = (Time.frameCount - framesAtRateWindow) / elapsed; framesAtRateWindow = Time.frameCount; rateWindowStarted = Time.unscaledTime; }
    }

    private void FixedUpdate()
    {
        if (Physics2D.simulationMode != SimulationMode2D.FixedUpdate) Physics2D.simulationMode = SimulationMode2D.FixedUpdate;
        Tick++;
        if (Tick % Mathf.Max(1, controlIntervalTicks) == 0)
        {
            controlStep++;
            foreach (NativeAgent agent in agents.ToArray())
                try { StepAgent(agent, controlStep); }
                catch (Exception exception)
                {
                    ControlFailures++;
                    LastControlError = agent.identity == null ? exception.Message : agent.identity.creatureId + ": " + exception.Message;
                    Debug.LogError("Native control failed for creature " + LastControlError + "\n" + exception);
                    // Disable only the invalid phenotype. Other creatures must
                    // continue receiving their fixed-step controls.
                    if (agent.brain != null && !agent.brain.IsDead) agent.brain.Kill("native_control_error");
                }
        }
        foreach (NativeAgent agent in agents.ToArray()) if (agent.brain != null) agent.brain.FixedStep();
        if (Time.time >= nextEvolution) { nextEvolution += evolutionIntervalSeconds; Evolve(); }
    }

    private void LateUpdate()
    {
        foreach (NativeAgent agent in agents.ToArray())
        {
            if (agent.identity == null || agent.identity.torso == null) continue;
            Vector2 current = agent.identity.torso.transform.position;
            Vector2 displacement = current - agent.previousPosition;
            if (agent.model.TrainDynamics(displacement))
            {
                sharedWeights.CopyM3From(agent.model.Weights);
                foreach (NativeAgent other in agents) if (other != agent) other.model.CopyM3From(sharedWeights);
                SnapshotVersion++;
            }
            agent.model.TrainPolicyMinibatch(8);
            LastM3Loss = agent.model.LastM3Loss;
            float energy = agent.identity.torso.energy;
            agent.m1EnergyFitness += energy - agent.previousEnergy;
            agent.previousEnergy = energy;
            agent.previousPosition = current;
            if (agent.brain.IsDead && !agent.deathRecorded) { float penalty = agent.brain.DeathReason == "predator" ? 100f : 25f; agent.deathPenalty += penalty; agent.m1EnergyFitness -= penalty; agent.deathRecorded = true; }
        }
        if (Time.time >= nextCheckpoint) { nextCheckpoint += checkpointIntervalSeconds; SaveCheckpoint(); }
    }

    private void StepAgent(NativeAgent agent, int currentControlStep)
    {
        if (agent.brain == null || agent.brain.IsDead || agent.identity.torso == null) return;
        if (!agent.model.Weights.IsFinite())
        {
            ControlFailures++;
            LastControlError = agent.identity.creatureId + ": non-finite model state repaired";
            Debug.LogError("Repairing non-finite native model state for creature " + agent.identity.creatureId);
            agent.model.RepairFrom(sharedWeights);
        }
        Dictionary<string, float> raw = agent.identity.torso.GetGlobalInputs();
        float[] global = new float[28];
        global[0] = raw["energy"] / 200f; global[1] = raw["position_x"] / 50f; global[2] = raw["position_y"] / 50f;
        global[3] = raw["velocity_x"] / 20f; global[4] = raw["velocity_y"] / 20f; global[5] = raw["angular_velocity"] / 360f;
        global[6] = raw["rotation_sin"]; global[7] = raw["rotation_cos"]; global[8] = raw["sees_food"]; global[9] = raw["rel_food_x"] / 20f; global[10] = raw["rel_food_y"] / 20f;
        global[11] = raw["sees_predator"]; global[12] = raw["rel_predator_x"] / 20f; global[13] = raw["rel_predator_y"] / 20f;
        Limb[] limbs = agent.brain.allLimbs.Where(limb => limb != null).Take(NativeCreatureModel.MaxLimbs).ToArray();
        float[] actions = agent.model.Infer(global, limbs, agent.brain.CurrentGoal, currentControlStep);
        if (actions.Any(value => float.IsNaN(value) || float.IsInfinity(value)))
        {
            ControlFailures++;
            LastControlError = agent.identity.creatureId + ": non-finite action repaired";
            Debug.LogError("Repairing non-finite native action for creature " + agent.identity.creatureId);
            agent.model.RepairFrom(sharedWeights);
            actions = agent.model.Infer(global, limbs, agent.brain.CurrentGoal, currentControlStep);
        }
        LastM2Loss = agent.model.LastM2Loss;
        LastActionMagnitude = actions.Take(limbs.Length).Select(Mathf.Abs).DefaultIfEmpty(0f).Average();
        agent.brain.ApplyNativeActions(actions, agent.model.CurrentGoal, controlIntervalTicks, Tick);
    }

    private void SeedPopulation()
    {
        for (int i = 0; i < Mathf.Clamp(populationTarget, 1, 10); i++)
        {
            BodyGenomeDto body = CreateSeedBody(i);
            Spawn(body, "body_seed_" + (i % 3), "m1_seed_" + (i % 3), sharedWeights.Clone(), i);
        }
    }

    private BodyGenomeDto CreateSeedBody(int index)
    {
        BodyGenomeDto body = new BodyGenomeDto { genome_id = Guid.NewGuid().ToString("N"), torso_width = 1.5f, torso_height = 1.5f, limbs = new List<LimbGeneDto>() };
        // Four root limbs keep the silhouettes diverse, then two descendants
        // ensure every initial creature starts with an actual limb hierarchy.
        for (int i = 0; i < 4; i++) body.limbs.Add(new LimbGeneDto { innovation_id = i + 1, parent_innovation_id = 0, attachment_slot = (index + i) % 8, width = 1.5f, height = .45f, max_torque = 65f });
        body.limbs.Add(new LimbGeneDto { innovation_id = 5, parent_innovation_id = 1, attachment_slot = 0, width = 1.25f, height = .4f, max_torque = 55f });
        body.limbs.Add(new LimbGeneDto { innovation_id = 6, parent_innovation_id = 5, attachment_slot = 4, width = 1.1f, height = .35f, max_torque = 45f });
        return body;
    }

    private void Spawn(BodyGenomeDto body, string species, string m1Species, NativeBrainWeights weights, int slot)
    {
        body.Validate(); Vector2 position = FindOpenSpawnPosition(slot);
        GameObject root = new GameObject("NativeCreature_" + body.genome_id.Substring(0, 8)); root.transform.position = position;
        CreatureIdentity identity = root.AddComponent<CreatureIdentity>(); identity.creatureId = Guid.NewGuid().ToString("N"); identity.speciesId = species;
        GameObject torsoObject = new GameObject("Torso"); torsoObject.transform.SetParent(root.transform); torsoObject.transform.position = position;
        torsoObject.AddComponent<Rigidbody2D>(); torsoObject.AddComponent<SpriteRenderer>(); torsoObject.AddComponent<BoxCollider2D>(); Torso torso = torsoObject.AddComponent<Torso>(); torso.InitFromGenome(identity, body); identity.torso = torso; identity.bodyGenomeJson = JsonConvert.SerializeObject(body);
        CreatureBrain brain = torsoObject.AddComponent<CreatureBrain>(); brain.Init(torso, torso.GetAllLimbs()); brain.SetNativeMetadata(species, m1Species);
        NativeAgent agent = new NativeAgent { identity = identity, brain = brain, body = body, model = new NativeCreatureModel(weights, 2003 + slot), previousPosition = position, previousEnergy = torso.energy }; agents.Add(agent);
    }

    private Vector2 FindOpenSpawnPosition(int preferredSlot)
    {
        int count = WorldLayout.MaximumNativeSpawnSlots;
        float requiredSqr = MinimumSpawnSeparation * MinimumSpawnSeparation;
        for (int offset = 0; offset < count; offset++)
        {
            Vector2 candidate = WorldLayout.CreatureSpawnPosition((preferredSlot + offset) % count);
            bool occupied = agents.Any(agent => agent.identity != null && agent.identity.torso != null && ((Vector2)agent.identity.torso.transform.position - candidate).sqrMagnitude < requiredSqr);
            if (!occupied) return candidate;
        }
        // A moving population can temporarily occupy every reserved slot. Keep
        // the existing slot geometry as the first choice, then search outward
        // without ever relaxing the established separation threshold.
        Vector2 origin = WorldLayout.CreatureSpawnPosition(preferredSlot % count);
        for (int ring = 1; ring <= 12; ring++)
            for (int dx = -ring; dx <= ring; dx++)
                for (int dy = -ring; dy <= ring; dy++)
                {
                    if (Mathf.Abs(dx) != ring && Mathf.Abs(dy) != ring) continue;
                    Vector2 candidate = origin + new Vector2(dx, dy) * MinimumSpawnSeparation;
                    bool occupied = agents.Any(agent => agent.identity != null && agent.identity.torso != null && ((Vector2)agent.identity.torso.transform.position - candidate).sqrMagnitude < requiredSqr);
                    if (!occupied) return candidate;
                }
        throw new InvalidOperationException("No non-overlapping native creature spawn position is available");
    }

    private void Evolve()
    {
        if (agents.Count < 2) return;
        Dictionary<string, float> sharedFitness = NativeNestedSpeciation.SharedFitness(species);
        NativeAgent parent = agents.Where(a => !a.brain.IsDead).OrderBy(a => sharedFitness.TryGetValue(a.identity.creatureId, out float fitness) ? fitness : float.PositiveInfinity).FirstOrDefault();
        NativeAgent victim = agents.Where(a => a != parent).OrderByDescending(a => a.model.BodyFitness).FirstOrDefault();
        if (parent == null || victim == null) return;
        NativeAgent m1Parent = agents.Where(a => !a.brain.IsDead && a.identity.speciesId == parent.identity.speciesId).OrderByDescending(a => a.m1EnergyFitness).FirstOrDefault() ?? parent;
        int slot = Array.IndexOf(agents.ToArray(), victim);
        GameObject victimRoot = victim.identity.transform.parent != null ? victim.identity.transform.parent.gameObject : victim.identity.gameObject;
        // Destroy is deferred by Unity; deactivate the old phenotype first so
        // its renderer and colliders cannot overlap the replacement this frame.
        if (victimRoot != null) victimRoot.SetActive(false);
        Destroy(victimRoot); agents.Remove(victim);
        BodyGenomeDto child = JsonConvert.DeserializeObject<BodyGenomeDto>(JsonConvert.SerializeObject(parent.body));
        MutateBody(child);
        NativeBrainWeights childWeights = parent.model.Weights.Clone();
        Array.Copy(m1Parent.model.Weights.m1Input, childWeights.m1Input, childWeights.m1Input.Length); Array.Copy(m1Parent.model.Weights.m1Hidden, childWeights.m1Hidden, childWeights.m1Hidden.Length); Array.Copy(m1Parent.model.Weights.m1Output, childWeights.m1Output, childWeights.m1Output.Length);
        childWeights.Mutate(random, .12f);
        childWeights.CopyM3From(sharedWeights);
        Spawn(child, parent.identity.speciesId, parent.brain.M1SpeciesId, childWeights, Mathf.Max(0, slot)); Generation++; RebuildSpeciation(); SaveCheckpoint();
    }

    private void MutateBody(BodyGenomeDto body)
    {
        body.genome_id = Guid.NewGuid().ToString("N");
        body.torso_width = Mathf.Clamp(body.torso_width + RandomRange(-.45f, .45f), .7f, 2.6f);
        body.torso_height = Mathf.Clamp(body.torso_height + RandomRange(-.35f, .35f), .7f, 2.4f);
        foreach (LimbGeneDto limb in body.limbs.Where(limb => limb.enabled))
        {
            limb.width = Mathf.Clamp(limb.width + RandomRange(-.5f, .5f), .35f, 3.2f);
            limb.height = Mathf.Clamp(limb.height + RandomRange(-.18f, .18f), .18f, 1.1f);
            limb.max_torque = Mathf.Clamp(limb.max_torque + RandomRange(-20f, 20f), 20f, 160f);
            float range = Mathf.Clamp((limb.max_angle - limb.min_angle) + RandomRange(-30f, 30f), 45f, 170f);
            float center = Mathf.Clamp((limb.min_angle + limb.max_angle) * .5f + RandomRange(-20f, 20f), -60f, 60f);
            limb.min_angle = center - range * .5f; limb.max_angle = center + range * .5f;
        }

        List<LimbGeneDto> enabled = body.limbs.Where(limb => limb.enabled).ToList();
        if (enabled.Count < NativeCreatureModel.MaxLimbs && random.NextDouble() < .65)
        {
            LimbGeneDto parent = enabled[random.Next(enabled.Count)];
            HashSet<int> usedSlots = new HashSet<int>(body.limbs.Where(limb => limb.enabled && limb.parent_innovation_id == parent.innovation_id).Select(limb => limb.attachment_slot));
            int[] available = Enumerable.Range(0, 8).Where(slot => !usedSlots.Contains(slot)).ToArray();
            if (available.Length > 0 && GeneDepth(body, parent.innovation_id) < NativeCreatureModel.MaxPathDepth)
                body.limbs.Add(new LimbGeneDto { innovation_id = body.limbs.Max(limb => limb.innovation_id) + 1, parent_innovation_id = parent.innovation_id, attachment_slot = available[random.Next(available.Length)], width = 1.1f, height = .35f, max_torque = 55f });
        }
        else if (enabled.Count > 4 && random.NextDouble() < .35)
        {
            LimbGeneDto removable = enabled.Where(limb => !body.limbs.Any(other => other.enabled && other.parent_innovation_id == limb.innovation_id)).OrderBy(_ => random.Next()).FirstOrDefault();
            if (removable != null) removable.enabled = false;
        }
        body.Validate();
    }

    private float RandomRange(float minimum, float maximum) => minimum + (float)random.NextDouble() * (maximum - minimum);
    private static int GeneDepth(BodyGenomeDto body, int innovationId)
    {
        int depth = 0;
        while (innovationId != 0) { LimbGeneDto gene = body.limbs.FirstOrDefault(limb => limb.enabled && limb.innovation_id == innovationId); if (gene == null) return NativeCreatureModel.MaxPathDepth; innovationId = gene.parent_innovation_id; depth++; }
        return depth;
    }

    private void RebuildSpeciation()
    {
        List<NativeCreatureControllerView> views = agents.Where(a => a.identity != null && a.brain != null).Select(a => new NativeCreatureControllerView(a.identity.creatureId, a.identity.speciesId, a.brain.M1SpeciesId, a.body, a.model.Weights, a.model.BodyFitness)).ToList();
        species = NativeNestedSpeciation.Rebuild(views, species);
        Dictionary<string, NativeAgent> byId = agents.Where(agent => agent.identity != null).ToDictionary(agent => agent.identity.creatureId);
        foreach (NativeBodySpecies body in species)
            foreach (NativeBrainSubSpecies sub in body.subSpecies)
                foreach (NativeCreatureControllerView member in sub.members)
                    if (byId.TryGetValue(member.id, out NativeAgent agent)) { agent.identity.speciesId = body.id; agent.brain.SetNativeMetadata(body.id, sub.id); }
    }

    public void SaveCheckpoint()
    {
        try
        {
            NativeEcosystemCheckpoint checkpoint = new NativeEcosystemCheckpoint { tick = Tick, generation = Generation, controlStep = controlStep, randomState = random.Next(), nextEvolutionTicks = (long)(nextEvolution * 1000f), sharedWeights = sharedWeights };
            foreach (NativeAgent agent in agents) checkpoint.creatures.Add(new NativeCreatureCheckpoint { creatureId = agent.identity.creatureId, speciesId = agent.identity.speciesId, m1SpeciesId = agent.brain.M1SpeciesId, bodyGenomeJson = JsonConvert.SerializeObject(agent.body), brain = agent.model.Capture(), bodyFitness = float.IsInfinity(agent.model.BodyFitness) ? float.MaxValue : agent.model.BodyFitness, deathPenalty = agent.deathPenalty, m1EnergyFitness = agent.m1EnergyFitness, previousEnergy = agent.previousEnergy, deathRecorded = agent.deathRecorded });
            string temp = checkpointPath + ".tmp"; string backup = checkpointPath + ".bak"; Directory.CreateDirectory(Path.GetDirectoryName(checkpointPath));
            string serialized = JsonConvert.SerializeObject(checkpoint, Formatting.None);
            using (FileStream stream = new FileStream(temp, FileMode.Create, FileAccess.Write, FileShare.None)) using (StreamWriter writer = new StreamWriter(stream)) { writer.Write(serialized); writer.Flush(); stream.Flush(true); }
            NativeEcosystemCheckpoint validation = JsonConvert.DeserializeObject<NativeEcosystemCheckpoint>(File.ReadAllText(temp));
            if (validation == null || validation.schema != 1 || validation.configuration != checkpoint.configuration) throw new InvalidDataException("Temporary native checkpoint failed validation");
            if (File.Exists(checkpointPath)) File.Replace(temp, checkpointPath, backup, true); else File.Move(temp, checkpointPath);
            CheckpointStatus = "Saved"; LastCheckpointError = ""; LastCheckpointSavedAt = DateTime.Now.ToString("HH:mm:ss");
        }
        catch (Exception exception) { CheckpointStatus = "Save failed"; LastCheckpointError = exception.Message; Debug.LogError("Native checkpoint save failed: " + exception); }
    }

    private void TryLoad()
    {
        if (!File.Exists(checkpointPath)) return;
        try
        {
            NativeEcosystemCheckpoint checkpoint = JsonConvert.DeserializeObject<NativeEcosystemCheckpoint>(File.ReadAllText(checkpointPath));
            if (checkpoint == null || checkpoint.schema != 1 || checkpoint.configuration != NativeEcosystemCheckpoint.ConfigurationFingerprint) throw new InvalidDataException("Native checkpoint schema/configuration mismatch");
            sharedWeights = checkpoint.sharedWeights ?? sharedWeights; Tick = checkpoint.tick; Generation = checkpoint.generation; controlStep = checkpoint.controlStep;
            foreach (NativeCreatureCheckpoint saved in checkpoint.creatures.Take(Mathf.Clamp(populationTarget, 1, 10))) { BodyGenomeDto body = JsonConvert.DeserializeObject<BodyGenomeDto>(saved.bodyGenomeJson); Spawn(body, saved.speciesId, saved.m1SpeciesId, saved.brain?.weights ?? sharedWeights.Clone(), agents.Count); NativeAgent agent = agents[agents.Count - 1]; agent.model.Restore(saved.brain); agent.identity.creatureId = saved.creatureId; agent.deathPenalty = saved.deathPenalty; agent.m1EnergyFitness = saved.m1EnergyFitness; agent.previousEnergy = saved.previousEnergy; agent.deathRecorded = saved.deathRecorded; }
            LastCheckpointSavedAt = "Loaded"; CheckpointStatus = "Loaded";
        }
        catch (Exception exception) { LastCheckpointError = exception.Message; CheckpointStatus = "Rejected; fresh population"; QuarantineCheckpointArtifacts("schema-or-corrupt"); }
    }

    private void QuarantineCheckpointArtifacts(string reason)
    {
        string[] artifacts = { checkpointPath, checkpointPath + ".bak", checkpointPath + ".tmp" };
        foreach (string artifact in artifacts)
            try { if (File.Exists(artifact)) File.Move(artifact, artifact + "." + reason + "." + DateTime.UtcNow.Ticks); } catch (Exception exception) { LastCheckpointError += " | quarantine: " + exception.Message; }
    }

    private void OnApplicationQuit() => SaveCheckpoint();
    private void OnDestroy() { if (Application.isPlaying) SaveCheckpoint(); }
}
