using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

public class CreatureBrain : MonoBehaviour
{
    public Torso torso;
    public List<Limb> allLimbs = new List<Limb>();
    public JObject LatestTelemetry { get; private set; }
    public bool IsDead { get; private set; }
    // The native ecosystem owns whether its current training phase permits
    // lifecycle turnover.  Keeping this on the brain also blocks direct death
    // calls from energy, predators, and fault handling while the phase is off.
    public bool DeathsEnabled { get; set; } = true;
    public string DeathReason { get; private set; }
    public float BornTime { get; private set; }
    public int LastAppliedTick { get; private set; } = -1;
    public int LastActionSourceTick { get; private set; } = -1;
    public float LastActionMagnitude { get; private set; }
    public string ControlMode => LatestTelemetry == null ? "waiting" : LatestTelemetry.Value<string>("control_mode") ?? "waiting";
    public Vector2 CurrentGoal { get; private set; }
    public Vector2 CurrentWorldGoal { get; private set; }
    public bool HasWorldGoal { get; private set; }
    public string M1SpeciesId { get; private set; } = "m1_unknown";
    public float BodyFitness { get; private set; } = float.PositiveInfinity;

    public void Init(Torso torsoRef, List<Limb> limbs) { torso = torsoRef; allLimbs = limbs; BornTime = Time.time; }
    public void SetNativeMetadata(string species, string m1Species) { M1SpeciesId = string.IsNullOrEmpty(m1Species) ? "m1_unknown" : m1Species; }
    public void ApplyNativeActions(float[] deltas, Vector2 goal, int controlTicks, int appliedTick) {
        if (IsDead) return; float largest = 0f; for (int i = 0; i < allLimbs.Count; i++) { float delta = deltas != null && i < deltas.Length ? deltas[i] : 0f; largest = Mathf.Max(largest, Mathf.Abs(delta)); allLimbs[i].ApplyIntervalDelta(delta, controlTicks, Time.fixedDeltaTime); } CurrentGoal = Vector2.ClampMagnitude(goal, NativeCreatureModel.GoalMagnitude); LastActionMagnitude = largest; LastAppliedTick = appliedTick; LastActionSourceTick = appliedTick;
    }
    public void ApplyNativeActions(float[] deltas, Vector2 localGoal, Vector2 worldGoal, int controlTicks, int appliedTick) { ApplyNativeActions(deltas, localGoal, controlTicks, appliedTick); CurrentWorldGoal = worldGoal; HasWorldGoal = true; }
    private void OnDestroy()
    {
        // Keep the bridge registry correct even when a phenotype is removed by
        // a scene reload or a transactional spawn rollback rather than by the
        // normal death/despawn path.
        PythonBridge bridge = FindAnyObjectByType<PythonBridge>();
        if (bridge != null) bridge.UnregisterCreature(this);
    }
    public void UpdateTelemetry(JObject value) { LatestTelemetry = value; }
    public int ActionAge(int currentTick) => LastAppliedTick < 0 ? currentTick : Mathf.Max(0, currentTick - LastAppliedTick);
    public float MeanJointSpeed {
        get {
            float total = 0f; int count = 0;
            foreach (Limb limb in allLimbs) if (limb != null) { total += Mathf.Abs(limb.JointSpeed); count++; }
            return count == 0 ? 0f : total / count;
        }
    }
    public int JointLimitCount {
        get { int count = 0; foreach (Limb limb in allLimbs) if (limb != null && limb.AtJointLimit) count++; return count; }
    }
    public bool ApplyActions(JObject deltas, Vector2 goal, int controlTicks, int sourceTick, int appliedTick) {
        if (IsDead || sourceTick <= LastActionSourceTick) return false;
        float largestDelta = 0f;
        foreach (Limb limb in allLimbs) {
            JToken token = deltas[limb.innovationId.ToString()];
            // Treat an omitted limb as an explicit zero command. This keeps a
            // malformed/partial response from leaving an older motor command
            // running until its previous interval expires.
            float delta = token == null ? 0f : token.Value<float>();
            largestDelta = Mathf.Max(largestDelta, Mathf.Abs(delta));
            limb.ApplyIntervalDelta(delta, controlTicks, Time.fixedDeltaTime);
        }
        CurrentGoal = Vector2.ClampMagnitude(goal, NativeCreatureModel.GoalMagnitude);
        LastActionSourceTick = sourceTick;
        LastAppliedTick = appliedTick;
        LastActionMagnitude = largestDelta;
        return true;
    }
    public void ExpireActions() { foreach (Limb limb in allLimbs) if (limb != null) limb.ApplyIntervalDelta(0f, 1, Time.fixedDeltaTime); }
    public void FixedStep() {
        if (IsDead) return;
        foreach (Limb limb in allLimbs) limb.AdvanceMotorTick();
        torso.DrainEnergy(allLimbs, Time.fixedDeltaTime);
        if (torso.energy <= 0f) Kill("energy");
    }
    public void Kill(string reason) {
        if (!DeathsEnabled || IsDead) return;
        IsDead = true; DeathReason = reason;
        CreatureIdentity identity = torso == null || torso.bodyPart == null ? null : torso.bodyPart.identity;
        if (identity != null)
            foreach (Predator predator in FindObjectsByType<Predator>(FindObjectsInactive.Include)) predator.Forget(identity);
        foreach (Limb limb in allLimbs) if (limb != null) limb.DisablePhenotype();
        if (torso != null) torso.DisablePhenotype();
    }
    public JObject ToObservation(IReadOnlyList<CreatureIdentity> population = null) {
        JObject locals = new JObject(); JArray limbs = new JArray();
        foreach (Limb limb in allLimbs) {
            Dictionary<string,float> input = limb.GetLocalInputs();
            JArray path = new JArray(); foreach (int segment in limb.genePath) path.Add(segment);
            JObject item = new JObject {
                ["innovation_id"] = limb.innovationId, ["gene_path"] = path, ["depth"] = limb.depth,
                ["joint_type_id"] = 0, ["previous_action"] = limb.ExecutedIntervalDelta,
                ["width"] = limb.dimensions.x, ["height"] = limb.dimensions.y, ["max_torque"] = limb.maxMotorTorque,
                ["joint_angle"] = input["joint_angle"], ["angular_velocity"] = input["angular_velocity"],
                ["min_angle"] = limb.MinAngle, ["max_angle"] = limb.MaxAngle,
                ["at_limit"] = limb.AtJointLimit ? 1 : 0,
                ["touch_self"] = input["touch_self"], ["touch_other_creature"] = input["touch_other_creature"],
                ["touch_environment"] = input["touch_environment"]
            };
            limbs.Add(item); locals[limb.innovationId.ToString()] = item;
        }
        JObject global = new JObject(); foreach (KeyValuePair<string,float> pair in torso.GetGlobalInputs()) global[pair.Key] = pair.Value;
        List<JObject> neighbours = new List<JObject>();
        if (population == null) population = FindObjectsByType<CreatureIdentity>(FindObjectsInactive.Exclude);
        foreach (CreatureIdentity other in population) {
            if (other == torso.bodyPart.identity || other.torso == null) continue;
            Vector2 relative = other.torso.transform.position - torso.transform.position;
            neighbours.Add(new JObject { ["x"] = relative.x, ["y"] = relative.y,
                                         ["same"] = other.speciesId == torso.bodyPart.identity.speciesId ? 1 : 0,
                                         ["valid"] = 1,
                                         ["distance"] = relative.sqrMagnitude });
        }
        neighbours.Sort((a, b) => a.Value<float>("distance").CompareTo(b.Value<float>("distance")));
        JArray nearest = new JArray(); for (int i = 0; i < Mathf.Min(3, neighbours.Count); i++) nearest.Add(neighbours[i]);
        global["neighbours"] = nearest;
        return new JObject { ["global_inputs"] = global, ["limbs"] = limbs,
                             ["current_goal"] = new JArray(CurrentGoal.x, CurrentGoal.y) };
    }
}
