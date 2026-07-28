using System.Collections.Generic;
using UnityEngine;

// Still no real decision-making -- that now lives entirely in Python.
// This class is pure plumbing: gathers inputs, tracks goal-setter memory,
// applies whatever deltas PythonBridge hands back.
//
// No FixedUpdate here anymore. PythonBridge is the single authority
// driving every creature's tick in the same explicit order each cycle --
// letting each CreatureBrain run its own independent FixedUpdate would
// make the gather/round-trip/apply sequence race against an unspecified
// per-component execution order across creatures, which defeats the
// point of one deterministic gate.
public class CreatureBrain : MonoBehaviour
{
    public Torso torso;
    public List<Limb> allLimbs;

    [Header("Goal-setter memory")]
    public int memoryFrameCount = 5;
    private FrameHistoryBuffer positionMemory;
    private FrameHistoryBuffer visionMemory;

    [Header("Mock testing (bypasses Python entirely)")]
    public float mockAngularSpeedDegPerSec = 60f;
    public float mockFrequency = 0.5f;

    public void Init(Torso torsoRef, List<Limb> limbs)
    {
        torso = torsoRef;
        allLimbs = limbs;
        positionMemory = new FrameHistoryBuffer(memoryFrameCount, 2);
        visionMemory = new FrameHistoryBuffer(memoryFrameCount, 4);

        PythonBridge bridge = FindAnyObjectByType<PythonBridge>();
        if (bridge != null)
            bridge.RegisterCreature(this);
        else
            Debug.LogWarning("CreatureBrain: no PythonBridge in the scene -- this creature won't be driven by Python.");
    }

    private void OnDestroy()
    {
        PythonBridge bridge = FindAnyObjectByType<PythonBridge>();
        if (bridge != null) bridge.UnregisterCreature(this);
    }

    // ---- INPUTS ----
    public Dictionary<string, float> GetGlobalInputs() => torso.GetGlobalInputs();
    public Dictionary<string, float> GetLocalInputs(Limb limb) => limb.GetLocalInputs();

    public Dictionary<Limb, Dictionary<string, float>> GetAllLocalInputs()
    {
        Dictionary<Limb, Dictionary<string, float>> result = new Dictionary<Limb, Dictionary<string, float>>();
        foreach (Limb limb in allLimbs)
            result[limb] = limb.GetLocalInputs();
        return result;
    }

    // ---- MEMORY ----
    // Called explicitly by PythonBridge once per tick, before that tick's
    // message is built -- not on any timer of its own.
    public void UpdateMemory()
    {
        Dictionary<string, float> globals = torso.GetGlobalInputs();
        float[] positionFrame = { globals["position_x"], globals["position_y"] };
        float[] visionFrame = {
            globals["sees_predator"], globals["sees_food"],
            globals["sees_player"], globals["sees_same_species"]
        };
        positionMemory.PushFrame(positionFrame);
        visionMemory.PushFrame(visionFrame);
    }

    public float[] GetGoalSetterInputs()
    {
        float[] posHistory = positionMemory.GetConcatenated();
        float[] visionHistory = visionMemory.GetConcatenated();
        float[] combined = new float[posHistory.Length + visionHistory.Length];
        System.Array.Copy(posHistory, 0, combined, 0, posHistory.Length);
        System.Array.Copy(visionHistory, 0, combined, posHistory.Length, visionHistory.Length);
        return combined;
    }

    // ---- OUTPUTS ----
    public void SetJointDeltaAngle(Limb limb, float deltaDegrees) => limb.ApplyDeltaAngle(deltaDegrees);

    public void SetAllJointDeltaAngles(Dictionary<Limb, float> deltas)
    {
        foreach (KeyValuePair<Limb, float> kv in deltas)
            kv.Key.ApplyDeltaAngle(kv.Value);
    }

    // ---- MOCK OUTPUT GENERATION ----
    // Nothing calls this automatically anymore now that PythonBridge owns
    // the tick loop -- still here for isolated testing without Python
    // connected at all, call it manually if useful.
    public Dictionary<Limb, float> GenerateMockOutputs()
    {
        Dictionary<Limb, float> deltas = new Dictionary<Limb, float>();
        foreach (Limb limb in allLimbs)
        {
            float phase = limb.depth * 0.75f;
            float angularSpeed = mockAngularSpeedDegPerSec *
                                  Mathf.Sin(Time.time * mockFrequency * 2f * Mathf.PI + phase);
            deltas[limb] = angularSpeed * Time.fixedDeltaTime;
        }
        return deltas;
    }
}