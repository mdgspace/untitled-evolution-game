using System.Collections.Generic;
using UnityEngine;
using Newtonsoft.Json.Linq;

public class CreatureBrain : MonoBehaviour
{
    public Torso torso;
    public List<Limb> allLimbs;
    public JObject latestTelemetry;

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

        if (GetComponent<CreatureTelemetryUI>() == null)
            gameObject.AddComponent<CreatureTelemetryUI>();

        PythonBridge bridge = FindObjectOfType<PythonBridge>();
        if (bridge != null)
            bridge.RegisterCreature(this);
        else
            Debug.LogWarning("CreatureBrain: no PythonBridge in the scene -- this creature won't be driven by Python.");
    }

    public void UpdateTelemetry(JObject telem)
    {
        latestTelemetry = telem;
    }

    private void OnDestroy()
    {
        PythonBridge bridge = FindObjectOfType<PythonBridge>();
        if (bridge != null) bridge.UnregisterCreature(this);
    }

    public Dictionary<string, float> GetGlobalInputs() => torso.GetGlobalInputs();
    public Dictionary<string, float> GetLocalInputs(Limb limb) => limb.GetLocalInputs();

    public Dictionary<Limb, Dictionary<string, float>> GetAllLocalInputs()
    {
        Dictionary<Limb, Dictionary<string, float>> result = new Dictionary<Limb, Dictionary<string, float>>();
        foreach (Limb limb in allLimbs)
            result[limb] = limb.GetLocalInputs();
        return result;
    }

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

    // NEW -- called by PythonBridge after this tick's deltas are applied,
    // before physics steps. Order matters: energy drain needs this tick's
    // lastAppliedDelta already set on every limb, and dopamine needs that
    // tick's energy cost already computed.
    public void UpdateEnergyAndDopamine()
    {
        torso.DrainEnergy(allLimbs);
        torso.UpdateDopamine();
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

    public void SetJointDeltaAngle(Limb limb, float deltaDegrees) => limb.ApplyDeltaAngle(deltaDegrees);

    public void SetAllJointDeltaAngles(Dictionary<Limb, float> deltas)
    {
        foreach (KeyValuePair<Limb, float> kv in deltas)
            kv.Key.ApplyDeltaAngle(kv.Value);
    }

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