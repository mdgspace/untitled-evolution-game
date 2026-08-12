using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using Newtonsoft.Json.Linq;

// Central authority for the whole scene's simulation stepping. Every tick:
// gather every registered creature's inputs -> send ONE batched message ->
// block until Python responds -> apply the returned deltas -> only THEN
// manually advance physics by exactly one step. Nothing in the scene
// simulates without Python's explicit go-ahead.
//
// Exactly one of these should exist per scene.
public class PythonBridge : MonoBehaviour
{
    [Header("Connection")]
    public string host = "127.0.0.1";
    public int port = 9999;

    private TcpClient client;
    private NetworkStream stream;
    private readonly List<CreatureBrain> registeredCreatures = new List<CreatureBrain>();

    private void Awake()
    {
        // REQUIRED, not optional: with the default Auto mode, Unity
        // simulates physics automatically during its own FixedUpdate
        // phase, BEFORE this script's FixedUpdate runs -- meaning
        // physics would already have advanced before Python's inputs are
        // even gathered. Script mode makes Physics2D.Simulate() (called
        // explicitly below) the ONLY thing that ever advances physics.
        Physics2D.simulationMode = SimulationMode2D.Script;
    }

    private void Start()
    {
        try
        {
            client = new TcpClient();
            client.Connect(host, port);
            stream = client.GetStream();
            Debug.Log($"PythonBridge: connected to {host}:{port}");
        }
        catch (Exception e)
        {
            Debug.LogError($"PythonBridge: failed to connect to {host}:{port} -- is bridge_server.py running? {e}");
        }
    }

    // Called by each CreatureBrain during its own Init -- keeps this
    // class from needing to scan the scene for creatures every tick.
    public void RegisterCreature(CreatureBrain brain)
    {
        if (!registeredCreatures.Contains(brain))
            registeredCreatures.Add(brain);
    }

    public void UnregisterCreature(CreatureBrain brain)
    {
        registeredCreatures.Remove(brain);
    }

  private void FixedUpdate()
    {
        if (stream == null || registeredCreatures.Count == 0)
            return;

        JObject message = BuildInputMessage();
        SendMessageBlocking(message);
        JObject response = ReceiveMessageBlocking();
        ApplyOutputs(response);

        // NEW -- after this tick's deltas are applied, before physics
        // advances: cost the movement and compute dopamine from it
        foreach (CreatureBrain brain in registeredCreatures)
            brain.UpdateEnergyAndDopamine();

        Physics2D.Simulate(Time.fixedDeltaTime);
    }

    // ---- message building ----

    private JObject BuildInputMessage()
    {
        JObject creatures = new JObject();
        foreach (CreatureBrain brain in registeredCreatures)
        {
            brain.UpdateMemory(); // exactly one memory push per creature per tick,
                                   // driven from here rather than the creature's
                                   // own timer -- see CreatureBrain's comments

            string creatureId = brain.torso.bodyPart.identity.creatureId;

            JObject globalInputs = DictToJObject(brain.GetGlobalInputs());

            JObject localInputs = new JObject();
            foreach (KeyValuePair<Limb, Dictionary<string, float>> kv in brain.GetAllLocalInputs())
                localInputs[kv.Key.limbId] = DictToJObject(kv.Value);

            JArray goalSetterInputs = new JArray();
            foreach (float f in brain.GetGoalSetterInputs())
                goalSetterInputs.Add(f);

            creatures[creatureId] = new JObject
            {
                ["global_inputs"] = globalInputs,
                ["local_inputs"] = localInputs,
                ["goal_setter_inputs"] = goalSetterInputs
            };
        }
        return new JObject { ["creatures"] = creatures };
    }

    private static JObject DictToJObject(Dictionary<string, float> dict)
    {
        JObject obj = new JObject();
        foreach (KeyValuePair<string, float> kv in dict)
            obj[kv.Key] = kv.Value;
        return obj;
    }

    private void ApplyOutputs(JObject response)
    {
        JObject creatures = (JObject)response["creatures"];
        foreach (CreatureBrain brain in registeredCreatures)
        {
            string creatureId = brain.torso.bodyPart.identity.creatureId;
            if (creatures[creatureId] == null)
                continue; // python sent nothing for this creature this tick -- skip it, don't crash

            JObject creatureData = (JObject)creatures[creatureId];
            if (creatureData["deltas"] is JObject deltasObj)
            {
                Dictionary<Limb, float> deltas = new Dictionary<Limb, float>();
                foreach (Limb limb in brain.allLimbs)
                {
                    if (deltasObj[limb.limbId] != null)
                        deltas[limb] = deltasObj[limb.limbId].Value<float>();
                }
                brain.SetAllJointDeltaAngles(deltas);
            }

            if (creatureData["telemetry"] is JObject telemObj)
            {
                brain.UpdateTelemetry(telemObj);
            }
        }
    }

    // ---- wire protocol: 4-byte big-endian length prefix + UTF8 JSON ----
    // Matches bridge_server.py's struct.pack(">I", ...) exactly -- both
    // ends must agree on endianness, or lengths decode as garbage.

    private void SendMessageBlocking(JObject message)
    {
        byte[] payload = Encoding.UTF8.GetBytes(message.ToString(Newtonsoft.Json.Formatting.None));
        byte[] header = BitConverter.GetBytes(payload.Length);
        if (BitConverter.IsLittleEndian) Array.Reverse(header);

        stream.Write(header, 0, header.Length);
        stream.Write(payload, 0, payload.Length);
    }

    private JObject ReceiveMessageBlocking()
    {
        byte[] header = ReadExact(4);
        if (BitConverter.IsLittleEndian) Array.Reverse(header);
        int length = BitConverter.ToInt32(header, 0);

        byte[] payload = ReadExact(length);
        return JObject.Parse(Encoding.UTF8.GetString(payload));
    }

    private byte[] ReadExact(int count)
    {
        byte[] buffer = new byte[count];
        int offset = 0;
        while (offset < count)
        {
            int read = stream.Read(buffer, offset, count - offset);
            if (read == 0)
                throw new Exception("PythonBridge: connection closed while reading.");
            offset += read;
        }
        return buffer;
    }

    private void OnDestroy()
    {
        stream?.Close();
        client?.Close();
    }
}