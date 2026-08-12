using UnityEngine;
using Newtonsoft.Json.Linq;
using System.Collections.Generic;

public class CreatureTelemetryUI : MonoBehaviour
{
    private CreatureBrain brain;
    private Torso torso;
    private GUIStyle boxStyle;
    private GUIStyle headerStyle;
    private GUIStyle labelStyle;
    private GUIStyle valueStyle;

    private Vector2 goalVector = Vector2.zero;
    private Vector2 predDisplacement = Vector2.zero;
    private Vector2 realDisplacement = Vector2.zero;

    private void Start()
    {
        brain = GetComponent<CreatureBrain>();
        if (brain != null) torso = brain.torso;
    }

    private void InitStyles()
    {
        if (boxStyle == null)
        {
            boxStyle = new GUIStyle(GUI.skin.box);
            boxStyle.normal.background = MakeTexture(2, 2, new Color(0.05f, 0.05f, 0.1f, 0.85f));
            boxStyle.padding = new RectOffset(12, 12, 12, 12);

            headerStyle = new GUIStyle(GUI.skin.label);
            headerStyle.fontSize = 14;
            headerStyle.fontStyle = FontStyle.Bold;
            headerStyle.normal.textColor = new Color(0.3f, 0.9f, 1.0f);

            labelStyle = new GUIStyle(GUI.skin.label);
            labelStyle.fontSize = 11;
            labelStyle.fontStyle = FontStyle.Bold;
            labelStyle.normal.textColor = Color.yellow;

            valueStyle = new GUIStyle(GUI.skin.label);
            valueStyle.fontSize = 11;
            valueStyle.normal.textColor = Color.white;
        }
    }

    private Texture2D MakeTexture(int width, int height, Color col)
    {
        Color[] pix = new Color[width * height];
        for (int i = 0; i < pix.Length; i++) pix[i] = col;
        Texture2D result = new Texture2D(width, height);
        result.SetPixels(pix);
        result.Apply();
        return result;
    }

    private void OnGUI()
    {
        if (brain == null || brain.latestTelemetry == null) return;
        InitStyles();

        JObject telem = brain.latestTelemetry;

        // Overlay Window Position
        Rect hudRect = new Rect(15, 15, 420, 480);
        GUILayout.BeginArea(hudRect, boxStyle);

        GUILayout.Label("🧠 CREATURE BRAIN TELEMETRY", headerStyle);
        GUILayout.Space(4);

        // Active Phase
        string phase = telem.Value<string>("active_phase") ?? "M1 -> M2 -> M3";
        GUILayout.Label($"Active Phase: {phase}", labelStyle);
        GUILayout.Space(6);

        // 1. M1 Goal Setter
        GUILayout.Label("1. M1 GOAL-SETTER (Strategic Layer)", labelStyle);
        string goalSource = telem.Value<string>("goal_source") ?? "Forward";
        JArray relGoal = telem["relative_goal"] as JArray;
        if (relGoal != null && relGoal.Count >= 2)
        {
            goalVector = new Vector2((float)relGoal[0], (float)relGoal[1]);
            GUILayout.Label($"   • Goal Source : {goalSource}", valueStyle);
            GUILayout.Label($"   • Relative Goal Vector g_t : ({goalVector.x:F3}, {goalVector.y:F3}, 0.000)", valueStyle);
        }
        GUILayout.Space(6);

        // 2. M2 Action Selector
        GUILayout.Label("2. M2 ACTION-SELECTOR (Achiever Transformer)", labelStyle);
        JArray jSensors = telem["m2_joint_sensors"] as JArray;
        JArray outDeltas = telem["m2_output_deltas"] as JArray;
        if (jSensors != null && outDeltas != null)
        {
            GUILayout.Label($"   • Input 4D Tokens [Angle, TouchS, TouchO, TouchE]:", valueStyle);
            foreach (JToken token in jSensors)
            {
                if (token is JArray tArr && tArr.Count >= 4)
                {
                    GUILayout.Label($"      - Angle: {tArr[0]}° | Touch [Self: {tArr[1]}, Other: {tArr[2]}, Env: {tArr[3]}]", valueStyle);
                }
            }
            GUILayout.Label($"   • Output Joint Actions a_t: [{string.Join(", ", outDeltas)}]", valueStyle);
        }
        GUILayout.Space(6);

        // 3. M3 Dynamics Predictor
        GUILayout.Label("3. M3 DYNAMICS PREDICTOR (Achiever Transformer)", labelStyle);
        JArray predDisp = telem["m3_pred_displacement"] as JArray;
        float goalLoss = telem.Value<float>("m3_goal_loss");
        if (predDisp != null && predDisp.Count >= 2)
        {
            predDisplacement = new Vector2((float)predDisp[0], (float)predDisp[1]);
            GUILayout.Label($"   • Predicted Displacement Δp_pred: ({predDisplacement.x:F4}, {predDisplacement.y:F4})", valueStyle);
            GUILayout.Label($"   • M2/M3 Goal Reach MSE Loss    : {goalLoss:F5}", valueStyle);
        }
        GUILayout.Space(6);

        // 4. M3 Online Dynamics Training
        GUILayout.Label("4. M3 ONLINE DYNAMICS TRAINING (Supervised SGD)", labelStyle);
        JArray realDisp = telem["m3_real_displacement"] as JArray;
        float dynLoss = telem.Value<float>("m3_dynamics_loss");
        int bufferSize = telem.Value<int>("replay_buffer_size");
        if (realDisp != null && realDisp.Count >= 2)
        {
            realDisplacement = new Vector2((float)realDisp[0], (float)realDisp[1]);
            GUILayout.Label($"   • Real Physics Displacement Δp_real: ({realDisplacement.x:F4}, {realDisplacement.y:F4})", valueStyle);
            GUILayout.Label($"   • M3 Dynamics Fitting MSE Loss    : {dynLoss:F5}", valueStyle);
            GUILayout.Label($"   • Replay Buffer Capacity          : {bufferSize} / 5000 samples", valueStyle);
        }

        GUILayout.EndArea();
    }

    private void OnDrawGizmos()
    {
        if (torso == null) return;

        Vector3 pos = torso.transform.position;

        // Green line: Body-Relative Goal Vector g_t
        Gizmos.color = Color.green;
        Gizmos.DrawRay(pos, (Vector3)goalVector);
        Gizmos.DrawWireSphere(pos + (Vector3)goalVector, 0.2f);

        // Cyan line: Predicted Displacement
        Gizmos.color = Color.cyan;
        Gizmos.DrawRay(pos, (Vector3)predDisplacement * 5f); // scaled up for visibility

        // Red line: Real Displacement
        Gizmos.color = Color.red;
        Gizmos.DrawRay(pos, (Vector3)realDisplacement * 5f);
    }
}
