using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

public static class BodyMorphologyRules
{
    public const int MaxLimbs = 6;
    public const int MaxDepth = 3;
    public const float TorsoWidth = 1.5f;
    public const float TorsoHeight = 1.5f;
    public const float LimbWidth = 1.25f;
    public const float LimbHeight = .4f;
    public const float JointTorque = 65f;
    public const float JointMinimum = -75f;
    public const float JointMaximum = 75f;
    public const float Tolerance = 1e-4f;

    public static readonly int[] TorsoSlots = { 0, 1, 4, 5, 6, 7 };
    public static readonly int[] LimbSlots = { 2, 3 };

    public static bool IsTorsoSlot(int slot) => Array.IndexOf(TorsoSlots, slot) >= 0;
    public static bool IsLimbSlot(int slot) => Array.IndexOf(LimbSlots, slot) >= 0;
    public static bool Approximately(float value, float expected) => Mathf.Abs(value - expected) <= Tolerance;

    public static Vector2 LocalAttachmentPoint(int parentInnovationId, int slot)
    {
        if (parentInnovationId == 0)
        {
            switch (slot)
            {
                case 0: return new Vector2(0f, .5f);
                case 1: return new Vector2(0f, -.5f);
                case 4: return new Vector2(.5f, .5f);
                case 5: return new Vector2(.5f, -.5f);
                case 6: return new Vector2(-.5f, .5f);
                case 7: return new Vector2(-.5f, -.5f);
                default: throw new ArgumentException("invalid torso attachment slot");
            }
        }
        if (slot == 2) return new Vector2(-.5f, 0f);
        if (slot == 3) return new Vector2(.5f, 0f);
        throw new ArgumentException("invalid limb attachment slot");
    }
}

[Serializable]
public class LimbGeneDto
{
    public int innovation_id;
    public int parent_innovation_id;
    public int attachment_slot;
    public float width = BodyMorphologyRules.LimbWidth;
    public float height = BodyMorphologyRules.LimbHeight;
    public float mass = 1f;
    public float inertia = .2f;
    public string joint_type = "hinge";
    public float min_angle = BodyMorphologyRules.JointMinimum;
    public float max_angle = BodyMorphologyRules.JointMaximum;
    public float max_torque = BodyMorphologyRules.JointTorque;
    public bool enabled = true;

    public static LimbGeneDto FromJson(JObject value) => new LimbGeneDto {
        innovation_id = value.Value<int>("innovation_id"), parent_innovation_id = value.Value<int>("parent_innovation_id"),
        attachment_slot = value.Value<int>("attachment_slot"), width = value.Value<float?>("width") ?? BodyMorphologyRules.LimbWidth,
        height = value.Value<float?>("height") ?? BodyMorphologyRules.LimbHeight, mass = value.Value<float?>("mass") ?? 1f,
        inertia = value.Value<float?>("inertia") ?? .2f, joint_type = value.Value<string>("joint_type") ?? "hinge",
        min_angle = value.Value<float?>("min_angle") ?? BodyMorphologyRules.JointMinimum, max_angle = value.Value<float?>("max_angle") ?? BodyMorphologyRules.JointMaximum,
        max_torque = value.Value<float?>("max_torque") ?? BodyMorphologyRules.JointTorque, enabled = value.Value<bool?>("enabled") ?? true
    };
}

[Serializable]
public class BodyGenomeDto
{
    public string genome_id;
    public float torso_width = BodyMorphologyRules.TorsoWidth;
    public float torso_height = BodyMorphologyRules.TorsoHeight;
    public float torso_mass = 2f;
    public float torso_inertia = .5f;
    public List<LimbGeneDto> limbs = new List<LimbGeneDto>();

    public static BodyGenomeDto FromJson(JObject value) {
        BodyGenomeDto result = new BodyGenomeDto {
            genome_id = value.Value<string>("genome_id"), torso_width = value.Value<float?>("torso_width") ?? BodyMorphologyRules.TorsoWidth,
            torso_height = value.Value<float?>("torso_height") ?? BodyMorphologyRules.TorsoHeight,
            torso_mass = value.Value<float?>("torso_mass") ?? 2f,
            torso_inertia = value.Value<float?>("torso_inertia") ?? .5f
        };
        if (value["limbs"] is JArray limbs)
            foreach (JToken token in limbs) if (token is JObject limb) result.limbs.Add(LimbGeneDto.FromJson(limb));
        return result;
    }

    public void Validate()
    {
        if (string.IsNullOrWhiteSpace(genome_id)) throw new ArgumentException("body genome_id is missing");
        if (!BodyMorphologyRules.Approximately(torso_width, BodyMorphologyRules.TorsoWidth) || !BodyMorphologyRules.Approximately(torso_height, BodyMorphologyRules.TorsoHeight)) throw new ArgumentException("noncanonical torso dimensions");
        HashSet<int> ids = new HashSet<int> { 0 };
        HashSet<string> slots = new HashSet<string>();
        int enabledCount = 0;
        foreach (LimbGeneDto limb in limbs) {
            if (!limb.enabled) continue;
            enabledCount++;
            if (limb.innovation_id <= 0 || !ids.Add(limb.innovation_id)) throw new ArgumentException("duplicate limb innovation");
            if (limb.joint_type != "hinge") throw new ArgumentException("unsupported joint type");
            if (!BodyMorphologyRules.Approximately(limb.width, BodyMorphologyRules.LimbWidth) || !BodyMorphologyRules.Approximately(limb.height, BodyMorphologyRules.LimbHeight)) throw new ArgumentException("noncanonical limb dimensions");
            if (!BodyMorphologyRules.Approximately(limb.max_torque, BodyMorphologyRules.JointTorque)) throw new ArgumentException("noncanonical joint torque");
            if (!BodyMorphologyRules.Approximately(limb.min_angle, BodyMorphologyRules.JointMinimum) || !BodyMorphologyRules.Approximately(limb.max_angle, BodyMorphologyRules.JointMaximum)) throw new ArgumentException("noncanonical joint limits");
            string key = limb.parent_innovation_id + ":" + limb.attachment_slot;
            if (!slots.Add(key)) throw new ArgumentException("occupied attachment slot");
        }
        if (enabledCount > BodyMorphologyRules.MaxLimbs) throw new ArgumentException("too many enabled limbs");
        HashSet<int> reachable = new HashSet<int> { 0 };
        bool changed = true;
        while (changed) {
            changed = false;
            foreach (LimbGeneDto limb in limbs)
                if (limb.enabled && reachable.Contains(limb.parent_innovation_id) && reachable.Add(limb.innovation_id)) changed = true;
        }
        Dictionary<int, LimbGeneDto> byId = new Dictionary<int, LimbGeneDto>();
        foreach (LimbGeneDto limb in limbs)
        {
            if (!limb.enabled) continue;
            if (!reachable.Contains(limb.innovation_id)) throw new ArgumentException("orphan or cyclic limb structure");
            byId[limb.innovation_id] = limb;
            bool validSlot = limb.parent_innovation_id == 0 ? BodyMorphologyRules.IsTorsoSlot(limb.attachment_slot) : BodyMorphologyRules.IsLimbSlot(limb.attachment_slot);
            if (!validSlot) throw new ArgumentException("attachment slot is invalid for its parent");
        }
        foreach (LimbGeneDto limb in byId.Values)
        {
            int depth = 1;
            int parentId = limb.parent_innovation_id;
            while (parentId != 0)
            {
                depth++;
                if (depth > BodyMorphologyRules.MaxDepth) throw new ArgumentException("limb path exceeds maximum depth");
                parentId = byId[parentId].parent_innovation_id;
            }
        }
    }
}
