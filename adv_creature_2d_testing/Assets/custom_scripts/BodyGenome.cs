using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

[Serializable]
public class LimbGeneDto
{
    public int innovation_id;
    public int parent_innovation_id;
    public int attachment_slot;
    public float width = 1.5f;
    public float height = 0.45f;
    public float mass = 1f;
    public float inertia = .2f;
    public string joint_type = "hinge";
    public float min_angle = -90f;
    public float max_angle = 90f;
    public float max_torque = 50f;
    public bool enabled = true;

    public static LimbGeneDto FromJson(JObject value) => new LimbGeneDto {
        innovation_id = value.Value<int>("innovation_id"), parent_innovation_id = value.Value<int>("parent_innovation_id"),
        attachment_slot = value.Value<int>("attachment_slot"), width = value.Value<float?>("width") ?? 1.5f,
        height = value.Value<float?>("height") ?? .45f, mass = value.Value<float?>("mass") ?? 1f,
        inertia = value.Value<float?>("inertia") ?? .2f, joint_type = value.Value<string>("joint_type") ?? "hinge",
        min_angle = value.Value<float?>("min_angle") ?? -90f, max_angle = value.Value<float?>("max_angle") ?? 90f,
        max_torque = value.Value<float?>("max_torque") ?? 50f, enabled = value.Value<bool?>("enabled") ?? true
    };
}

[Serializable]
public class BodyGenomeDto
{
    public string genome_id;
    public float torso_width = 1.5f;
    public float torso_height = 1.5f;
    public float torso_mass = 2f;
    public float torso_inertia = .5f;
    public List<LimbGeneDto> limbs = new List<LimbGeneDto>();

    public static BodyGenomeDto FromJson(JObject value) {
        BodyGenomeDto result = new BodyGenomeDto {
            genome_id = value.Value<string>("genome_id"), torso_width = value.Value<float?>("torso_width") ?? 1.5f,
            torso_height = value.Value<float?>("torso_height") ?? 1.5f,
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
        if (torso_width < .1f || torso_height < .1f) throw new ArgumentException("invalid torso dimensions");
        HashSet<int> ids = new HashSet<int> { 0 };
        HashSet<string> slots = new HashSet<string>();
        foreach (LimbGeneDto limb in limbs) {
            if (!limb.enabled) continue;
            if (limb.innovation_id <= 0 || !ids.Add(limb.innovation_id)) throw new ArgumentException("duplicate limb innovation");
            if (limb.attachment_slot < 0 || limb.attachment_slot >= 8) throw new ArgumentException("invalid attachment slot");
            if (limb.joint_type != "hinge") throw new ArgumentException("unsupported joint type");
            string key = limb.parent_innovation_id + ":" + limb.attachment_slot;
            if (!slots.Add(key)) throw new ArgumentException("occupied attachment slot");
        }
        HashSet<int> reachable = new HashSet<int> { 0 };
        bool changed = true;
        while (changed) {
            changed = false;
            foreach (LimbGeneDto limb in limbs)
                if (limb.enabled && reachable.Contains(limb.parent_innovation_id) && reachable.Add(limb.innovation_id)) changed = true;
        }
        foreach (LimbGeneDto limb in limbs)
            if (limb.enabled && !reachable.Contains(limb.innovation_id)) throw new ArgumentException("orphan or cyclic limb structure");
    }
}
