using System;
using System.Collections.Generic;
using System.Linq;

public sealed class NativeBrainSubSpecies
{
    public string id;
    public NativeCreatureControllerView representative;
    public readonly List<NativeCreatureControllerView> members = new List<NativeCreatureControllerView>();
}

public sealed class NativeBodySpecies
{
    public string id;
    public NativeCreatureControllerView representative;
    public readonly List<NativeBrainSubSpecies> subSpecies = new List<NativeBrainSubSpecies>();
}

public readonly struct NativeCreatureControllerView
{
    public readonly string id;
    public readonly string bodySpecies;
    public readonly string brainSpecies;
    public readonly BodyGenomeDto body;
    public readonly NativeBrainWeights weights;
    public readonly float fitness;
    public NativeCreatureControllerView(string id, string bodySpecies, string brainSpecies, BodyGenomeDto body, NativeBrainWeights weights, float fitness) { this.id = id; this.bodySpecies = bodySpecies; this.brainSpecies = brainSpecies; this.body = body; this.weights = weights; this.fitness = fitness; }
}

public static class NativeNestedSpeciation
{
    public const float BodyThreshold = .25f, BrainThreshold = .45f;
    public static List<NativeBodySpecies> Rebuild(IReadOnlyList<NativeCreatureControllerView> population, List<NativeBodySpecies> existing = null)
    {
        List<NativeBodySpecies> result = existing ?? new List<NativeBodySpecies>(); foreach (NativeBodySpecies species in result) { foreach (NativeBrainSubSpecies sub in species.subSpecies) sub.members.Clear(); species.subSpecies.Clear(); }
        foreach (NativeCreatureControllerView creature in population)
        {
            NativeBodySpecies body = result.FirstOrDefault(s => BodyDistance(creature.body, s.representative.body) <= BodyThreshold);
            if (body == null) { body = new NativeBodySpecies { id = string.IsNullOrEmpty(creature.bodySpecies) ? "body_" + creature.id : creature.bodySpecies, representative = creature }; result.Add(body); }
            NativeBrainSubSpecies sub = body.subSpecies.FirstOrDefault(s => BrainDistance(creature.weights, s.representative.weights) <= BrainThreshold);
            if (sub == null) { sub = new NativeBrainSubSpecies { id = string.IsNullOrEmpty(creature.brainSpecies) ? "m1_" + creature.id : creature.brainSpecies, representative = creature }; body.subSpecies.Add(sub); }
            sub.members.Add(creature);
        }
        return result.Where(s => s.subSpecies.Any()).ToList();
    }
    public static Dictionary<string, float> SharedFitness(IReadOnlyList<NativeBodySpecies> species)
    {
        Dictionary<string, float> result = new Dictionary<string, float>(); foreach (NativeBodySpecies body in species) { int total = body.subSpecies.Sum(s => s.members.Count); foreach (NativeBrainSubSpecies sub in body.subSpecies) foreach (NativeCreatureControllerView member in sub.members) result[member.id] = member.fitness / Math.Max(1, sub.members.Count) / Math.Max(1, total); } return result;
    }
    private static float BodyDistance(BodyGenomeDto a, BodyGenomeDto b) { if (a == null || b == null) return float.PositiveInfinity; float value = Math.Abs(a.torso_width - b.torso_width) + Math.Abs(a.torso_height - b.torso_height) + Math.Abs(a.limbs.Count - b.limbs.Count); return value / 3f; }
    private static float BrainDistance(NativeBrainWeights a, NativeBrainWeights b) { if (a == null || b == null) return float.PositiveInfinity; float total = 0f; int count = 0; foreach (var pair in a.Arrays().Zip(b.Arrays(), (x, y) => new { x, y })) for (int i = 0; i < Math.Min(pair.x.Length, pair.y.Length); i++) { total += Math.Abs(pair.x[i] - pair.y[i]); count++; } return count == 0 ? 0f : total / count; }
}
