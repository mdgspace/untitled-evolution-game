using System;
using System.Collections.Generic;
using UnityEngine;

// Runtime-owned materials make the learning environment explicit and
// reproducible without relying on Unity's implicit fallback material.
public static class NativePhysicsMaterials
{
    private static PhysicsMaterial2D body;
    private static PhysicsMaterial2D ground;
    // Use Unity's overloaded null comparison. Runtime-created Unity objects can
    // retain a non-null CLR reference after destruction, which ??= cannot detect.
    public static PhysicsMaterial2D Body { get { if (body == null) body = Create("NativeBodyTraction", .6f); return body; } }
    public static PhysicsMaterial2D Ground { get { if (ground == null) ground = Create("NativeGroundTraction", .8f); return ground; } }
    private static PhysicsMaterial2D Create(string name, float friction) => new PhysicsMaterial2D(name) { friction = friction, bounciness = 0f };
}

// Explicit semantic marker used by grounded sensing; object names are not a
// physics contract and can safely change without corrupting M3 labels.
public sealed class GroundSurface : MonoBehaviour { }

[RequireComponent(typeof(Rigidbody2D), typeof(SpriteRenderer), typeof(BoxCollider2D))]
public class Torso : MonoBehaviour
{
    public Vector2 dimensions;
    public float energy = 100f;
    public float maxEnergy = 200f;
    public float baselineEnergyPerSecond = .25f;
    // Pay for actual angular travel, not merely requested actions.  This makes
    // rapid oscillation expensive even when it fails to translate the torso.
    public float movementEnergyPerDegree = .01f;
    // Still much larger than the baseline cost, but long enough for a new
    // controller to receive multiple decisions and explore before starvation.
    public float idleEnergyPerSecond = .75f;
    public float idleAfterSeconds = 15f;
    // Native evolution reduces this during the fixed-goal curriculum so a
    // creature can survive several ten-second left/right trials.
    public float EnergyPenaltyScale { get; set; } = 1f;
    [Range(.1f, 1f)] public float phenotypeScale = .45f;
    public BodyPart bodyPart;
    public List<Limb> childLimbs = new List<Limb>();

    private Rigidbody2D rb;
    private float idleSeconds;
    private float foodGain;
    private float baselineCost;
    private float movementCost;
    private float idleCost;
    private readonly HashSet<Collider2D> groundContacts = new HashSet<Collider2D>();
    private readonly Collider2D[] visionHits = new Collider2D[32];
    private static readonly ContactFilter2D VisionFilter = new ContactFilter2D { useTriggers = true, useLayerMask = false, useDepth = false, useNormalAngle = false };
    public Rigidbody2D Rigidbody => rb;
    public float FoodGain => foodGain;
    public bool TouchingGround => groundContacts.Count > 0;

    public void InitFromGenome(CreatureIdentity identity, BodyGenomeDto genome)
    {
        rb = GetComponent<Rigidbody2D>(); rb.bodyType = RigidbodyType2D.Dynamic;
        rb.simulated = true; rb.gravityScale = 1f; rb.constraints = RigidbodyConstraints2D.None;
        rb.sleepMode = RigidbodySleepMode2D.StartAwake; rb.interpolation = RigidbodyInterpolation2D.Interpolate; rb.collisionDetectionMode = CollisionDetectionMode2D.Continuous;
        float areaScale = phenotypeScale * phenotypeScale; float inertiaScale = areaScale * areaScale;
        rb.mass = Mathf.Clamp(genome.torso_mass * areaScale, .05f, 20f);
        rb.inertia = Mathf.Clamp(genome.torso_inertia * inertiaScale, .0025f, 20f);
        dimensions = new Vector2(genome.torso_width, genome.torso_height) * phenotypeScale;
        SpriteRenderer sr = GetComponent<SpriteRenderer>(); sr.sprite = BodyUtils.GetSquareSprite(); sr.color = new Color(.85f,.3f,.3f);
        BoxCollider2D col = GetComponent<BoxCollider2D>(); col.size = Vector2.one; col.sharedMaterial = NativePhysicsMaterials.Body;
        transform.localScale = new Vector3(dimensions.x, dimensions.y, 1f);
        bodyPart = gameObject.AddComponent<BodyPart>(); bodyPart.identity = identity;
        Dictionary<int, Rigidbody2D> parents = new Dictionary<int, Rigidbody2D> { { 0, rb } };
        Dictionary<int, int[]> paths = new Dictionary<int, int[]> { { 0, new int[0] } };
        List<LimbGeneDto> pending = new List<LimbGeneDto>(genome.limbs);
        while (pending.Count > 0) {
            bool built = false;
            for (int i = pending.Count - 1; i >= 0; i--) {
                LimbGeneDto gene = pending[i];
                if (!gene.enabled || !parents.ContainsKey(gene.parent_innovation_id)) { if (!gene.enabled) pending.RemoveAt(i); continue; }
                Rigidbody2D parent = parents[gene.parent_innovation_id];
                Vector2 localAttachment = BodyMorphologyRules.LocalAttachmentPoint(gene.parent_innovation_id, gene.attachment_slot);
                Vector2 attach = parent.transform.TransformPoint(localAttachment);
                Vector2 dir = parent.transform.TransformDirection(localAttachment.normalized).normalized;
                GameObject go = new GameObject("Limb_" + gene.innovation_id); go.transform.SetParent(transform.parent);
                Limb limb = go.AddComponent<Limb>(); int[] parentPath = paths[gene.parent_innovation_id];
                int[] path = new int[parentPath.Length + 1]; parentPath.CopyTo(path, 0); path[path.Length - 1] = gene.attachment_slot;
                limb.InitFromGene(parents[gene.parent_innovation_id], attach, dir, gene, path, identity, phenotypeScale);
                childLimbs.Add(limb); parents[gene.innovation_id] = limb.Rigidbody; paths[gene.innovation_id] = path;
                pending.RemoveAt(i); built = true;
            }
            if (!built) break; // malformed orphan gene: Python validation should already have removed it.
        }
    }

    public List<Limb> GetAllLimbs() => new List<Limb>(childLimbs);
    public void AddEnergy(float amount) { float accepted = Mathf.Min(amount, maxEnergy - energy); energy += accepted; foodGain += accepted; }

    public void DrainEnergy(List<Limb> limbs, float dt)
    {
        float movedDegrees = 0f;
        foreach (Limb limb in limbs) movedDegrees += Mathf.Abs(limb.JointSpeed) * dt;
        // Limb flailing is not locomotion.  Penalise a creature whose torso
        // remains stationary even when it is spending energy at its joints.
        bool idle = rb.linearVelocity.magnitude < .05f;
        idleSeconds = idle ? idleSeconds + dt : 0f;
        float penaltyScale = Mathf.Max(0f, EnergyPenaltyScale);
        float baseline = baselineEnergyPerSecond * dt * penaltyScale;
        float movement = movedDegrees * movementEnergyPerDegree * penaltyScale;
        float idleThisTick = idleSeconds >= idleAfterSeconds ? idleEnergyPerSecond * dt * penaltyScale : 0f;
        baselineCost += baseline; movementCost += movement; idleCost += idleThisTick;
        energy = Mathf.Max(0f, energy - baseline - movement - idleThisTick);
    }

    public Dictionary<string, float> GetGlobalInputs()
    {
        VisionReadout vision = ScanVision(); float radians = rb.rotation * Mathf.Deg2Rad;
        return new Dictionary<string, float> {
            {"energy",energy},{"position_x",transform.position.x},{"position_y",transform.position.y},
            {"velocity_x",rb.linearVelocity.x},{"velocity_y",rb.linearVelocity.y},{"angular_velocity",rb.angularVelocity},
            {"rotation_sin",Mathf.Sin(radians)},{"rotation_cos",Mathf.Cos(radians)},
            {"sees_food",vision.food ? 1f:0f},{"rel_food_x",vision.foodRel.x},{"rel_food_y",vision.foodRel.y},
            {"sees_predator",vision.predator ? 1f:0f},{"rel_predator_x",vision.predatorRel.x},{"rel_predator_y",vision.predatorRel.y},
            {"food_gain",foodGain},{"baseline_cost",baselineCost},{"movement_cost",movementCost},{"idle_cost",idleCost}
        };
    }

    // Native M1 consumes this fixed numeric layout directly. It avoids
    // constructing a string-keyed Dictionary on every control decision.
    public void FillNativeGlobalInputs(float[] output)
    {
        if (output == null || output.Length < 14) throw new ArgumentException("Native global input buffer must contain 14 values.");
        VisionReadout vision = ScanVision();
        float radians = rb.rotation * Mathf.Deg2Rad;
        output[0] = energy / 200f; output[1] = transform.position.x / 50f; output[2] = transform.position.y / 50f;
        output[3] = rb.linearVelocity.x / 20f; output[4] = rb.linearVelocity.y / 20f; output[5] = rb.angularVelocity / 360f;
        output[6] = Mathf.Sin(radians); output[7] = Mathf.Cos(radians); output[8] = vision.food ? 1f : 0f;
        output[9] = vision.foodRel.x / 20f; output[10] = vision.foodRel.y / 20f;
        output[11] = vision.predator ? 1f : 0f; output[12] = vision.predatorRel.x / 20f; output[13] = vision.predatorRel.y / 20f;
        for (int i = 14; i < output.Length; i++) output[i] = 0f;
    }

    public void DisablePhenotype()
    {
        if (rb != null) { rb.linearVelocity = Vector2.zero; rb.angularVelocity = 0f; rb.simulated = false; }
        foreach (Collider2D collider in GetComponents<Collider2D>()) collider.enabled = false;
    }
    private void OnCollisionEnter2D(Collision2D collision) { TrackGroundContact(collision.collider, true); }
    private void OnCollisionExit2D(Collision2D collision) { TrackGroundContact(collision.collider, false); }
    private void TrackGroundContact(Collider2D collider, bool entering)
    {
        if (collider == null || collider.GetComponent<GroundSurface>() == null) return;
        if (entering) groundContacts.Add(collider); else groundContacts.Remove(collider);
    }
    private struct VisionReadout { public bool food, predator; public Vector2 foodRel, predatorRel; }
    private VisionReadout ScanVision() {
        VisionReadout result = new VisionReadout(); float food = float.MaxValue, predator = float.MaxValue; Vector2 origin = transform.position;
        int hitCount = Physics2D.OverlapCircle(origin, 8f, VisionFilter, visionHits);
        for (int index = 0; index < hitCount; index++) {
            Collider2D hit = visionHits[index];
            if (hit == null) continue;
            if (hit.attachedRigidbody == rb) continue; Vector2 rel = (Vector2)hit.transform.position-origin; float dist = rel.sqrMagnitude;
            if (hit.CompareTag("Food") && dist < food) { food=dist; result.food=true; result.foodRel=rel; }
            if (hit.CompareTag("Predator") && dist < predator) { predator=dist; result.predator=true; result.predatorRel=rel; }
        }
        return result;
    }
}
