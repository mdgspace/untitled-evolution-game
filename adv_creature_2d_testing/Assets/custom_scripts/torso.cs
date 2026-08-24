using System.Collections.Generic;
using UnityEngine;

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
    // At the fixed 50 Hz simulation rate, .2 seconds is ten physics ticks.
    // A stationary torso must quickly either translate or die and backfill.
    public float idleEnergyPerSecond = 20f;
    public float idleAfterSeconds = .4f;
    [Range(.1f, 1f)] public float phenotypeScale = .45f;
    public BodyPart bodyPart;
    public List<Limb> childLimbs = new List<Limb>();

    private Rigidbody2D rb;
    private float idleSeconds;
    private float foodGain;
    private float baselineCost;
    private float movementCost;
    private float idleCost;
    public Rigidbody2D Rigidbody => rb;
    public float FoodGain => foodGain;

    public void InitFromGenome(CreatureIdentity identity, BodyGenomeDto genome)
    {
        rb = GetComponent<Rigidbody2D>(); rb.bodyType = RigidbodyType2D.Dynamic;
        rb.simulated = true; rb.gravityScale = 1f; rb.constraints = RigidbodyConstraints2D.None;
        rb.sleepMode = RigidbodySleepMode2D.NeverSleep;
        rb.mass = Mathf.Clamp(genome.torso_mass, .1f, 20f);
        rb.inertia = Mathf.Clamp(genome.torso_inertia, .01f, 20f);
        dimensions = new Vector2(genome.torso_width, genome.torso_height) * phenotypeScale;
        SpriteRenderer sr = GetComponent<SpriteRenderer>(); sr.sprite = BodyUtils.GetSquareSprite(); sr.color = new Color(.85f,.3f,.3f);
        BoxCollider2D col = GetComponent<BoxCollider2D>(); col.size = Vector2.one;
        transform.localScale = new Vector3(dimensions.x, dimensions.y, 1f);
        bodyPart = gameObject.AddComponent<BodyPart>(); bodyPart.identity = identity;
        Dictionary<int, Rigidbody2D> parents = new Dictionary<int, Rigidbody2D> { { 0, rb } };
        Dictionary<int, Vector2> parentDimensions = new Dictionary<int, Vector2> { { 0, dimensions } };
        Dictionary<int, Vector2> parentPositions = new Dictionary<int, Vector2> { { 0, transform.position } };
        Dictionary<int, int[]> paths = new Dictionary<int, int[]> { { 0, new int[0] } };
        List<LimbGeneDto> pending = new List<LimbGeneDto>(genome.limbs);
        while (pending.Count > 0) {
            bool built = false;
            for (int i = pending.Count - 1; i >= 0; i--) {
                LimbGeneDto gene = pending[i];
                if (!gene.enabled || !parents.ContainsKey(gene.parent_innovation_id)) { if (!gene.enabled) pending.RemoveAt(i); continue; }
                Vector2 dir = BodyUtils.SlotDirections[Mathf.Clamp(gene.attachment_slot, 0, BodyUtils.SlotDirections.Length - 1)];
                Vector2 attach = parentPositions[gene.parent_innovation_id] + dir * Mathf.Max(parentDimensions[gene.parent_innovation_id].x, parentDimensions[gene.parent_innovation_id].y) / 2f;
                GameObject go = new GameObject("Limb_" + gene.innovation_id); go.transform.SetParent(transform.parent);
                Limb limb = go.AddComponent<Limb>(); int[] parentPath = paths[gene.parent_innovation_id];
                int[] path = new int[parentPath.Length + 1]; parentPath.CopyTo(path, 0); path[path.Length - 1] = gene.attachment_slot;
                limb.InitFromGene(parents[gene.parent_innovation_id], attach, dir, gene, path, identity, phenotypeScale);
                childLimbs.Add(limb); parents[gene.innovation_id] = limb.Rigidbody; parentDimensions[gene.innovation_id] = limb.dimensions;
                parentPositions[gene.innovation_id] = limb.transform.position; paths[gene.innovation_id] = path;
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
        float baseline = baselineEnergyPerSecond * dt;
        float movement = movedDegrees * movementEnergyPerDegree;
        float idleThisTick = idleSeconds >= idleAfterSeconds ? idleEnergyPerSecond * dt : 0f;
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

    public void DisablePhenotype()
    {
        if (rb != null) { rb.linearVelocity = Vector2.zero; rb.angularVelocity = 0f; rb.simulated = false; }
        foreach (Collider2D collider in GetComponents<Collider2D>()) collider.enabled = false;
    }
    private struct VisionReadout { public bool food, predator; public Vector2 foodRel, predatorRel; }
    private VisionReadout ScanVision() {
        VisionReadout result = new VisionReadout(); float food = float.MaxValue, predator = float.MaxValue; Vector2 origin = transform.position;
        foreach (Collider2D hit in Physics2D.OverlapCircleAll(origin, 8f)) {
            if (hit.attachedRigidbody == rb) continue; Vector2 rel = (Vector2)hit.transform.position-origin; float dist = rel.sqrMagnitude;
            if (hit.CompareTag("Food") && dist < food) { food=dist; result.food=true; result.foodRel=rel; }
            if (hit.CompareTag("Predator") && dist < predator) { predator=dist; result.predator=true; result.predatorRel=rel; }
        }
        return result;
    }
}
