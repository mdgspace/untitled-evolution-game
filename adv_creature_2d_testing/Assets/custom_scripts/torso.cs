using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Rigidbody2D))]
[RequireComponent(typeof(SpriteRenderer))]
[RequireComponent(typeof(BoxCollider2D))]
public class Torso : MonoBehaviour
{
    [Header("Size (world units)")]
    public float minSize = 1.0f;
    public float maxSize = 2.0f;
    public Vector2 dimensions;

    [Header("Limbs")]
    public int numConnections;
    public List<Limb> childLimbs = new List<Limb>();

    [Header("Energy")]
    public float energy = 100f;
    public float baselineMetabolism = 0.05f;    // cost per tick just for existing
    public float movementCostPerDegree = 0.01f; // cost per tick per degree of commanded rotation, summed across all limbs
    private float lastEnergyCost = 0f;          // this tick's total cost, read by UpdateDopamine

    [Header("Dopamine")]
    public float dopamine = 0f;
    public int dopamineWindowSize = 5;
    public float moveThreshold = 0.5f;
    public float burstWeight = 1.0f;
    private FrameHistoryBuffer dopaminePositionMemory;
    private FrameHistoryBuffer dopamineEnergyCostMemory;

    [Header("Vision")]
    public float visionRadius = 8f;
    public LayerMask visionMask = ~0;

    public BodyPart bodyPart;

    private HashSet<int> occupiedSlots = new HashSet<int>();
    private Rigidbody2D rb;
    private SpriteRenderer sr;
    private BoxCollider2D col;

    public Rigidbody2D Rigidbody => rb;

    public void Init(CreatureIdentity identity)
    {
        rb = GetComponent<Rigidbody2D>();
        sr = GetComponent<SpriteRenderer>();
        col = GetComponent<BoxCollider2D>();

        bodyPart = gameObject.AddComponent<BodyPart>();
        bodyPart.identity = identity;

        dimensions = new Vector2(Random.Range(minSize, maxSize), Random.Range(minSize, maxSize));

        sr.sprite = BodyUtils.GetSquareSprite();
        sr.color = new Color(0.85f, 0.3f, 0.3f);
        transform.localScale = new Vector3(dimensions.x, dimensions.y, 1f);
        col.size = Vector2.one;
        rb.bodyType = RigidbodyType2D.Dynamic;

        dopaminePositionMemory = new FrameHistoryBuffer(dopamineWindowSize, 2);
        dopamineEnergyCostMemory = new FrameHistoryBuffer(dopamineWindowSize, 1);

        numConnections = Random.Range(1, 8);

        for (int i = 0; i < numConnections; i++)
        {
            if (!TryGetFreeSlot(out int slotId, out Vector2 localDir))
                break;

            Vector2 worldDir = transform.TransformDirection(localDir);
            Vector2 attachPoint = GetAttachmentPoint(localDir);

            GameObject limbGO = new GameObject("Limb_d1");
            limbGO.transform.SetParent(transform.parent);

            Limb limb = limbGO.AddComponent<Limb>();
            limb.Init(rb, attachPoint, worldDir, 1, identity);

            childLimbs.Add(limb);
        }
    }

    private bool TryGetFreeSlot(out int slotId, out Vector2 direction)
    {
        for (int i = 0; i < BodyUtils.SlotDirections.Length; i++)
        {
            if (!occupiedSlots.Contains(i))
            {
                slotId = i;
                direction = BodyUtils.SlotDirections[i];
                occupiedSlots.Add(i);
                return true;
            }
        }
        slotId = -1;
        direction = Vector2.zero;
        return false;
    }

    private Vector2 GetAttachmentPoint(Vector2 localDirection)
    {
        float approxRadius = Mathf.Max(dimensions.x, dimensions.y) / 2f;
        Vector2 worldDirection = transform.TransformDirection(localDirection);
        return (Vector2)transform.position + worldDirection * approxRadius;
    }

    public List<Limb> GetAllLimbs()
    {
        List<Limb> all = new List<Limb>();
        foreach (Limb l in childLimbs)
            all.AddRange(l.GetSubtreeLimbs());
        return all;
    }

    // ================= ENERGY =================
    // Called once per tick by CreatureBrain.UpdateEnergyAndDopamine, AFTER
    // this tick's deltas have been applied to every limb (so
    // lastAppliedDelta reflects what was actually commanded this tick),
    // and BEFORE Physics2D.Simulate() advances the world.
    public void DrainEnergy(List<Limb> limbs)
    {
        float movementCost = 0f;
        foreach (Limb limb in limbs)
            movementCost += Mathf.Abs(limb.lastAppliedDelta) * movementCostPerDegree;

        lastEnergyCost = baselineMetabolism + movementCost;
        energy -= lastEnergyCost;
        energy = Mathf.Max(energy, 0f);
        // NOTE: nothing happens when energy hits 0 yet -- death/removal
        // handling is inventory item #9, not built yet.
    }

    // ================= DOPAMINE =================
    // Formula: penalize any window where average movement falls below
    // moveThreshold outright (-1), otherwise reward speed minus a
    // quadratic penalty on bursty energy spending, tanh-clamped to
    // [-1, 1]. Matches the "explicit movement gate" version we settled on
    // earlier as the one to start with.
    public void UpdateDopamine()
    {
        dopaminePositionMemory.PushFrame(new float[] { transform.position.x, transform.position.y });
        dopamineEnergyCostMemory.PushFrame(new float[] { lastEnergyCost });

        float[] posHistory = dopaminePositionMemory.GetConcatenated();
        int frameCount = dopamineWindowSize;

        float totalDisplacement = 0f;
        for (int i = 1; i < frameCount; i++)
        {
            Vector2 prev = new Vector2(posHistory[(i - 1) * 2], posHistory[(i - 1) * 2 + 1]);
            Vector2 curr = new Vector2(posHistory[i * 2], posHistory[i * 2 + 1]);
            totalDisplacement += Vector2.Distance(prev, curr);
        }
        float avgSpeed = totalDisplacement / Mathf.Max(frameCount - 1, 1);

        if (avgSpeed < moveThreshold)
        {
            dopamine = -1f;
            return;
        }

        float[] energyHistory = dopamineEnergyCostMemory.GetConcatenated();
        float burstPenalty = 0f;
        foreach (float e in energyHistory)
            burstPenalty += e * e;
        burstPenalty /= Mathf.Max(energyHistory.Length, 1);

        dopamine = (float)System.Math.Tanh(avgSpeed - burstWeight * burstPenalty);
    }

    // ================= VISION =================
    private struct VisionReadout
    {
        public bool seesPredator, seesFood, seesPlayer, seesSameSpecies;
        public Vector2 nearestFoodRelPos;
    }

    private VisionReadout ScanVision()
    {
        VisionReadout result = new VisionReadout();
        Vector2 origin = transform.position;
        float minFoodDistSq = float.MaxValue;

        Collider2D[] hits = Physics2D.OverlapCircleAll(origin, visionRadius, visionMask);
        foreach (Collider2D hit in hits)
        {
            if (hit.attachedRigidbody == rb) continue;

            if (hit.CompareTag("Predator")) result.seesPredator = true;
            if (hit.CompareTag("Food"))
            {
                result.seesFood = true;
                Vector2 rel = (Vector2)hit.transform.position - origin;
                float distSq = rel.sqrMagnitude;
                if (distSq < minFoodDistSq)
                {
                    minFoodDistSq = distSq;
                    result.nearestFoodRelPos = rel;
                }
            }
            if (hit.CompareTag("Player")) result.seesPlayer = true;

            BodyPart part = hit.GetComponent<BodyPart>();
            if (part != null && part.identity != bodyPart.identity &&
                part.identity.speciesId == bodyPart.identity.speciesId)
            {
                result.seesSameSpecies = true;
            }
        }
        return result;
    }

    // ================= GLOBAL INPUTS =================
    public Dictionary<string, float> GetGlobalInputs()
    {
        VisionReadout vision = ScanVision();
        return new Dictionary<string, float>
        {
            { "velocity_x", rb.linearVelocity.x },
            { "velocity_y", rb.linearVelocity.y },
            { "angular_velocity", rb.angularVelocity },
            { "rotation", rb.rotation },
            { "energy", energy },
            { "dopamine", dopamine },
            { "position_x", transform.position.x },
            { "position_y", transform.position.y },
            { "sees_predator", vision.seesPredator ? 1f : 0f },
            { "sees_food", vision.seesFood ? 1f : 0f },
            { "rel_food_x", vision.seesFood ? vision.nearestFoodRelPos.x : 0f },
            { "rel_food_y", vision.seesFood ? vision.nearestFoodRelPos.y : 0f },
            { "sees_player", vision.seesPlayer ? 1f : 0f },
            { "sees_same_species", vision.seesSameSpecies ? 1f : 0f }
        };
    }
}