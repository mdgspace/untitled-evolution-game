using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Rigidbody2D))]
[RequireComponent(typeof(SpriteRenderer))]
[RequireComponent(typeof(BoxCollider2D))]
public class Torso : MonoBehaviour
{
    [Header("Size (world units)")]
    public float minSize = 1.5f;
    public float maxSize = 3.0f;
    public Vector2 dimensions;

    [Header("Limbs")]
    public int numConnections;
    public List<Limb> childLimbs = new List<Limb>();

    [Header("Global sensor state")]
    public float energy = 100f;

    [Header("Vision")]
    public float visionRadius = 8f;
    public LayerMask visionMask = ~0;   // everything by default -- narrow in the
                                          // Inspector for performance once creature
                                          // counts get large

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

    // ================= VISION =================
    // Full 360-degree detection within visionRadius -- no directional/cone
    // filtering, so there's no dependency on which way the torso happens
    // to be facing (which was an open approximation in the cone version).
    private struct VisionReadout
    {
        public bool seesPredator, seesFood, seesPlayer, seesSameSpecies;
    }

    private VisionReadout ScanVision()
    {
        VisionReadout result = new VisionReadout();
        Vector2 origin = transform.position;

        Collider2D[] hits = Physics2D.OverlapCircleAll(origin, visionRadius, visionMask);
        foreach (Collider2D hit in hits)
        {
            if (hit.attachedRigidbody == rb) continue; // skip own torso collider

            // requires "Predator"/"Food"/"Player" tags to exist in this
            // project (Project Settings -> Tags and Layers)
            if (hit.CompareTag("Predator")) result.seesPredator = true;
            if (hit.CompareTag("Food")) result.seesFood = true;
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
            { "position_x", transform.position.x },
            { "position_y", transform.position.y },
            { "sees_predator", vision.seesPredator ? 1f : 0f },
            { "sees_food", vision.seesFood ? 1f : 0f },
            { "sees_player", vision.seesPlayer ? 1f : 0f },
            { "sees_same_species", vision.seesSameSpecies ? 1f : 0f }
        };
    }
}