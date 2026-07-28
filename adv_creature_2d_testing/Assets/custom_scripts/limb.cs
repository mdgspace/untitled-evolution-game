using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Rigidbody2D))]
[RequireComponent(typeof(SpriteRenderer))]
[RequireComponent(typeof(BoxCollider2D))]
[RequireComponent(typeof(HingeJoint2D))]
public class Limb : MonoBehaviour
{
    public const int MaxDepth = 5;

    [Header("Size (world units)")]
    public Vector2 dimensions;

    [Header("Chain info")]
    public int depth;
    public bool hasConnection;
    public Limb childLimb;

    [Header("Joint motor")]
    public float maxMotorTorque = 50f;

    public BodyPart bodyPart;
    public string limbId;   // stable id so Python can correlate deltas back to the right limb

    private Rigidbody2D rb;
    private SpriteRenderer sr;
    private BoxCollider2D col;
    private HingeJoint2D hinge;

    private HashSet<Collider2D> activeContacts = new HashSet<Collider2D>();
    public bool TouchingSelf { get; private set; }
    public bool TouchingOtherCreature { get; private set; }
    public bool TouchingEnvironment { get; private set; }

    public Rigidbody2D Rigidbody => rb;
    public HingeJoint2D Hinge => hinge;

    public void Init(Rigidbody2D parentRigidbody, Vector2 parentAttachPoint,
                      Vector2 worldDirection, int depthValue, CreatureIdentity identity)
    {
        rb = GetComponent<Rigidbody2D>();
        sr = GetComponent<SpriteRenderer>();
        col = GetComponent<BoxCollider2D>();
        hinge = GetComponent<HingeJoint2D>();

        bodyPart = gameObject.AddComponent<BodyPart>();
        bodyPart.identity = identity;
        limbId = System.Guid.NewGuid().ToString();

        depth = depthValue;
        dimensions = new Vector2(Random.Range(1.0f, 2.5f), Random.Range(0.3f, 0.6f));
        hasConnection = Random.value < 0.5f && depth < MaxDepth;

        sr.sprite = BodyUtils.GetSquareSprite();
        sr.color = Color.Lerp(new Color(0.3f, 0.5f, 0.9f), new Color(0.3f, 0.9f, 0.6f),
                               depth / (float)MaxDepth);
        transform.localScale = new Vector3(dimensions.x, dimensions.y, 1f);
        col.size = Vector2.one;
        rb.bodyType = RigidbodyType2D.Dynamic;

        float angle = Mathf.Atan2(worldDirection.y, worldDirection.x) * Mathf.Rad2Deg;
        transform.rotation = Quaternion.Euler(0f, 0f, angle);
        transform.position = parentAttachPoint + worldDirection * (dimensions.x / 2f);

        hinge.autoConfigureConnectedAnchor = false;
        hinge.connectedBody = parentRigidbody;
        hinge.anchor = new Vector2(-0.5f, 0f);
        hinge.connectedAnchor = parentRigidbody.transform.InverseTransformPoint(parentAttachPoint);
        hinge.enabled = true;

        JointMotor2D motor = hinge.motor;
        motor.motorSpeed = 0f;
        motor.maxMotorTorque = maxMotorTorque;
        hinge.motor = motor;
        hinge.useMotor = false;

        if (hasConnection)
            SpawnChild(identity);
    }

    private void SpawnChild(CreatureIdentity identity)
    {
        Vector2 tipWorld = (Vector2)transform.position + (Vector2)transform.right * (dimensions.x / 2f);
        Vector2 childDirection = transform.right;

        GameObject go = new GameObject($"Limb_d{depth + 1}");
        go.transform.SetParent(transform.parent);

        childLimb = go.AddComponent<Limb>();
        childLimb.Init(rb, tipWorld, childDirection, depth + 1, identity);
    }

    public List<Limb> GetSubtreeLimbs()
    {
        List<Limb> limbs = new List<Limb> { this };
        if (childLimb != null)
            limbs.AddRange(childLimb.GetSubtreeLimbs());
        return limbs;
    }

    // ================= TOUCH (LOCAL INPUT) =================
    private void OnCollisionEnter2D(Collision2D collision)
    {
        activeContacts.Add(collision.collider);
        RecomputeTouchState();
    }

    private void OnCollisionExit2D(Collision2D collision)
    {
        activeContacts.Remove(collision.collider);
        RecomputeTouchState();
    }

    private void RecomputeTouchState()
    {
        TouchingSelf = false;
        TouchingOtherCreature = false;
        TouchingEnvironment = false;

        foreach (Collider2D contact in activeContacts)
        {
            BodyPart otherPart = contact.GetComponent<BodyPart>();
            if (otherPart == null)
            {
                TouchingEnvironment = true;
                continue;
            }
            if (otherPart.identity == bodyPart.identity)
                TouchingSelf = true;
            else
                TouchingOtherCreature = true;
        }
    }

    // ================= LOCAL INPUTS =================
    public Dictionary<string, float> GetLocalInputs()
    {
        return new Dictionary<string, float>
        {
            { "joint_angle", hinge.jointAngle },
            { "touch_self", TouchingSelf ? 1f : 0f },
            { "touch_other_creature", TouchingOtherCreature ? 1f : 0f },
            { "touch_environment", TouchingEnvironment ? 1f : 0f }
        };
    }

    // ================= OUTPUT =================
    public void ApplyDeltaAngle(float deltaDegrees)
    {
        JointMotor2D motor = hinge.motor;
        motor.motorSpeed = deltaDegrees / Time.fixedDeltaTime;
        motor.maxMotorTorque = maxMotorTorque;
        hinge.motor = motor;
        hinge.useMotor = true;
    }
}