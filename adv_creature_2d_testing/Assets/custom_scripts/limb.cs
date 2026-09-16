using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Rigidbody2D))]
[RequireComponent(typeof(SpriteRenderer))]
[RequireComponent(typeof(BoxCollider2D))]
[RequireComponent(typeof(HingeJoint2D))]
public class Limb : MonoBehaviour
{
    public Vector2 dimensions;
    public int innovationId;
    public int parentInnovationId;
    public int attachmentSlot;
    public int depth;
    public int[] genePath;
    public string limbId;
    public float maxMotorTorque;
    public BodyPart bodyPart;
    public float lastAppliedDelta;
    public float ExecutedIntervalDelta { get; private set; }
    public bool TouchingSelf { get; private set; }
    public bool TouchingOtherCreature { get; private set; }
    public bool TouchingEnvironment { get; private set; }

    private Rigidbody2D rb;
    private HingeJoint2D hinge;
    private readonly HashSet<Collider2D> contacts = new HashSet<Collider2D>();
    private int actionTicksRemaining;
    private float contactTangentialVelocity;

    public Rigidbody2D Rigidbody => rb;
    public float JointAngle => hinge == null ? 0f : hinge.jointAngle;
    public float JointSpeed => hinge == null ? 0f : hinge.jointSpeed;
    public Vector2 BaseWorldPosition => hinge == null ? transform.position : transform.TransformPoint(hinge.anchor);
    public Vector2 DistalWorldPosition => transform.TransformPoint(new Vector2(.5f, 0f));
    public Vector2 LocalEndpointVelocity => rb == null ? Vector2.zero : rb.GetPointVelocity(DistalWorldPosition);
    public float Mass => rb == null ? 0f : rb.mass;
    public float Inertia => rb == null ? 0f : rb.inertia;
    public float ContactTangentialVelocity => contactTangentialVelocity;
    public float MinAngle => hinge == null ? 0f : hinge.limits.min;
    public float MaxAngle => hinge == null ? 0f : hinge.limits.max;
    public bool AtJointLimit {
        get {
            if (hinge == null || !hinge.useLimits) return false;
            JointAngleLimits2D limits = hinge.limits;
            return hinge.jointAngle <= limits.min + 1f || hinge.jointAngle >= limits.max - 1f;
        }
    }

    public void InitFromGene(Rigidbody2D parent, Vector2 attachPoint, Vector2 worldDirection,
                             LimbGeneDto gene, int[] path, CreatureIdentity identity, float phenotypeScale)
    {
        rb = GetComponent<Rigidbody2D>();
        rb.simulated = true; rb.gravityScale = 1f; rb.constraints = RigidbodyConstraints2D.None;
        rb.sleepMode = RigidbodySleepMode2D.StartAwake;
        rb.interpolation = RigidbodyInterpolation2D.Interpolate; rb.collisionDetectionMode = CollisionDetectionMode2D.Continuous;
        float areaScale = phenotypeScale * phenotypeScale; float inertiaScale = areaScale * areaScale;
        rb.mass = Mathf.Clamp(gene.mass * areaScale, .02f, 10f);
        rb.inertia = Mathf.Clamp(gene.inertia * inertiaScale, .001f, 10f);
        hinge = GetComponent<HingeJoint2D>();
        innovationId = gene.innovation_id; parentInnovationId = gene.parent_innovation_id;
        attachmentSlot = gene.attachment_slot; genePath = path; depth = path.Length;
        limbId = innovationId.ToString(); dimensions = new Vector2(gene.width, gene.height) * phenotypeScale;
        maxMotorTorque = gene.max_torque * phenotypeScale;
        bodyPart = gameObject.AddComponent<BodyPart>(); bodyPart.identity = identity;
        SpriteRenderer sr = GetComponent<SpriteRenderer>(); sr.sprite = BodyUtils.GetSquareSprite();
        sr.color = Color.Lerp(new Color(.3f,.5f,.9f), new Color(.3f,.9f,.6f), depth / 5f);
        BoxCollider2D col = GetComponent<BoxCollider2D>(); col.size = Vector2.one; col.sharedMaterial = NativePhysicsMaterials.Body;
        transform.localScale = new Vector3(dimensions.x, dimensions.y, 1f);
        transform.position = attachPoint + worldDirection * dimensions.x / 2f;
        transform.rotation = Quaternion.Euler(0f, 0f, Mathf.Atan2(worldDirection.y, worldDirection.x) * Mathf.Rad2Deg);
        rb.bodyType = RigidbodyType2D.Dynamic;
        hinge.autoConfigureConnectedAnchor = false; hinge.connectedBody = parent; hinge.enableCollision = false;
        hinge.breakForce = float.PositiveInfinity; hinge.breakTorque = float.PositiveInfinity;
        hinge.anchor = new Vector2(-.5f, 0f); hinge.connectedAnchor = parent.transform.InverseTransformPoint(attachPoint);
        JointAngleLimits2D limits = hinge.limits; limits.min = gene.min_angle; limits.max = gene.max_angle;
        hinge.limits = limits; hinge.useLimits = true;
        JointMotor2D motor = hinge.motor; motor.motorSpeed = 0f; motor.maxMotorTorque = maxMotorTorque;
        hinge.motor = motor; hinge.useMotor = true;
    }

    public Dictionary<string, float> GetLocalInputs() => new Dictionary<string, float> {
        { "joint_angle", hinge.jointAngle }, { "angular_velocity", hinge.jointSpeed },
        { "touch_self", TouchingSelf ? 1f : 0f }, { "touch_other_creature", TouchingOtherCreature ? 1f : 0f },
        { "touch_environment", TouchingEnvironment ? 1f : 0f }
    };

    public void ApplyIntervalDelta(float deltaDegrees, int controlTicks, float fixedDeltaTime)
    {
        float requested = Mathf.Clamp(deltaDegrees, -NativeCreatureModel.MaximumActionDegrees, NativeCreatureModel.MaximumActionDegrees);
        if (hinge != null && hinge.useLimits)
        {
            JointAngleLimits2D limits = hinge.limits;
            requested = Mathf.Clamp(requested, limits.min - hinge.jointAngle, limits.max - hinge.jointAngle);
        }
        ExecutedIntervalDelta = requested;
        lastAppliedDelta = ExecutedIntervalDelta / Mathf.Max(1, controlTicks);
        // A command is a lease, not a one-interval pulse.  The asynchronous
        // bridge renews it with a newer learned action; a slow response must
        // not silently change a motor to zero and make a healthy creature look
        // frozen.
        actionTicksRemaining = Mathf.Max(1, controlTicks);
        if (hinge != null)
        {
            hinge.enabled = true;
            hinge.useMotor = true;
        }
        rb.WakeUp(); if (hinge != null && hinge.connectedBody != null) hinge.connectedBody.WakeUp();
        SetSpeed(lastAppliedDelta / Mathf.Max(.0001f, fixedDeltaTime));
    }

    public void AdvanceMotorTick()
    {
        if (actionTicksRemaining > 0) actionTicksRemaining--;
    }

    private void SetSpeed(float speed) { JointMotor2D motor = hinge.motor; motor.motorSpeed = speed; motor.maxMotorTorque = maxMotorTorque; hinge.motor = motor; }
    public void DisablePhenotype() {
        actionTicksRemaining = 0; lastAppliedDelta = 0f; ExecutedIntervalDelta = 0f;
        if (hinge != null) { hinge.useMotor = false; hinge.enabled = false; }
        if (rb != null) { rb.linearVelocity = Vector2.zero; rb.angularVelocity = 0f; rb.simulated = false; }
        foreach (Collider2D collider in GetComponents<Collider2D>()) collider.enabled = false;
        contacts.Clear(); contactTangentialVelocity = 0f; TouchingSelf = TouchingOtherCreature = TouchingEnvironment = false;
    }
    private void OnCollisionEnter2D(Collision2D collision) { contacts.Add(collision.collider); RecomputeTouch(); UpdateContactMotion(collision); }
    private void OnCollisionStay2D(Collision2D collision) { UpdateContactMotion(collision); }
    private void OnCollisionExit2D(Collision2D collision) { contacts.Remove(collision.collider); RecomputeTouch(); if(!TouchingEnvironment)contactTangentialVelocity=0f; }
    private void UpdateContactMotion(Collision2D collision){if(rb==null||collision==null||collision.collider==null||collision.collider.GetComponent<BodyPart>()!=null||collision.contactCount==0)return;ContactPoint2D point=collision.GetContact(0);Vector2 tangent=new Vector2(-point.normal.y,point.normal.x),otherVelocity=collision.rigidbody==null?Vector2.zero:collision.rigidbody.GetPointVelocity(point.point);contactTangentialVelocity=Vector2.Dot(rb.GetPointVelocity(point.point)-otherVelocity,tangent);}
    private void RecomputeTouch() {
        TouchingSelf = TouchingOtherCreature = TouchingEnvironment = false;
        contacts.RemoveWhere(other => other == null);
        foreach (Collider2D other in contacts) {
            BodyPart part = other.GetComponent<BodyPart>();
            if (part == null) TouchingEnvironment = true;
            else if (part.identity == bodyPart.identity) TouchingSelf = true;
            else TouchingOtherCreature = true;
        }
    }
}
