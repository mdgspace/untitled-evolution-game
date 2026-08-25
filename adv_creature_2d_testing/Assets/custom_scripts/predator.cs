using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(SpriteRenderer), typeof(CircleCollider2D))]
public class Predator : MonoBehaviour
{
    public float patrolSpeed;
    public float chaseSpeed = .25f;
    public int killContactTicks = 3;
    private Vector2 direction;
    // Track collider counts, not just identities: a torso/limb can have several
    // colliders inside the predator at once, and one limb exiting must not break
    // an otherwise continuous contact interval.
    private readonly Dictionary<CreatureIdentity, int> overlapCounts = new Dictionary<CreatureIdentity, int>();
    private readonly Dictionary<CreatureIdentity, int> consecutive = new Dictionary<CreatureIdentity, int>();

    public void Forget(CreatureIdentity identity)
    {
        if (identity == null) return;
        overlapCounts.Remove(identity);
        consecutive.Remove(identity);
    }

    public bool IsOverlapping(CreatureIdentity identity) => identity != null && overlapCounts.TryGetValue(identity, out int count) && count > 0;

    private void Start() { direction = Random.insideUnitCircle.normalized; GetComponent<CircleCollider2D>().isTrigger = true; }
    public void PlaceAt(Vector2 position) { transform.position = position; overlapCounts.Clear(); consecutive.Clear(); }
    private void OnTriggerEnter2D(Collider2D other)
    {
        BodyPart part = other.GetComponent<BodyPart>();
        if (part == null || part.identity == null) return;
        overlapCounts.TryGetValue(part.identity, out int count);
        overlapCounts[part.identity] = count + 1;
    }
    private void OnTriggerExit2D(Collider2D other)
    {
        BodyPart part = other.GetComponent<BodyPart>();
        if (part == null || part.identity == null) return;
        if (!overlapCounts.TryGetValue(part.identity, out int count)) return;
        if (count <= 1) overlapCounts.Remove(part.identity);
        else overlapCounts[part.identity] = count - 1;
    }
    private void FixedUpdate()
    {
        CreatureIdentity nearest = null; float nearestSqr = float.PositiveInfinity;
        foreach (CreatureIdentity identity in FindObjectsByType<CreatureIdentity>(FindObjectsInactive.Exclude))
        {
            if (identity == null || identity.torso == null || identity.torso.GetComponent<CreatureBrain>()?.IsDead == true) continue;
            float distance = ((Vector2)identity.torso.transform.position - (Vector2)transform.position).sqrMagnitude;
            if (distance < nearestSqr) { nearestSqr = distance; nearest = identity; }
        }
        Vector2 velocity = nearest == null ? direction * patrolSpeed : ((Vector2)nearest.torso.transform.position - (Vector2)transform.position).normalized * chaseSpeed;
        if (velocity.sqrMagnitude > 0f) transform.position += (Vector3)(velocity * Time.fixedDeltaTime);
        foreach (CreatureIdentity identity in new List<CreatureIdentity>(overlapCounts.Keys))
            if (identity == null || identity.torso == null) { overlapCounts.Remove(identity); consecutive.Remove(identity); }
        foreach (CreatureIdentity identity in new List<CreatureIdentity>(consecutive.Keys))
            if (identity == null || !overlapCounts.ContainsKey(identity)) consecutive.Remove(identity);
        foreach (CreatureIdentity identity in new List<CreatureIdentity>(overlapCounts.Keys)) {
            consecutive.TryGetValue(identity, out int ticks); ticks++; consecutive[identity] = ticks;
            CreatureBrain brain = identity.torso == null ? null : identity.torso.GetComponent<CreatureBrain>();
            if (brain != null && ticks >= killContactTicks) brain.Kill("predator");
        }
    }
}
