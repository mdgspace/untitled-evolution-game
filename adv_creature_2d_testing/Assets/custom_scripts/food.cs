using UnityEngine;

[RequireComponent(typeof(SpriteRenderer))]
[RequireComponent(typeof(CircleCollider2D))]
public class Food : MonoBehaviour
{
    public float energyValue = 20f;
    public float respawnDelay = 5f;
    public bool consumed = false;
    private EnvironmentSpawner environment;
    private Coroutine respawn;

    public void Init(EnvironmentSpawner owner) { environment = owner; }

    public void PlaceAt(Vector2 position)
    {
        if (respawn != null) { StopCoroutine(respawn); respawn = null; }
        consumed = false;
        transform.position = position;
        GetComponent<Collider2D>().enabled = true;
        GetComponent<SpriteRenderer>().enabled = true;
    }

    // Kept separate from BodyPart's self/other/environment touch
    // classification -- food has no BodyPart (it isn't a creature), so
    // Limb's existing touch logic correctly reports "environment" when
    // something eats it. Energy gain lives here instead of as a 4th
    // touch category, so GetLocalInputs()'s shape doesn't need to change
    // for every consumer of it.
    private void OnTriggerEnter2D(Collider2D collision)
    {
        if (consumed) return;

        BodyPart part = collision.GetComponent<BodyPart>();
        if (part == null) return; // hit the ground or something else non-creature

        if (part.identity.torso != null)
        {
            part.identity.torso.AddEnergy(energyValue);
            Consume();
        }
    }

    private void Consume()
    {
        consumed = true;
        GetComponent<Collider2D>().enabled = false;
        GetComponent<SpriteRenderer>().enabled = false;
        if (respawn == null) respawn = StartCoroutine(RespawnAfterDelay());
    }

    private System.Collections.IEnumerator RespawnAfterDelay()
    {
        yield return new WaitForSeconds(respawnDelay);
        if (environment != null) environment.RespawnFood(this);
        respawn = null;
    }
}

[RequireComponent(typeof(SpriteRenderer), typeof(CircleCollider2D))]
public sealed class HighEnergyFood : MonoBehaviour
{
    public float energyValue = 150f;
    public float escapeSpeed = 3.5f;
    public float surfaceY = -3.34f;
    public bool consumed;

    public void PlaceAt(Vector2 position)
    {
        consumed = false;
        transform.position = new Vector2(position.x, surfaceY);
        GetComponent<Collider2D>().enabled = true;
        GetComponent<SpriteRenderer>().enabled = true;
    }

    private void FixedUpdate()
    {
        if (consumed) return;
        CreatureIdentity nearest = null; float nearestSqr = float.PositiveInfinity;
        foreach (CreatureIdentity identity in FindObjectsByType<CreatureIdentity>(FindObjectsInactive.Exclude))
        {
            if (identity == null || identity.torso == null || identity.torso.GetComponent<CreatureBrain>()?.IsDead == true) continue;
            float distance = ((Vector2)identity.torso.transform.position - (Vector2)transform.position).sqrMagnitude;
            if (distance < nearestSqr) { nearestSqr = distance; nearest = identity; }
        }
        if (nearest != null)
        {
            float awayX = transform.position.x - nearest.torso.transform.position.x;
            if (Mathf.Abs(awayX) > .001f)
            {
                float direction = Mathf.Sign(awayX);
                transform.position = new Vector2(transform.position.x + direction * escapeSpeed * Time.fixedDeltaTime, surfaceY);
            }
            else transform.position = new Vector2(transform.position.x, surfaceY);
        }
    }

    private void OnTriggerEnter2D(Collider2D collision)
    {
        if (consumed) return;
        BodyPart part = collision.GetComponent<BodyPart>();
        if (part == null || part.identity == null || part.identity.torso == null) return;
        part.identity.torso.AddEnergy(energyValue);
        consumed = true;
        GetComponent<Collider2D>().enabled = false;
        GetComponent<SpriteRenderer>().enabled = false;
    }
}
