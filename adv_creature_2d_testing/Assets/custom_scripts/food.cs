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
