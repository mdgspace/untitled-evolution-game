using UnityEngine;

[RequireComponent(typeof(SpriteRenderer))]
[RequireComponent(typeof(CircleCollider2D))]
public class Food : MonoBehaviour
{
    public float energyValue = 20f;
    public bool consumed = false;

    // Kept separate from BodyPart's self/other/environment touch
    // classification -- food has no BodyPart (it isn't a creature), so
    // Limb's existing touch logic correctly reports "environment" when
    // something eats it. Energy gain lives here instead of as a 4th
    // touch category, so GetLocalInputs()'s shape doesn't need to change
    // for every consumer of it.
    private void OnCollisionEnter2D(Collision2D collision)
    {
        if (consumed) return;

        BodyPart part = collision.collider.GetComponent<BodyPart>();
        if (part == null) return; // hit the ground or something else non-creature

        if (part.identity.torso != null)
        {
            part.identity.torso.energy += energyValue;
            Consume();
        }
    }

    private void Consume()
    {
        consumed = true;
        gameObject.SetActive(false); // hidden, not destroyed -- EnvironmentSpawner.RespawnFood reuses it later
    }
}