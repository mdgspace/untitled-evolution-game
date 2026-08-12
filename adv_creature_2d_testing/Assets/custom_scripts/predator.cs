using UnityEngine;

// Deliberately minimal: a tagged, optionally-drifting obstacle Vision can
// detect. No attack/damage logic -- that depends on decisions not made
// yet (energy cost? instant death? just a scare signal?), so it's item
// #11 in the inventory rather than guessed at here. isTrigger is true for
// now, meaning creatures currently pass through predators with no
// physical response and no touch-classification event either (Limb only
// listens for OnCollisionEnter2D, not OnTriggerEnter2D) -- vision
// detection is the only thing that currently works.
[RequireComponent(typeof(SpriteRenderer))]
[RequireComponent(typeof(CircleCollider2D))]
public class Predator : MonoBehaviour
{
    public float patrolSpeed = 0f; // 0 = stationary

    private Vector2 direction;

    private void Start()
    {
        direction = Random.insideUnitCircle.normalized;
    }

    private void Update()
    {
        if (patrolSpeed > 0f)
            transform.position += (Vector3)(direction * patrolSpeed * Time.deltaTime);
    }
}